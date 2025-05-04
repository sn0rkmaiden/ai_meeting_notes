import gc
import json

import torch.cuda
import whisperx
from pathlib import Path
from sys import argv
from tqdm import tqdm
from dataclasses import dataclass
import math
from omegaconf import OmegaConf
from nemo.collections.asr.models import NeuralDiarizer
import whisper_timestamped as whisper
import pandas as pd
from datetime import timedelta
from subprocess import run


@dataclass
class Phrase:
    start: float
    end: float
    text: str
    speaker: str

    @staticmethod
    def from_segment(segment: dict):
        return Phrase(segment["start"], segment["end"], segment["text"], segment.get("speaker", "NONE"))


def clean_text(text: str):
    return text.strip("«»‘’'\" ")


def join_phrases(phrases: list[Phrase], grace_time: float = 3.0):
    """
    Склеивает близкие фразы
    :param phrases:
    :param grace_time:
    :return:
    """

    if len(phrases) == 0:
        return []

    curr_text = clean_text(phrases[0].text)
    curr_start = phrases[0].start
    curr_end = phrases[0].end
    curr_speaker = phrases[0].speaker

    phrases = phrases + [Phrase(math.inf, math.inf, "", "")]

    result = []

    for phrase in phrases[1:]:
        if phrase.speaker != curr_speaker or phrase.start > curr_end + grace_time:
            result += [Phrase(curr_start, curr_end, curr_text, curr_speaker)]
            curr_text = clean_text(phrase.text)
            curr_speaker = phrase.speaker
            curr_start, curr_end = phrase.start, phrase.end
        else:
            curr_text += " " + clean_text(phrase.text)
            curr_end = phrase.end

    return result


def test_join_phrases():
    initial = [Phrase(4, 5, "start", "speaker1"),
               Phrase(8.1, 10, "«mid»", "speaker1"),
               Phrase(11, 12, "end", "speaker1"),
               Phrase(13, 14, "end2", "speaker3")]
    expected = [Phrase(4, 5, "start", "speaker1"), Phrase(8.1, 12, "mid end", "speaker1"),
                Phrase(13, 14, "end2", "speaker3")]
    assert join_phrases(initial, 3) == expected


def phrases_to_markdown(phrases: list[Phrase]) -> str:
    """
    Преобразует список фраз в Markdown формат.
    """

    clean_phrases = join_phrases(phrases, grace_time=3.0)
    return "\n".join((f"# {phrase.speaker}\n{phrase.text}" for phrase in clean_phrases))


def load_diarization(filename):
    records = []
    with open(filename) as f:
        for num, line in enumerate(f.readlines()):
            _, _, _, start, duration, _, _, speaker, _, _ = line.split()
            start, duration = float(start), float(duration)
            records += [(f"[{timedelta(seconds=start)} -> {timedelta(seconds=start + duration)}]",
                         str(num),
                         speaker,
                         start,
                         start + duration)]
    return pd.DataFrame.from_records(records, columns=["segment", "label", "speaker", "start", "end"])


def clean_run(model, runnable):
    """
    Enables machines with very low VRAM to run this script.
    :param model: the model to upload to CUDA
    :param runnable: the action to perform on model
    :return: the result of runnable
    """

    model.to("cuda")
    result = runnable(model)
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    return result


def to_whisperx_aligned_transcript(result):
    segments = []
    word_segments = []

    for segment in result["segments"]:
        if not segment["text"] or segment["text"].isspace():
            continue
        words = []
        for word in segment.get("words", []):
            words += [{"word": word["text"], "start": word["start"], "end": word["end"], "score": word["confidence"]}]
        word_segments += words
        segments += [{"start": segment["start"], "end": segment["end"], "text": segment["text"], "words": words}]

    return {"segments": segments, "word_segments": word_segments}


class Diarizer:
    def __init__(self,
                 whisper_arch="openai/whisper-large-v3-turbo",
                 language_code="ru",
                 model_config="../src/nemo-config.yaml",
                 input_manifest_file="input_manifest.json"):
        config = OmegaConf.load(model_config)
        self.diarizer = NeuralDiarizer(config).eval()

        self.transcriber = whisper.load_model(whisper_arch, device="cpu", in_memory=True).eval()

        self.input_manifest_file = input_manifest_file
        self.language_code = language_code
        self.sample_rate = config.sample_rate

    def prepare_audio(self, audio_file: Path):
        out_file = audio_file.name + ".wav"

        run(["ffmpeg", "-i", audio_file, "-y", "-ac", "1", "-r", str(self.sample_rate), out_file]).check_returncode()

        meta = {
            'audio_filepath': out_file,
            'offset': 0,
            'label': 'infer',
            'duration': None,
            'rttm_filepath': None
        }
        with open(self.input_manifest_file, "w") as file:
            json.dump(meta, file)
            file.write("\n")

    def diarize(self, audio: str) -> list[Phrase]:
        with torch.no_grad():
            audio_path = Path(audio)
            self.prepare_audio(audio_path)

            clean_run(self.diarizer, lambda d: d.diarize())
            diarization = load_diarization(Path("pred_rttms") / (audio_path.name + ".rttm"))

            voice = list(diarization[["start", "end"]].itertuples(index=False, name=None))

            audio = whisper.load_audio(audio)
            result = clean_run(self.transcriber,
                               lambda t: whisper.transcribe_timestamped(t, audio, language=self.language_code,
                                                                        vad=voice))
            del audio

            aligned_transcript = to_whisperx_aligned_transcript(result)
        result = whisperx.assign_word_speakers(diarization, aligned_transcript, fill_nearest=False)
        return [Phrase.from_segment(segment) for segment in result["segments"]]


def phrases_to_json(phrases: list[Phrase]) -> str:
    return json.dumps(
        [{"text": phrase.text, "start": phrase.start, "end": phrase.end, "speaker": phrase.speaker} for phrase in
         phrases], ensure_ascii=False)


def phrases_to_label_studio_json(phrases: list[Phrase], audio_path) -> str:
    result = []
    speakers = dict()
    for id, annotation in enumerate(phrases):
        speaker = annotation.speaker
        if speaker not in speakers:
            speakers[speaker] = len(speakers) + 1
        result += [
            {"value": {"start": annotation.start, "end": annotation.end, "labels": [f"Speaker {speakers[speaker]}"]},
             "from_name": "labels",
             "to_name": "audio",
             "type": "labels",
             "id": str(id)},
            {"value": {"start": annotation.start, "end": annotation.end, "text": [annotation.text]},
             "from_name": "transcription",
             "to_name": "audio",
             "type": "textarea",
             "id": str(id)}]
    return json.dumps([{"data": {"audio": audio_path}, "id": 1, "predictions": [{"result": result}]}],
                      ensure_ascii=False)


if __name__ == "__main__":
    infiles = argv[1:]
    bar = tqdm(infiles)
    diarizer = Diarizer()
    for infile in bar:
        bar.set_postfix(current_file=infile)
        outfile = infile + ".json"
        segments = diarizer.diarize(infile)
        with open(outfile, "w") as out:
            out.write(phrases_to_label_studio_json(segments, infile))
