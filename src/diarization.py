import gc
import json

from typing import Optional, Literal

import pyannote.core
import torch.cuda
import whisperx as wx
import whisper_timestamped as td
from pathlib import Path
from sys import argv
from tqdm import tqdm
from dataclasses import dataclass
import math
from omegaconf import OmegaConf
from nemo.collections.asr.models import NeuralDiarizer
import pandas as pd
from datetime import timedelta
from subprocess import run
import re
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature, SlidingWindow


@dataclass
class Phrase:
    start: float
    end: float
    text: str
    speaker: str

    @staticmethod
    def from_segment(segment: dict, speaker_format="Speaker {}"):
        if "speaker" in segment:
            speaker = speaker_format.format(re.match("speaker_(\\d+)", segment["speaker"]).group(1))
        else:
            speaker = "None"
        return Phrase(segment["start"], segment["end"], segment["text"], speaker)


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
                 # Maybe try dvislobokov/whisper-large-v3-turbo-russian later
                 whisper_arch: str = "large-v3-turbo",
                 alignment: Literal["timestamped", "whisperx"] = "timestamped",
                 language_code: str = "ru",
                 model_config: str = "../src/nemo-config.yaml",
                 input_manifest_file: str = "input_manifest.json",
                 speaker_format: str = "Speaker {}",
                 beam_size: Optional[int] = None,
                 dtype: str = "float32",
                 batch_size: int = 1):
        self.config = OmegaConf.load(model_config)
        self.diarizer = NeuralDiarizer(self.config).eval()

        self.alignment = alignment
        self.whisper_arch = whisper_arch

        match alignment:
            case "timestamped":
                self.transcriber = td.load_model(whisper_arch, device="cpu", in_memory=True).eval()
            case "whisperx":
                self.aligner, self.meta = wx.load_align_model(language_code, "cpu")

        self.input_manifest_file = input_manifest_file
        self.language_code = language_code
        self.sample_rate = self.config.sample_rate

        self.speaker_format = speaker_format
        self.beam_size = beam_size
        self.dtype = dtype
        self.batch_size = batch_size

    def prepare_audio(self, audio_file: Path):
        out_file = audio_file.name + ".wav"

        run(["ffmpeg",
             "-i", audio_file,
             "-y",
             # "-f", "s16le",
             "-ac", "1",
             "-ar", str(self.sample_rate),
             # "-acodec", "pcm_s16le",
             out_file]).check_returncode()

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

        return out_file

    def load_vad_frame(self, filename):
        with open(filename) as fd:
            lines = [float(line) for line in fd.readlines()]
        step = self.config.diarizer.vad.parameters.shift_length_in_sec
        window = self.config.diarizer.vad.parameters.window_length_in_sec
        return SlidingWindowFeature(np.array(lines).reshape((-1, 1)), SlidingWindow(window, step))

    def diarize(self, audio: str) -> tuple[Annotation, list[Phrase]]:
        with torch.no_grad():
            source_audio = Path(audio)
            wav_audio = self.prepare_audio(source_audio)

            with open(wav_audio, "rb") as fd:
                audio = np.frombuffer(fd.read(), np.int16).flatten()[22:].astype(np.float32) / 32768.0
                print(audio.max())

            clean_run(self.diarizer, lambda d: d.diarize())
            diarization = load_diarization(Path("pred_rttms") / (source_audio.name + ".rttm"))

            match self.alignment:
                case "timestamped":
                    voice = list(diarization[["start", "end"]].itertuples(index=False, name=None))
                    options = dict(language=self.language_code, beam_size=self.beam_size)
                    result = clean_run(self.transcriber,
                                       lambda t: td.transcribe_timestamped(t, audio, vad=voice, **options))
                    aligned_transcript = to_whisperx_aligned_transcript(result)
                case "whisperx":
                    options = dict(language=self.language_code)
                    # vad_frame = self.load_vad_frame(Path("vad_outputs") / (source_audio.name + ".frame"))
                    # vad_params = dict(vad_onset=self.config.diarizer.vad.parameters.onset,
                    #                   vad_offset=self.config.diarizer.vad.parameters.offset)
                    transcriber = wx.load_model(self.whisper_arch,
                                                device="cuda",
                                                compute_type=self.dtype,
                                                asr_options=dict(beam_size=self.beam_size),
                                                # vad_model=lambda ignored: vad_frame,
                                                # vad_options=vad_params,
                                                language=self.language_code)
                    result = transcriber.transcribe(audio, language=self.language_code)
                    del transcriber
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(torch.cuda.memory_summary())
                    aligned_transcript = clean_run(self.aligner,
                                                   lambda t: wx.align(result["segments"], t, self.meta, audio, "cuda"))

        result = wx.assign_word_speakers(diarization, aligned_transcript, fill_nearest=False)

        annotations = Annotation()
        for ind, row in diarization.iterrows():
            annotations[Segment(row["start"], row["end"])] = row["speaker"]

        return annotations, [Phrase.from_segment(segment, self.speaker_format) for segment in result["segments"]]


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


def label_studio_json_to_phrases(path) -> dict[str, tuple[list[Phrase], float]]:
    result = {}

    with open(path) as fd:
        data = json.load(fd)

    for obj in data:
        audio_path = obj["data"]["audio"]

        time = {}
        texts = {}
        speakers = {}
        original_length = 0

        for annotation in obj["annotations"][0]["result"]:
            value = annotation["value"]
            original_length = annotation["original_length"]
            match annotation["from_name"]:
                case "labels":
                    time[annotation["id"]] = (value["start"], value["end"])
                    speakers[annotation["id"]] = value["labels"][0]
                case "transcription":
                    time[annotation["id"]] = (value["start"], value["end"])
                    texts[annotation["id"]] = re.sub(" +", " ", " ".join(value["text"]))

        phrases = []

        for (start, end), text, speaker in zip(*([d[key] for key in time.keys()] for d in [time, texts, speakers])):
            phrases += [Phrase(start, end, text, speaker)]

        result[audio_path] = (phrases, original_length)

    return result


def main():
    infiles = argv[1:]
    bar = tqdm(infiles)
    diarizer = Diarizer()
    for infile in bar:
        bar.set_postfix(current_file=infile)
        outfile = infile + ".json"
        segments = diarizer.diarize(infile)
        with open(outfile, "w") as out:
            out.write(phrases_to_label_studio_json(segments, infile))


if __name__ == "__main__":
    main()
