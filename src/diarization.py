import gc
import json

import torch.cuda
import whisperx
import os
from pathlib import Path
import torchaudio
from torchaudio import functional as audio_func
from sys import argv
from tqdm import tqdm
from dataclasses import dataclass
import math

sampling_rate = 16000


def load_audio(file, target_sample_rate=sampling_rate):
    tensor, sample_rate = torchaudio.load(file)

    # torchaudio.load does not downmix audio
    tensor = tensor.mean(dim=0, keepdim=True)

    tensor /= tensor.abs().max()
    return audio_func.resample(tensor, sample_rate, target_sample_rate).reshape((-1)).numpy()


@dataclass
class Phrase:
    start: float
    end: float
    text: str
    speaker: str

    @staticmethod
    def from_segment(segment: dict):
        return Phrase(segment["start"], segment["end"], segment["text"], segment["speaker"])


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


class Diarizer:
    def __init__(self,
                 device="cuda",
                 batch_size=1,
                 compute_type="int8",
                 whisper_arch="large-v3-turbo",
                 language_code="ru"):
        self.transcriber = whisperx.load_model(whisper_arch, device, compute_type=compute_type)
        self.aligner, self.dictionary = whisperx.load_align_model(language_code=language_code, device=device)
        self.diarizer = whisperx.DiarizationPipeline(use_auth_token=os.environ["HUGGING_FACE_TOKEN"], device=device)
        self.batch_size = batch_size
        self.device = device
        self.language_code = language_code

    def diarize(self, audio: str | Path) -> list[Phrase]:
        audio = load_audio(audio)
        result = self.transcriber.transcribe(audio, batch_size=self.batch_size, language=self.language_code)
        result = whisperx.align(result["segments"], self.aligner, self.dictionary, audio, self.device,
                                return_char_alignments=False)
        diarization = self.diarizer(audio)
        result = whisperx.assign_word_speakers(diarization, result, fill_nearest=True)
        phrases = [Phrase.from_segment(segment) for segment in result["segments"]]

        del audio, result, diarization
        gc.collect()
        torch.cuda.empty_cache()

        return phrases


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
