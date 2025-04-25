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

sampling_rate = 16000


def load_audio(file, target_sample_rate=sampling_rate, **kwargs):
    tensor, sample_rate = torchaudio.load(file)
    tensor /= tensor.abs().max()
    return audio_func.resample(tensor, sample_rate, target_sample_rate).reshape((-1)).numpy()


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

    def diarize(self, audio: str | Path):
        audio = load_audio(audio)
        result = self.transcriber.transcribe(audio, batch_size=self.batch_size, language=self.language_code)
        result = whisperx.align(result["segments"], self.aligner, self.dictionary, audio, self.device,
                                return_char_alignments=False)
        diarization = self.diarizer(audio)
        result = whisperx.assign_word_speakers(diarization, result, fill_nearest=True)
        segments = result["segments"]
        for segment in segments:
            del segment["words"]
        del audio, result, diarization
        gc.collect()
        torch.cuda.empty_cache()
        return segments


def diarization_to_json(objects):
    return json.dumps(objects, ensure_ascii=False)


def diarization_to_label_studio_json(objects, audio_path):
    result = []
    speakers = dict()
    for id, annotation in enumerate(objects):
        speaker = annotation["speaker"]
        if speaker not in speakers:
            speakers[speaker] = len(speakers) + 1
        result += [
            {"value": {"start": annotation["start"], "end": annotation["end"], "labels": [f"Speaker {speakers[speaker]}"]},
             "from_name": "labels",
             "to_name": "audio",
             "type": "labels",
             "id": str(id)},
            {"value": {"start": annotation["start"], "end": annotation["end"], "text": [annotation["text"]]},
             "from_name": "transcription",
             "to_name": "audio",
             "type": "textarea",
             "id": str(id)}]
    return json.dumps([{"data": {"audio": audio_path}, "id": 1, "predictions": [{"result": result}]}], ensure_ascii=False)


if __name__ == "__main__":
    infiles = argv[1:]
    bar = tqdm(infiles)
    diarizer = Diarizer()
    for infile in bar:
        bar.set_postfix(current_file=infile)
        outfile = infile + ".json"
        segments = diarizer.diarize(infile)
        with open(outfile, "w") as out:
            out.write(diarization_to_label_studio_json(segments, infile))
