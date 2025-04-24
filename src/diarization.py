import gc
import torch.cuda
import whisperx
import os
from pathlib import Path
import torchaudio
from torchaudio import functional as audio_func

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

    def diarize(self, audio):
        audio = load_audio(audio)
        result = self.transcriber.transcribe(audio, batch_size=self.batch_size, language=self.language_code)
        result = whisperx.align(result["segments"], self.aligner, self.dictionary, audio, self.device, return_char_alignments=False)
        diarization = self.diarizer(audio)
        result = whisperx.assign_word_speakers(diarization, result, fill_nearest=True)
        segments = result["segments"]
        for segment in segments:
            del segment["words"]
        del audio, result, diarization
        gc.collect()
        torch.cuda.empty_cache()
        return segments


if __name__ == "__main__":
    audio_file = Path("../data/dev/concat-cv/clips/0.wav").resolve()
    diarizer = Diarizer()
    segments = diarizer.diarize(audio_file)
    print(segments)