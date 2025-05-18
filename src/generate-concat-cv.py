import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as audio_func
from pyannote.audio import Pipeline
import numpy as np

random.seed(1337)
np.random.seed(1337)

hf_token = os.environ["HUGGING_FACE_TOKEN"]

default_sample_rate = 16000
min_upvote_count = 2
output_path = Path("data/dev/concat-cv")
output_count = 10
clients_per_output = 5
output_sample_count = 100

output_path.mkdir(parents=True, exist_ok=True)

dataset_base_path = Path("data/cv-corpus-21.0-2025-03-14/ru/")
clips_path = dataset_base_path / "clips"
test_csv_path = dataset_base_path / "dev.tsv"
invalidated_csv_path = dataset_base_path / "invalidated.tsv"

files = pd.read_csv(test_csv_path.absolute(), sep="\t")

invalidated = pd.read_csv(invalidated_csv_path, sep="\t")
invalidated_ids = set(invalidated["client_id"])
files = files[~files["client_id"].isin(invalidated_ids)]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    device = torch.device("cpu")
    print("No GPU available. Training will run on CPU.")

vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=hf_token).to(device)
initial_params = {"onset": 0.4, "offset": 0.3, "min_duration_on": 0.0, "min_duration_off": 0.1}
vad.instantiate(initial_params)


def load_audio(file, target_sample_rate=default_sample_rate, **kwargs):
    tensor, sample_rate = torchaudio.load(file)
    tensor /= tensor.abs().max()
    return audio_func.resample(tensor, sample_rate, target_sample_rate)


def measure_sample(tensor, sample_rate=default_sample_rate):
    activity = vad({"waveform": tensor, "sample_rate": sample_rate})
    if len(activity.get_timeline()) == 0:
        return None
    start = activity.get_timeline()[0].start
    end = activity.get_timeline()[-1].end
    return start, end


def concat_and_save(index, samples):
    audios = []
    metas = []
    prev_end = 0

    for idx, row in samples.iterrows():
        audio = load_audio(clips_path / row["path"])
        timings = measure_sample(audio.to(device))
        if timings is None:
            print(f"{index} has no voice detected. Skipping")
            continue
        audios += [audio]
        metas += [(row["client_id"][:6], row["sentence"], timings[0] + prev_end / default_sample_rate,
                   timings[1] + prev_end / default_sample_rate)]
        prev_end += audio.size(1)

    concat_audio = torch.concat(audios, 1)
    rel_path = f"clips/{index}.wav"
    abs_path = output_path / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(abs_path, concat_audio, default_sample_rate)

    result = []
    for id, (speaker, text, start, end) in enumerate(metas):
        result += [
            {"value": {"start": start, "end": end, "labels": [f"Speaker {speaker}"]},
             "original_length": prev_end / default_sample_rate,
             "from_name": "labels",
             "to_name": "audio",
             "type": "labels",
             "id": str(id)},
            {"value": {"start": start, "end": end, "text": [text]},
             "original_length": prev_end / default_sample_rate,
             "from_name": "transcription",
             "to_name": "audio",
             "type": "textarea",
             "id": str(id)}]

    return {"data": {"audio": rel_path}, "id": 1, "annotations": [{"result": result}]}


metas = []

for i in range(output_count):
    print(f"Creating file number {i}")

    clients = files["client_id"].sample(clients_per_output)

    rows = files[files["client_id"].isin(clients)]
    rows = rows.sample(min(len(rows), output_sample_count))

    metas += [concat_and_save(i, rows)]

json.dump(metas, open(output_path / "labels.json", "w"), ensure_ascii=False)
