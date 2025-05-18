from sys import argv

import jiwer

from diarization import Diarizer, label_studio_json_to_phrases, Phrase, phrases_to_markdown
import glob
from pyannote.metrics.diarization import JaccardErrorRate, DiarizationErrorRate
from pyannote.core import Annotation, Segment
from jiwer import wer
from pathlib import Path
from tqdm.autonotebook import tqdm


def phrases_to_annotation(phrases: list[Phrase]) -> Annotation:
    result = Annotation()
    for phrase in phrases:
        result[Segment(phrase.start, phrase.end)] = phrase.speaker
    return result


def concat_phrases(phrases: list[Phrase]) -> str:
    return " ".join(phrase.text for phrase in sorted(phrases, key=lambda p: p.start))


def evaluate_diarization(root: str):
    json_files = glob.glob(root + "/**/*.json", recursive=True)
    diarizer = Diarizer(whisper_arch="large-v3-turbo", alignment="whisperx", beam_size=5)

    to_evaluate = {}

    for file in json_files:
        p = Path(file)
        for rel, res in label_studio_json_to_phrases(file).items():
            to_evaluate[str(p.parent / rel)] = res

    results = {}

    for file in tqdm(to_evaluate.keys()):
        results[file] = diarizer.diarize(file)

    jer = JaccardErrorRate()
    der = DiarizationErrorRate()
    der_no_miss = DiarizationErrorRate(miss=0.0)

    total_jer = 0
    total_der = 0
    total_der_no_miss = 0

    total_length = sum(l for _, l in to_evaluate.values())

    for (expected, length), (got_annotation, got) in zip(*[to_evaluate.values(), (results[k] for k in to_evaluate.keys())]):
        expected_annotation = phrases_to_annotation(expected)

        total_jer += jer(expected_annotation, got_annotation) * length / total_length
        total_der += der(expected_annotation, got_annotation) * length / total_length
        total_der_no_miss += der_no_miss(expected_annotation, got_annotation) * length / total_length

    print("JER", total_jer)
    print("DER", total_der)
    print("DER no miss", total_der)

    expected_text = [concat_phrases(phrases) for phrases, _ in to_evaluate.values()]
    got_text = [concat_phrases(results[k][1]) for k in to_evaluate.keys()]

    for md1, md2 in zip(expected_text, got_text):
        print(md1)
        print(md2)

    tr = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.ReduceToListOfListOfWords()
    ])
    print("WER", wer(expected_text, got_text, reference_transform=tr, hypothesis_transform=tr))



def main():
    match argv[1]:
        case "diarization":
            evaluate_diarization(argv[2])
        case _:
            raise RuntimeError(f"{argv[1]} is not supported.")


if __name__ == "__main__":
    main()
