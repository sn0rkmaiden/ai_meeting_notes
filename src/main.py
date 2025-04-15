import gradio as gr
import datetime
import shutil
import diarization
import json

diarizer = diarization.Diarizer()

def clean_text(text: str):
    return text.strip("«»‘’'\" ")

def join_segments(segments: list, grace_time: float = 3.0):
    if len(segments) == 0:
        return []

    buf = clean_text(segments[0]["text"])
    prev_end = segments[0]["end"]
    prev_speaker = segments[0]["speaker"]

    result = []

    for segment in segments:
        if prev_end + grace_time < segment["start"] or prev_speaker != segment["speaker"]:
            result += [{"text": buf, "speaker": prev_speaker}]
            buf = clean_text(segment["text"])
            prev_speaker = segment["speaker"]
        else:
            buf += " " + clean_text(segment["text"])
        prev_end = segment["end"]

    result += [{"text": buf, "speaker": prev_speaker}]

    return result

def segments_to_markdown(segments: list) -> str:
    """
    Преобразует сегменты аудио в Markdown для рендера
    :param segments:
    :return: str
    """

    clean_segments = join_segments(segments)

    return "\n".join((f"# {segment["speaker"]}\n{segment["text"]}" for segment in clean_segments))

# функция которая выведет результат нейронки в
def transcribe_audio(filepath):
    segments = diarizer.diarize(filepath)

    transcription_path = f"transcription_{datetime.datetime.now().isoformat()}.json"
    with open(transcription_path, "w") as f:
        json.dump(segments, f)

    return [segments_to_markdown(segments), transcription_path]


def handle_audio(audio_path):
    if audio_path is None:
        return "Аудио не записано."

    new_path = f"recorded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    shutil.copy(audio_path, new_path)

    result_text = transcribe_audio(new_path)

    return result_text


with gr.Blocks() as demo:
    gr.Markdown("Запись аудио → Обработка → Вывод текста")

    with gr.Row():
        audio_input = gr.Audio(label="Запишите аудио", type="filepath")
        with gr.Column():
            text_output = gr.Textbox(label="Результат")
            download_output = gr.DownloadButton(label="Скачать")

    record_btn = gr.Button("Отправить")

    record_btn.click(fn=handle_audio, inputs=audio_input, outputs=[text_output, download_output])

demo.launch()
