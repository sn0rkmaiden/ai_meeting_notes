import datetime
import shutil

import gradio as gr

import diarization


def transcribe_audio(filepath):
    """
    Обрабатывает аудиофайл
    :param filepath: Путь к аудиофайлу
    :return: Пути к транскрипциям аудиофайла в markdown, простом json и json Label Studio
    """

    phrases = diarizer.diarize(filepath)

    time = datetime.datetime.now().isoformat()
    plain = f"transcription_{time}.json"
    with open(plain, "w") as f:
        f.write(diarization.phrases_to_json(phrases))

    return [diarization.phrases_to_markdown(phrases), plain]


def handle_audio(audio_path):
    if audio_path is None:
        return "Аудио не записано."

    new_path = f"recorded_{datetime.datetime.now().isoformat()}.wav"
    shutil.move(audio_path, new_path)

    return transcribe_audio(new_path)


diarizer = diarization.Diarizer()

with gr.Blocks() as server:
    gr.Markdown("Запись аудио → Обработка → Вывод текста")

    with gr.Row():
        audio_input = gr.Audio(label="Запишите аудио", type="filepath")
        with gr.Column():
            text_output = gr.Textbox(label="Результат")
            download = gr.DownloadButton(label="Скачать")

    record_btn = gr.Button("Отправить")

    record_btn.click(fn=handle_audio, inputs=audio_input, outputs=[text_output, download])

server.launch(share=True)
