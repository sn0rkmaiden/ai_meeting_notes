import datetime
import shutil
import gradio as gr
import diarization
import summarize


def transcribe_audio(filepath, timestamp):
    """
    Обрабатывает аудиофайл
    :param timestamp: Время записи
    :param filepath: Путь к аудиофайлу
    :return: Пути к транскрипциям аудиофайла в markdown, простом json и json Label Studio
    """

    phrases = diarizer.diarize(filepath)

    plain = f"transcript_{timestamp}.json"
    with open(plain, "w") as f:
        f.write(diarization.phrases_to_json(phrases))

    return [diarization.phrases_to_markdown(phrases), plain]


def handle_audio(audio_path):
    if audio_path is None:
        return "Аудио не записано."

    timestamp = datetime.datetime.now().isoformat()
    new_path = f"recorded_{timestamp}"
    if "." in audio_path:
        new_path += audio_path[audio_path.rfind("."):]
    shutil.move(audio_path, new_path)

    return transcribe_audio(new_path, timestamp)


def summarize_text(text):
    return llm.summarize(text)


diarizer = diarization.Diarizer()

llm = summarize.Summarizer()

with gr.Blocks() as server:
    gr.Markdown("Запись аудио → Обработка → Вывод текста")

    with gr.Row():
        audio_input = gr.Audio(label="Запишите аудио", type="filepath")
        with gr.Column():
            text_output = gr.Textbox(label="Результат")
            sum_output = gr.Textbox(label="Суммаризация")
            download = gr.DownloadButton(label="Скачать")

    record_btn = gr.Button("Отправить")

    summarize_btn = gr.Button("Суммаризовать")

    record_btn.click(fn=handle_audio, inputs=audio_input, outputs=[text_output, download])

    summarize_btn.click(fn=summarize_text, inputs=text_output, outputs=[sum_output])

server.launch(share=True)
