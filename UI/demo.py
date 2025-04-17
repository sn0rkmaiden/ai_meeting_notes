import gradio as gr
import datetime
import shutil

# функция которая выведет результат нейронки в
def transcribe_audio(filepath):
    return f"Обработка файла: {filepath} — завершена успешна."

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
        text_output = gr.Textbox(label="Результат")

    record_btn = gr.Button("Отправить")

    record_btn.click(fn=handle_audio, inputs=audio_input, outputs=text_output)

demo.launch()