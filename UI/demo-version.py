import gradio as gr
import datetime
import shutil
import os
import time
import asyncio
from typing import Optional


async def transcribe_audio(filepath: str) -> tuple[str, str, str]:

    base_text = {
        "recorded":  "тут должен быть ваш транскрипт",
        "uploaded": "Вот так вот"}

    file_type = "uploaded" if "upload" in filepath.lower() else "recorded"
    full_transcript = base_text[file_type] + "\n\n" + datetime.datetime.now().strftime(
        'Транскрипция создана: %Y-%m-%d %H:%M:%S')

    extracted_phrases = (
        "1. тут\n"
        "2. должны быть\n"
        "3. ключевые фразы"
    )

    summary = "Здесь суммаризация"

    return full_transcript, extracted_phrases, summary


async def handle_audio(audio_record: Optional[str], audio_upload: Optional[str], progress=gr.Progress()) -> tuple[
    str, str, str]:
    audio_path = audio_upload if audio_upload else audio_record

    if audio_path is None:
        return "Аудио не загружено", "Аудио не загружено", "Аудио не загружено"

    try:
        progress(0.1, desc="🔄 Обработка файла...")

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        if audio_upload:
            new_path = os.path.join("uploads", f"uploaded_{timestamp}{os.path.splitext(audio_path)[1]}")
        else:
            new_path = os.path.join("recordings", f"recorded_{timestamp}.wav")

        shutil.copy(audio_path, new_path)

        progress(0.5, desc="🔍 Анализ аудио...")
        transcript, phrases, summary = await transcribe_audio(new_path)

        progress(1.0, desc="✅ Анализ завершен!")
        return transcript, phrases, summary

    except Exception as e:
        return f"❌ Ошибка: {str(e)}", f"❌ Ошибка: {str(e)}", f"❌ Ошибка: {str(e)}"


custom_css = """
:root {
    --input-bg: #f8f9fa;
    --input-border: #e0e0e0;
}
.dark {
    --input-bg: #2b2b2b;
    --input-border: #444;
}

#audio-analyzer-app {
    max-width: 900px !important;
}
.audio-input-container {
    background: var(--input-bg) !important;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid var(--input-border) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.output-container {
    height: 70vh;
    min-height: 400px;
}
.tab-content {
    height: calc(100% - 40px) !important;
}
.textbox {
    height: 100% !important;
    font-family: monospace !important;
}
.transcript-textbox {
    height: 100% !important;
    white-space: pre-wrap;
}
.analyze-btn {
    background: #2196F3 !important;
    color: white !important;
}
.progress-bar {
    margin-top: 10px;
}
.tab-button {
    padding: 8px 15px !important;
}
.file-upload {
    border: 2px dashed #aaa !important;
    padding: 20px !important;
    border-radius: 10px !important;
}
.file-upload:hover {
    border-color: #2196F3 !important;
}
.audio-player {
    width: 100% !important;
    margin-top: 10px;
}
"""

with gr.Blocks(title="Аудио Анализатор", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""# 🎤 Универсальный Аудио Анализатор
    <p style="color: #666;">Запишите с микрофона или загрузите аудиофайл для полного анализа</p>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="audio-input-container"):
            with gr.Tabs():
                with gr.Tab("🎙️ Запись", id="record_tab"):
                    audio_input = gr.Audio(
                        label="Запись с микрофона",
                        type="filepath",
                        sources=["microphone"],
                        elem_classes="audio-player"
                    )

                with gr.Tab("📁 Загрузка", id="upload_tab"):
                    file_input = gr.Audio(
                        label="Загрузите аудиофайл",
                        type="filepath",
                        sources=["upload"],
                        elem_classes=["audio-player", "file-upload"]
                    )

            with gr.Row():
                clear_btn = gr.Button("Очистить", variant="secondary")
                analyze_btn = gr.Button("Анализировать", variant="primary", elem_classes="analyze-btn")

            progress_bar = gr.HTML(visible=False)

        with gr.Column(scale=2, elem_classes="output-container"):
            with gr.Tabs():
                with gr.Tab("📜 Полный текст", elem_classes="tab-button"):
                    transcript_output = gr.Textbox(
                        label="Транскрипция аудио",
                        lines=18,
                        interactive=True,
                        elem_classes="transcript-textbox"
                    )
                with gr.Tab("🔍 Ключевые фразы", elem_classes="tab-button"):
                    phrases_output = gr.Textbox(
                        label="Выделенные фразы",
                        lines=18,
                        interactive=False,
                        elem_classes="textbox"
                    )
                with gr.Tab("📝 Краткое содержание", elem_classes="tab-button"):
                    summary_output = gr.Textbox(
                        label="Суммаризированный текст",
                        lines=18,
                        interactive=False,
                        elem_classes="textbox"
                    )


    def show_progress():
        return gr.HTML("""
        <div class="progress-bar">
            <progress value="0" max="1" style="width: 100%"></progress>
            <div id="progress-text">Подготовка...</div>
        </div>
        """, visible=True)


    def get_active_audio(audio_record, audio_upload):
        return audio_upload if audio_upload else audio_record

    analyze_btn.click(
        show_progress,
        outputs=progress_bar
    ).then(
        handle_audio,
        inputs=[audio_input, file_input],  # Теперь передаем оба источника как отдельные входы
        outputs=[transcript_output, phrases_output, summary_output],
    ).then(
        lambda: gr.HTML(visible=False),
        outputs=progress_bar
    )

    clear_btn.click(
        lambda: [None, None, "", "", "", gr.HTML(visible=False)],
        outputs=[audio_input, file_input, transcript_output, phrases_output, summary_output, progress_bar]
    )

if __name__ == "__main__":
    try:
        demo.queue(concurrency_count=4).launch(server_port=7860)
    except TypeError:
        demo.queue().launch(server_port=7860)