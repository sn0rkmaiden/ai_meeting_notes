import datetime
import shutil
import gradio as gr
import diarization
import summarize
from typing import Optional


def handle_audio(audio_record: Optional[str], audio_upload: Optional[str], progress=gr.Progress()) -> tuple[str, str, str]:
    audio_path = audio_upload if audio_upload else audio_record

    if audio_path is None:
        return "Аудио не загружено", "Аудио не загружено", "Аудио не загружено"

    timestamp = datetime.datetime.now().isoformat()
    new_path = f"recorded_{timestamp}"
    if "." in audio_path:
        new_path += audio_path[audio_path.rfind("."):]
    shutil.move(audio_path, new_path)

    progress(0, "🔍 Анализ аудио...")

    phrases = diarizer.diarize(new_path)

    md = diarization.phrases_to_markdown(phrases)

    progress(0.33, "🔍 Выделение ключевых слов...")

    keywords = (
        "1. Невыдуманные истории\n"
        "2. О которых невозможно\n"
        "3. Не рассказать"
    )

    progress(0.67, "🔍 Составление краткого содержания...")

    summary = llm.summarize(md)

    progress(1.0, "✅ Анализ завершен!")

    return md, keywords, summary


diarizer = diarization.Diarizer()

llm = summarize.Summarizer()


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
