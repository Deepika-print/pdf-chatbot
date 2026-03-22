import gradio as gr
import os
from ingest import ingest_pdf
from query import load_vector_store, answer, GROQ_MODEL

_index = None
_chunks = None

def handle_upload(pdf_file):
    global _index, _chunks
    if pdf_file is None:
        return "No file uploaded."
    try:
        n = ingest_pdf(pdf_file)
        _index, _chunks = load_vector_store()
        return f"Ready! Indexed {n} chunks from '{os.path.basename(pdf_file)}'. Ask me anything."
    except Exception as e:
        return f"Error: {e}"

def respond(message, history):
    if not message.strip():
        return history, ""
    if _index is None:
        return history + [[message, "Please upload a PDF first."]], ""
    ans, sources = answer(message, _index, _chunks)
    snippets = "\n".join(f"> {s['text'][:180].replace(chr(10),' ')}..." for s in sources)
    full_reply = f"{ans}\n\n**Sources:**\n{snippets}"
    return history + [[message, full_reply]], ""

with gr.Blocks(title="PDF Chatbot") as demo:
    gr.Markdown("# PDF Chatbot\nUpload a PDF and ask questions about it.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1 — Upload PDF")
            pdf_input = gr.File(label="Choose a PDF", file_types=[".pdf"], type="filepath")
            status_box = gr.Textbox(label="Status", value="No PDF loaded yet.", interactive=False, lines=3)
            gr.Markdown(f"**LLM:** {GROQ_MODEL} via Ollama")
        with gr.Column(scale=2):
            gr.Markdown("### Step 2 — Ask questions")
            chatbot = gr.Chatbot(label="Chat", height=460, type="messages")
            msg_box = gr.Textbox(label="Your question", placeholder="e.g. What is normalization?", lines=2)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", scale=1)
    pdf_input.change(fn=handle_upload, inputs=[pdf_input], outputs=[status_box])
    send_btn.click(fn=respond, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    msg_box.submit(fn=respond, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_box])

demo.launch(server_port=7860, inbrowser=True)