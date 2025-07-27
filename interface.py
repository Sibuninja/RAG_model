import gradio as gr
from rag_engine import load_and_split, build_vectorstore, create_qa_chain

vectordb = None
qa_chain = None

# Upload and index the file
def upload_file(file):
    global vectordb, qa_chain
    chunks = load_and_split(file.name)
    vectordb = build_vectorstore(chunks)
    qa_chain = create_qa_chain(vectordb)
    return "✅ Document indexed successfully!"

# Handle user question
def ask_question(question):
    if qa_chain is None:
        return "⚠️ Please upload a document first."
    return qa_chain.run(question)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔍📄 Chat with Your PDF (RAG-powered)")

    with gr.Row():
        file_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Index File")
    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        ask_btn = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer", interactive=False)

    upload_btn.click(upload_file, inputs=file_input, outputs=status_output)
    ask_btn.click(ask_question, inputs=question_input, outputs=answer_output)

# Entry point for app.py
demo.launch()
