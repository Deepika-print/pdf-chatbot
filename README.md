# PDF Chatbot — RAG + Groq + FAISS

Ask questions about any PDF. Answers come only from your document — no hallucination.

## Live Demo
🔗 [huggingface.co/spaces/deepika-singh/pdf-chatbot](https://huggingface.co/spaces/deepika-singh/pdf-chatbot)

## How it works

1. Upload any PDF
2. Text is extracted page by page using PyMuPDF
3. Text is split into overlapping chunks of 400 words
4. Each chunk is converted to a 384-dim vector using sentence-transformers
5. Vectors are stored in a FAISS index for fast similarity search
6. When you ask a question, it is embedded and matched against the index
7. Top 3 matching chunks are sent to Groq LLaMA as context
8. LLM answers based only on your document — no hallucination

## Stack

| Component | Library |
|---|---|
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| LLM | Groq API (llama-3.1-8b-instant) |
| Web UI | Gradio |

## Run locally

```bash

Open http://127.0.0.1:7860 in your browser.

## Project structure

```
pdf-chatbot/
├── app.py           — Gradio web UI
├── ingest.py        — PDF parsing, chunking, embedding, FAISS indexing
├── query.py         — retrieval, prompt building, Groq LLM call
├── requirements.txt — dependencies
└── vector_store/    — auto-created after first PDF upload
```

## What I learned building this

- How RAG (Retrieval-Augmented Generation) works end-to-end
- Vector embeddings and cosine similarity search with FAISS
- Chunking strategies and overlap for better retrieval quality
- Prompt engineering to prevent LLM hallucination
- Deploying ML apps on Hugging Face Spaces

## License

MIT — see [LICENSE](LICENSE)
