import fitz
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = "vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        print("  Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        print("  Embedding model ready.")
    return _model


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = " ".join(page.get_text("text").split())
        if text.strip():
            pages.append(f"[Page {i+1}] {text}")
    doc.close()
    if not pages:
        raise ValueError("No text found. PDF may be scanned/image-based.")
    print(f"  Extracted {len(''.join(pages)):,} characters from {len(pages)} pages.")
    return "\n\n".join(pages)


def chunk_text(text, chunk_size=400, overlap=60):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    print(f"  Created {len(chunks)} chunks.")
    return chunks


def embed_chunks(chunks):
    m = get_model()
    print(f"  Embedding {len(chunks)} chunks...")
    vecs = m.encode(chunks, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    return vecs.astype("float32")


def save_vector_store(chunks, embeddings):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # Always delete old files first — prevents WinError 183 on Windows
    for fname in ("index.faiss", "chunks.pkl"):
        fpath = os.path.join(VECTOR_STORE_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_STORE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved {len(chunks)} chunks to {VECTOR_STORE_DIR}/")


def ingest_pdf(pdf_path):
    print(f"\n[Ingest] {os.path.basename(pdf_path)}")
    print("Step 1/4  Extracting text...")
    text = extract_text(pdf_path)
    print("Step 2/4  Chunking...")
    chunks = chunk_text(text)
    print("Step 3/4  Embedding...")
    embeddings = embed_chunks(chunks)
    print("Step 4/4  Saving vector store...")
    save_vector_store(chunks, embeddings)
    print(f"[Ingest] Done — {len(chunks)} chunks ready.\n")
    return len(chunks)


if __name__ == "__main__":
    import sys
    ingest_pdf(sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf")