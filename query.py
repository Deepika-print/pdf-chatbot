import faiss
import pickle
import numpy as np
import os
from groq import Groq
from ingest import get_model, VECTOR_STORE_DIR

GROQ_MODEL = "llama3-8b-8192"

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it as a Space Secret on Hugging Face.")
    return Groq(api_key=api_key)

def load_vector_store():
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError("No vector store found. Upload a PDF first.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"[Query] Loaded {len(chunks)} chunks.")
    return index, chunks

def retrieve(question, index, chunks, top_k=3):
    m = get_model()
    q_vec = m.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, top_k)
    return [{"rank": i+1, "score": float(s), "text": chunks[idx]}
            for i, (s, idx) in enumerate(zip(scores[0], indices[0]))]

def build_prompt(question, context_chunks):
    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    return f"""You are a helpful assistant. Answer using ONLY the document context below.
If the answer is not in the context, say "I couldn't find that in the document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

def answer(question, index, chunks, top_k=3):
    context_chunks = retrieve(question, index, chunks, top_k)
    prompt = build_prompt(question, context_chunks)
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        ans = response.choices[0].message.content
    except Exception as e:
        ans = f"Error: {e}"
    return ans, context_chunks