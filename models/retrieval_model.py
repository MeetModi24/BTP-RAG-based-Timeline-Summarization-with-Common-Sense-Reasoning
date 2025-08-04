# models/retrieval_model.py

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from config import RETRIEVER_MODEL_NAME, TOP_K_RETRIEVAL

def load_retriever(model_name=RETRIEVER_MODEL_NAME):
    retriever = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return retriever.to(device)

def build_faiss_index(articles, retriever):
    article_texts = [a["text"] for a in articles]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8 if device == "cpu" else 32

    embeddings = retriever.encode(
        article_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings

def retrieve_documents(query, retriever, index, articles, top_k=TOP_K_RETRIEVAL):
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    query_embedding = retriever.encode([query]).astype(np.float32)
    top_k = min(top_k, index.ntotal)

    distances, indices = index.search(query_embedding, top_k)
    retrieved = [articles[i] for i in indices[0]]
    return retrieved
