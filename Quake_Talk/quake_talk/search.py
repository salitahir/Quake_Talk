# quake_talk/search.py

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_faiss_index(index_path: str = os.getenv("INDEX_FILE", "artifacts/tweets.index")) -> faiss.Index:
    """Read & cache the FAISS index."""
    return faiss.read_index(
        index_path,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
    )

@lru_cache(maxsize=1)
def _load_embedding_model(name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")) -> SentenceTransformer:
    """Instantiate & cache the SentenceTransformer."""
    return SentenceTransformer(name)

def semantic_search(question: str, top_k: int = 10):
        # 1) load (cached) model & index
    model = _load_embedding_model()
    idx   = _load_faiss_index()

    # 2) encode & search
    query_emb = model.encode(question)
    distances, indices = idx.search(np.array([query_emb]), top_k)
    return indices.flatten().tolist(), distances.flatten().tolist()