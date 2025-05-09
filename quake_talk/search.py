# quake_talk/search.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load your FAISS index once
IDX = faiss.read_index(
    os.getenv("INDEX_FILE", "artifacts/tweets.index"),
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
)

# 2. Load the same embedding model
EMODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL  = SentenceTransformer(EMODEL)

def semantic_search(question: str, top_k: int = 10):
    """
    Returns the indices and distances of the top_k most similar tweets.
    """
    # 1. Encode and normalize
    q_emb = MODEL.encode(question, normalize_embeddings=True).astype("float32")
    # 2. Search FAISS
    distances, indices = IDX.search(q_emb[None, :], top_k)
    return indices[0], distances[0]
