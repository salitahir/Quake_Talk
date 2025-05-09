# scripts/build_index.py
import os
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()
DIM       = int(os.getenv("EMBED_DIM", 384))
INDEX_OUT = os.getenv("INDEX_FILE", "artifacts/tweets.index")
PQ_BYTES  = int(os.getenv("PQ_BYTES", 96))        # 96 → PQ96
N_LIST    = int(os.getenv("IVF_NLIST", 1024))     # number of Voronoi cells
EMB_FILE  = os.getenv("EMBEDDING_FILE", "artifacts/tweet_embeddings.npy")

def main():
    # 1. Load embeddings
    vecs = np.load(EMB_FILE).astype("float32")

    # 2. Build IVF + PQ index
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFPQ(quantizer, DIM, N_LIST, PQ_BYTES, 8)
    index.train(vecs)
    index.add(vecs)
    
    # 3. Serialize
    faiss.write_index(index, INDEX_OUT)
    print(f"✅ Built FAISS index at {INDEX_OUT}")

if __name__ == "__main__":
    main()