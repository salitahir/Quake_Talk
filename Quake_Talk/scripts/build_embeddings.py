# scripts/build_embeddings.py
import os
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
EMODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PARQUET_IN = os.getenv("CLEANED_PARQUET", "artifacts/tweets_cleaned.parquet")
EMB_OUT    = os.getenv("EMBEDDING_FILE", "artifacts/tweet_embeddings.npy")

def main():
    # 1. Load cleaned tweets
    df = pl.read_parquet(PARQUET_IN, columns=["content_clean"])
    texts = df["content_clean"].to_list()

    # 2. Encode
    model = SentenceTransformer(EMODEL)
    embeddings = model.encode(texts, show_progress_bar=True)

    # 3. Save
    np.save(EMB_OUT, embeddings)
    print(f"✅ Saved {embeddings.shape[0]} embeddings to {EMB_OUT}")

if __name__ == "__main__":
    main()