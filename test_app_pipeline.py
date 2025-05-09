# test_app_pipeline.py

from app import load_data, truncate_by_tokens, get_token_encoder
from quake_talk.search import semantic_search
from quake_talk.gpt    import ask
import pandas as pd


def main():
    # 1) Load & inspect data
    df = load_data()
    print(f"✅ Loaded {len(df)} rows")
    print("Columns:", df.columns.tolist())

    # 2) Date‐range check
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    print(f"Date range in data: {min_date} → {max_date}")

    # 3) Full‐range filter sanity
    subset = df[(df["date"].dt.date >= min_date) & (df["date"].dt.date <= max_date)]
    print(f"Rows after full‐range filter: {len(subset)}")

    # 4) Semantic search test
    question = "What were the main concerns expressed?"
    idxs, dists = semantic_search(question, top_k=3)
    print("Top3 indices:", idxs)
    print("Distances:", dists)

    # 5) Build a tiny context & truncate it
    sample_texts = subset["content_clean"].tolist()[:5]
    raw_ctx = " ".join(sample_texts)
    encoder = get_token_encoder("all-MiniLM-L6-v2")
    truncated = truncate_by_tokens(raw_ctx, max_tokens=20, enc=encoder)
    print("Truncated context (20 tokens):", truncated)

    # 6) GPT call (requires OPENAI_API_KEY in .env or env)
    print("Asking GPT…")
    answer = ask(question, truncated, temperature=0.5, max_tokens=200)
    print("GPT answer:", answer)


if __name__ == "__main__":
    main()
