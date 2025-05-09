# scripts/clean_dataset.py
"""
Batch process raw CSV into cleaned Parquet for Phase 1.
Loads environment variables, applies text cleaning (with optional lemmatization),
and writes compressed Parquet output for downstream use.
"""

print("🔄 Starting cleaner…")

# Section 1: Load environment and secrets
from dotenv import load_dotenv
import os

load_dotenv()  # populate os.environ from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # not strictly needed in Phase 1
HF_TOKEN       = os.getenv("HF_TOKEN")

# Section 2: Core script imports and configuration
import polars as pl
from quake_talk.preprocessing.clean_text import clean_tweet

# Determine input and output paths (override via env vars if set)
RAW_CSV         = os.getenv("RAW_CSV", "data/tweets_english.csv")
CLEANED_PARQUET = os.getenv("CLEANED_PARQUET", "artifacts/tweets_cleaned.parquet")
CLEANED_CSV     = os.getenv("CLEANED_CSV", None)  # e.g., "artifacts/tweets_cleaned.csv"

# Cleaning options (can also be set via env vars)
REMOVE_STOPWORDS = os.getenv("REMOVE_STOPWORDS", "true").lower() == "true"
LEMMATIZE        = os.getenv("LEMMATIZE",      "false").lower() == "true"


def main() -> None:
    """
    Main entry point: reads raw tweets, cleans text, and writes output.
    """
    # 1. Read raw CSV into a Polars DataFrame
    df = pl.read_csv(RAW_CSV)

    # 2. Apply cleaning to the tweet content column
    df = df.with_columns(
        pl.col("content")
          .map_elements(
              lambda txt: clean_tweet(
                  txt,
                  remove_stopwords=REMOVE_STOPWORDS,
                  lemmatize=LEMMATIZE
              ),
              return_dtype=pl.Utf8
          )
          .alias("content_clean")
    )

    # 3. Write cleaned data to Parquet
    df.write_parquet(CLEANED_PARQUET, compression="zstd")
    print(f"✅ Cleaned data written to {CLEANED_PARQUET}")

    # 4. Optionally write a CSV copy
    if CLEANED_CSV:
        df.write_csv(CLEANED_CSV)
        print(f"✅ Cleaned data CSV written to {CLEANED_CSV}")


if __name__ == "__main__":
    main()