# app.py
import os
import streamlit as st
import pandas as pd
import polars as pl
import tiktoken

from quake_talk.search import semantic_search
from quake_talk.gpt    import ask
from quake_talk.preprocessing.clean_text import (
    extract_sentences,
    summarize_with_gpt4o,
    extract_keywords,
)

# ── Config & Constants ─────────────────────────────────────────────
# Env‐driven default limit (from .env / .env.example)
MAX_CONTEXT_TOKENS_DEFAULT = int(os.getenv("MAX_CONTEXT_TOKENS", 8000))
EMODEL                    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Caching ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # Read once; convert to pandas for easy masking
    df = pl.read_parquet("artifacts/tweets_cleaned.parquet").to_pandas()

    # Convert your existing 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_resource(show_spinner=False)
def get_token_encoder(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

# ── Token‐based truncation ─────────────────────────────────────────
def truncate_by_tokens(text: str, max_tokens: int, enc) -> str:
    toks = enc.encode(text)
    if len(toks) > max_tokens:
        toks = toks[-max_tokens:]
    return enc.decode(toks)

# ── Keyword‐filter helper ─────────────────────────────────────────
import re

def keyword_filter(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """
    Return only the rows where `content_clean` contains
    any of the given keywords as whole words (case‐insensitive).
    """
    if not keywords:
        return df
    # build a pattern like r'\b(?:fire|water|help)\b'
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b"
    return df[df["content_clean"].str.contains(pattern, case=False, regex=True)]

# ── Load once ──────────────────────────────────────────────────────
df      = load_data()
encoder = get_token_encoder(EMODEL)

# ── Sidebar Widgets ───────────────────────────────────────────────
st.sidebar.header("Filters & Options")

# 1. Date range (now using 'date', not 'tweet_date')
min_d = df["date"].dt.date.min()
max_d = df["date"].dt.date.max()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)

# 2. Max sentences
max_sents = st.sidebar.slider("Max sentences", 50, 1000, 300, 50)

# 3. Semantic vs Keyword
filter_method = st.sidebar.selectbox("Filtering method", ["Semantic", "Keyword"])

# 4. Show raw context?
show_context = st.sidebar.checkbox("Show context", value=False)

# 5. Generate summary?
generate_summary = st.sidebar.checkbox("Generate summary", value=False)

# 6. GPT Temperature
temperature = st.sidebar.slider("GPT Temperature", 0.0, 1.0, 0.4, 0.01)

# 7. Max context tokens
max_context_tokens = st.sidebar.number_input(
    "Max context tokens",
    min_value=100,
    max_value=MAX_CONTEXT_TOKENS_DEFAULT,
    value=MAX_CONTEXT_TOKENS_DEFAULT,
    help="How many tokens of context to send to GPT",
)

# ── Main UI ────────────────────────────────────────────────────────
st.title("📰 Quake Talk Streamlit Demo")

question = st.text_input("Ask a question about the Turkey–Syria tweets:")

if st.button("Ask GPT-4o"):
    # 1) Date filter on your 'date' column
    mask = (
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    )
    subset = df.loc[mask]

    if subset.empty:
        st.warning("No tweets in that date range.")
    else:
        # 2) Content filter
        if filter_method == "Semantic":
            idxs, dists = semantic_search(question, top_k=max_sents)
            subset = subset.reset_index(drop=True).iloc[idxs]
        else:
            kw = extract_keywords(question)
            subset = keyword_filter(subset, kw)

        if subset.empty:
            st.warning("No tweets matched your filter.")
        else:
            texts = subset["content_clean"].tolist()

            # 3) Extract & truncate context
            raw_ctx = " ".join(extract_sentences(texts, max_sents))
            ctx     = truncate_by_tokens(raw_ctx, max_context_tokens, encoder)

            # 4) Optionally show raw context
            if show_context:
                st.markdown("**Context:**")
                st.write(raw_ctx)

            # 5) Optionally generate summary
            if generate_summary:
                with st.spinner("Generating summary…"):
                    summ = summarize_with_gpt4o(ctx)
                st.markdown("**Summary:**")
                st.write(summ)

            # 6) Ask GPT
            with st.spinner("Querying GPT-4o…"):
                answer = ask(question, ctx, temperature=temperature)
            st.markdown("**Answer:**")
            st.write(answer)