# app.py
import streamlit as st
import os
import pandas as pd
import polars as pl
import tiktoken

from quake_talk.search import semantic_search
from quake_talk.gpt    import ask, summarize_with_gpt4o
from quake_talk.preprocessing.clean_text import extract_sentences, extract_keywords

# ── Config & Constants ─────────────────────────────────────────────
# Env‐driven default limit (from .env / .env.example)
MAX_CONTEXT_TOKENS_DEFAULT = int(os.getenv("MAX_CONTEXT_TOKENS", 8000))
EMODEL                    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Caching ────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pl.read_parquet("artifacts/tweets_cleaned.parquet").to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_resource
def get_token_encoder(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

# ── App start ──────────────────────────────────────────────────
df      = load_data()
encoder = get_token_encoder(EMODEL)

# Title + byline
st.title("📰 Quake Talk 4.0")
st.markdown(
    "<small>Developed & Deployed by "
    "<a href='https://www.linkedin.com/in/salitahir/' target='_blank'>Syed Ali Tahir</a></small>",
    unsafe_allow_html=True,
)

st.write("---")

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

# 1) Compute the absolute bounds of data
min_d = df["date"].dt.date.min()
max_d = df["date"].dt.date.max()

# 2) Let the user pick a single date or a range
raw_dates = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)

# 3) Normalize to two dates
if isinstance(raw_dates, (tuple, list)) and len(raw_dates) == 2:
    start_date, end_date = raw_dates
else:
    # If they’ve only clicked one day so far, use it for both ends
    start_date = end_date = raw_dates

# 4) Now it’s always safe to build your mask
date_mask = (
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
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
# Page title

question = st.text_input("Ask a question about the 2023 Turkey–Syria Earthquake Tweets:")
run      = st.button("Ask GPT-4o")

if not run:
    st.info("Enter a question above and click “Ask GPT-4o” to begin.")
else:
    # 1) filter by date
    mask   = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    subset = df.loc[mask]

    if subset.empty:
        st.warning("No tweets in that date range.")
    else:
        if filter_method == "Semantic":
            # 2a) Content filter
            candidate_k = max_sents * 2
            idxs, dists = semantic_search(question, top_k=candidate_k)

            # 2b) keep only those hits whose original tweet is in the date window
            valid = [i for i in idxs if mask.iat[i]]

            # 2c) if nothing left, warn and skip further processing
            if not valid:
                st.warning("No semantically-relevant tweets in that date range.")
                subset = subset.iloc[0:0]  # make it empty
            else:
                # 2d) trim to the real max_sents
                valid = valid[:max_sents]
                subset = df.iloc[valid]
        else:
            # your keyword branch unchanged
            kw     = extract_keywords(question)
            subset = keyword_filter(subset, kw)

        if subset.empty:
            st.warning("No tweets matched your filter.")
        else:
            # 3) DataFrame view
            with st.expander("📋 Show filtered tweets", expanded=False):
                st.dataframe(subset, use_container_width=True)

            # 4) build context
            texts   = subset["content_clean"].tolist()
            raw_ctx = " ".join(extract_sentences(texts, max_sents))
            ctx     = truncate_by_tokens(raw_ctx, max_context_tokens, encoder)

            # 5) Context expander
            if show_context:
                with st.expander("🔍 Show Context", expanded=False):
                    st.write(raw_ctx)

            # 6) Summary expander
            if generate_summary:
                with st.expander("📝 Generate Summary", expanded=False):
                    with st.spinner("Summarizing…"):
                        summ = summarize_with_gpt4o(ctx)
                    st.markdown("**Summary:**")
                    st.write(summ)

            # 7) Final answer
            with st.spinner("Querying GPT-4o…"):
                answer = ask(question, ctx, temperature=temperature)
            st.markdown("**Answer:**")
            st.write(answer)