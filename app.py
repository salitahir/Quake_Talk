import streamlit as st
import os
import pandas as pd
import polars as pl
import tiktoken
import re

from quake_talk.search import semantic_search
from quake_talk.gpt    import ask, summarize_with_gpt4o
from quake_talk.preprocessing.clean_text import extract_sentences, extract_keywords

# ── Page config: wide mode & favicon ─────────────────────────────────
st.set_page_config(
    page_title="🌏 Quake Talk 4.0",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data & models ───────────────────────────────────────────────
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

df      = load_data()
encoder = get_token_encoder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# ── Header: title and byline ────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])
with col_left:
    st.title("🌏 Quake Talk 4.0")

st.divider()

with col_right:
    st.markdown(
        """
        <div style="text-align:right; font-size:0.8em;">
          Developed & Deployed by
          <a href="https://www.linkedin.com/in/salitahir/" target="_blank">
            Syed Ali Tahir
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Full-width right-aligned “Overview” link (small caption) ──────────
st.markdown(
    """
    <div style="text-align:right; font-size:0.8em; margin-bottom:0.5em;">
      <a
        href="https://www.canva.com/design/DAGlyTsyFjo/MHYjTbNINhCSEOmxVzpXAA/view?utm_content=DAGlyTsyFjo&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hd2aea126e2"
        target="_blank"
      >
        Overview and Business Application
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Full-width project description ────────────────────────────────────
st.markdown(
    """
    <div style="text-align:justify; line-height:1.5em;">
      An interactive GPT-powered tool to improve analytical frameworks within
      disaster contexts and beyond. Quake Talk uses both keyword and semantic
      filtering to mine relevant context from user-generated social media data
      relating to the 2023 Turkey–Syria earthquake through a dynamic and
      interactive prototype application.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Helpers ────────────────────────────────────────────────────────────
def truncate_by_tokens(text: str, max_tokens: int, enc) -> str:
    toks = enc.encode(text)
    return enc.decode(toks[-max_tokens:]) if len(toks) > max_tokens else text

def keyword_filter(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return df
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b"
    return df[df["content_clean"].str.contains(pattern, case=False, regex=True)]

# ── Sidebar controls ─────────────────────────────────────────────────
st.sidebar.header("Filters & Options")
min_d = df["date"].dt.date.min()
max_d = df["date"].dt.date.max()
raw_dates = st.sidebar.date_input(
    "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if isinstance(raw_dates, (tuple, list)) and len(raw_dates) == 2:
    start_date, end_date = raw_dates
else:
    start_date = end_date = raw_dates

max_sents = st.sidebar.slider("Max sentences", 50, 1000, 300, 50)
filter_method    = st.sidebar.selectbox("Filtering method", ["Semantic", "Keyword"])
show_context     = st.sidebar.checkbox("Show context", value=False)
generate_summary = st.sidebar.checkbox("Generate summary", value=False)
temperature      = st.sidebar.slider("GPT Temperature", 0.0, 1.0, 0.4, 0.01)
max_context_tokens = st.sidebar.slider(
    "Max context tokens",
    min_value=100,
    max_value=int(os.getenv("MAX_CONTEXT_TOKENS", 8000)),
    value=int(os.getenv("MAX_CONTEXT_TOKENS", 8000)),
    step=100,
)

# ── Main UI ───────────────────────────────────────────────────────────
question = st.text_input("Ask a question about the 2023 Turkey–Syria Earthquake Tweets:")
run = st.button("🔍 Ask GPT-4o")

if not run:
    st.info("Enter a question above and click “🔍 Ask GPT-4o” to begin.")
else:
    # 1) date filter
    mask   = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    subset = df.loc[mask]

    if subset.empty:
        st.warning("No tweets in that date range.")
    else:
        # 2) content filter
        if filter_method == "Semantic":
            candidate_k = max_sents * 2
            idxs, dists = semantic_search(question, top_k=candidate_k)
            valid = [i for i in idxs if mask.iat[i]]
            if not valid:
                st.warning("No semantically-relevant tweets in that date range.")
                subset = subset.iloc[0:0]
            else:
                subset = df.iloc[valid[:max_sents]]
        else:
            kw     = extract_keywords(question)
            subset = keyword_filter(subset, kw)

        if subset.empty:
            st.warning("No tweets matched your filter.")
        else:
            # 3) show filtered tweets
            with st.expander("📋 Show filtered tweets", expanded=False):
                st.dataframe(subset, use_container_width=True)

            # 4) build & show context
            texts   = subset["content_clean"].tolist()
            raw_ctx = " ".join(extract_sentences(texts, max_sents))
            ctx     = truncate_by_tokens(raw_ctx, max_context_tokens, encoder)

            if show_context:
                with st.expander("🔍 Context", expanded=False):
                    st.write(raw_ctx)

            # 5) summary
            if generate_summary:
                with st.expander("📝 Summary", expanded=False):
                    with st.spinner("Summarizing…"):
                        summ = summarize_with_gpt4o(ctx)
                    st.write(summ)

            # 6) final answer
            with st.spinner("Querying GPT-4o…"):
                answer = ask(question, ctx, temperature=temperature)
            with st.expander("💬 Answer", expanded=True):
                st.write(answer)