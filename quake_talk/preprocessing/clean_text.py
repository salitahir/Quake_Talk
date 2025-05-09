# quake_talk/preprocessing/clean_text.py

"""
Provides text-cleaning utilities for social media data,
including HTML unescaping, punctuation control, and optional
context-aware lemmatization via spaCy.
"""

import re
import html
from typing import Optional

# 1) spaCy for context-aware tokenization & lemmatization
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except (ImportError, OSError):
    nlp = None

# 2) NLTK for fallback lemmatizer & stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# Download only the NLTK data we need
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  

STOPWORDS = set(stopwords.words("english"))
NLTK_LEMMATIZER = WordNetLemmatizer()

def clean_tweet(
    text: Optional[str],
    remove_stopwords: bool = True,
    lemmatize: bool = False
) -> str:
    """
    1. Handle null/empty.
    2. HTML-unescape entities.
    3. Remove hashtags (#tag), mentions (@user), URLs, and emails.
    4. Keep only letters, numbers, basic punctuation (.,!?'"”).
    5. Normalize whitespace.
    6. Lowercase.
    7. Tokenize (spaCy if available; regex fallback).
    8. Optionally drop stopwords.
    9. Optionally lemmatize (spaCy if available; NLTK fallback).
    """
    if not text:
        return ""

    # 2. HTML unescape
    text = html.unescape(text)

    # 3. Remove noise
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S*@\S*", "", text)

    # 4. Remove all punctuation (leave only letters, numbers, whitespace)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Lowercase
    text = text.lower()

    # 7. Tokenize
    if nlp is not None:
        doc = nlp(text)
        tokens = [tok.text for tok in doc if not tok.is_space]
    else:
        # simple regex fallback
        tokens = re.findall(r"\w+", text)

    # 8. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    # 9. Lemmatize if requested
    if lemmatize:
        if nlp is not None:
            doc = nlp(" ".join(tokens))
            tokens = [tok.lemma_ for tok in doc]
        else:
            tokens = [NLTK_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def extract_sentences(texts: list[str], max_sents: int = 300) -> list[str]:
    combined = " ".join(t.strip() for t in texts)
    sents = sent_tokenize(combined)
    return sents[:max_sents]

def extract_keywords(question: str) -> list[str]:
    words = re.findall(r"\w+", question.lower())
    lemm = [NLTK_LEMMATIZER.lemmatize(w) for w in words]
    return [w for w in lemm if w not in STOPWORDS]

from quake_talk.gpt import ask  # ensure this import comes *after* gpt.py exists

def summarize_with_gpt4o(context: str) -> str:
    prompt = "Please summarize the key concerns, themes, and sentiments expressed in these tweets."
    return ask(prompt, context)