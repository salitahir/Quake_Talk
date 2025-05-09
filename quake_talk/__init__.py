# quake_talk/__init__.py

"""
Quake_Talk package
===============

Provides:
  - clean_tweet       (text cleaning)
  - semantic_search   (FAISS-powered similarity search)
  - ask               (GPT query wrapper)
"""

__version__ = "0.1.0"

# Core cleaning utility
from .preprocessing.clean_text import clean_tweet

# Phase 2 modules
from .search import semantic_search
from .gpt    import ask

__all__ = [
    "clean_tweet",
    "semantic_search",
    "ask",
]
