"""Text preprocessing shared by the TF-IDF index (documents) and its query side.

Pipeline: cleaning (done once for the whole corpus in data.py, Arrow regex kernels)
          -> lower-casing -> tokenisation -> stop-word removal -> Porter stemming.

BM25 does NOT use this module: Lucene's analyzer does the same four steps inside Java at both index and
query time (StandardTokenizer, lower-case, English stop-words, Porter stemmer). Dense models must see raw
text (their WordPiece tokenisers expect natural sentences), so they are not stemmed either.
"""
from __future__ import annotations

from typing import Callable

import Stemmer  # PyStemmer: C implementation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# Alphanumeric tokens of length >= 2 (unicode aware, underscores excluded). Punctuation disappears here.
TOKEN_PATTERN = r"(?u)\b[^\W_]{2,}\b"
STOP_WORDS = sorted(ENGLISH_STOP_WORDS)
STEMMER_ALGORITHM = "porter"   # same algorithm as Lucene's PorterStemFilter used by the BM25 index


def new_stemmer() -> "Stemmer.Stemmer":
    return Stemmer.Stemmer(STEMMER_ALGORITHM)


def count_vectorizer(**kwargs) -> CountVectorizer:
    """CountVectorizer with the project-wide lower-casing / tokenisation / stop-word settings."""
    return CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, stop_words=STOP_WORDS, **kwargs)


def query_analyzer() -> Callable[[str], list[str]]:
    """text -> stemmed tokens. Identical to the document side, where stemming is applied per vocabulary term
    instead of per token (stemming is deterministic per token, so the two are equivalent)."""
    base = count_vectorizer().build_analyzer()
    stemmer = new_stemmer()
    return lambda text: stemmer.stemWords(base(text))
