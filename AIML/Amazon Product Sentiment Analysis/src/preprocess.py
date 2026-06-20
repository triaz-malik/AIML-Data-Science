"""Text cleaning for the classical (TF-IDF) pipeline.

Steps: lowercase -> strip URLs/HTML -> strip punctuation/digits ->
remove stopwords -> lemmatize.

NOTE: transformer models (DistilBERT/BERT) use their OWN tokenizer and want
the *raw* text — do NOT feed them this cleaned output. This cleaning is for
the TF-IDF + Logistic Regression baseline and for word-cloud EDA.
"""
from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def ensure_nltk() -> None:
    """Download the NLTK resources we depend on (idempotent, quiet)."""
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_lemmatizer: WordNetLemmatizer | None = None
_stopwords: set[str] | None = None


def _lazy_init():
    global _lemmatizer, _stopwords
    if _lemmatizer is None:
        ensure_nltk()
        _lemmatizer = WordNetLemmatizer()
        # Keep negations — they carry sentiment ("not good" != "good").
        keep = {"not", "no", "nor", "never", "none", "cannot"}
        _stopwords = set(stopwords.words("english")) - keep


def clean_text(text: str) -> str:
    """Return a cleaned, lemmatized, stopword-free version of `text`."""
    _lazy_init()
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = _MULTISPACE_RE.sub(" ", text).strip().split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stopwords and len(t) > 2
    ]
    return " ".join(tokens)


def clean_series(texts) -> list[str]:
    """Vectorized-ish helper: clean an iterable/Series of strings."""
    _lazy_init()
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    demo = "This product is AMAZING!!! Check http://x.com <br> Not bad at all."
    print(f"Before: {demo}")
    print(f"After : {clean_text(demo)}")
