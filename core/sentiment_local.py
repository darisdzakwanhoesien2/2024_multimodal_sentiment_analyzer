import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from core.lexicons import load_all_lexicons

lemmatizer = WordNetLemmatizer()
lexicons = load_all_lexicons()


def _lemmatize_words(text: str):
    return [lemmatizer.lemmatize(w.lower()) for w in text.split() if w.strip()]


def compute_lexicon_scores(text: str):
    words = _lemmatize_words(text)
    total = len(words)
    results = {}
    if lexicons.empty:
        # return zeros if lexicons not available
        return results

    for d in lexicons["dict"].unique():
        df = lexicons[lexicons["dict"] == d]
        pos = set(df[df["sentiment"] == "positive"]["word"])
        neg = set(df[df["sentiment"] == "negative"]["word"])
        p = sum(w in pos for w in words)
        n = sum(w in neg for w in words)
        results[f"{d}_positive"] = p / total if total else 0.0
        results[f"{d}_negative"] = n / total if total else 0.0
        results[f"{d}_net"] = (p - n) / total if total else 0.0
    return results


def apply_local_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lexicon-based features and append them to DataFrame."""
    lex_scores = df["text"].fillna("").apply(compute_lexicon_scores)
    lex_df = pd.DataFrame(lex_scores.tolist()).fillna(0.0)
    return pd.concat([df.reset_index(drop=True), lex_df.reset_index(drop=True)], axis=1)
