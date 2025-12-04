import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from core.lexicons import load_all_lexicons

lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()
lexicons = load_all_lexicons()


def compute_lexicon_scores(text):
    words = [lemmatizer.lemmatize(w.lower()) for w in text.split()]
    results = {}

    for d in lexicons["dict"].unique():
        df = lexicons[lexicons["dict"] == d]
        pos = set(df[df["sentiment"] == "positive"]["word"])
        neg = set(df[df["sentiment"] == "negative"]["word"])

        p = sum(w in pos for w in words)
        n = sum(w in neg for w in words)
        total = len(words)

        results[f"{d}_pos"] = p / total if total else 0
        results[f"{d}_neg"] = n / total if total else 0
        results[f"{d}_net"] = (p - n) / total if total else 0

    return results


def apply_local_sentiment(df):
    """Attach lexicon + VADER sentiment to DataFrame."""
    lex_scores = df["text"].apply(compute_lexicon_scores)
    lex_df = pd.DataFrame(lex_scores.tolist())

    vader_scores = df["text"].apply(vader.polarity_scores)
    vader_df = pd.DataFrame(vader_scores.tolist()).rename(
        columns={"compound": "vader_compound"}
    )

    return pd.concat([df, lex_df, vader_df], axis=1)
