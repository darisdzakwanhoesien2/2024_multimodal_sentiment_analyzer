# core/lexicon_sentiment.py
import numpy as np
from nltk.corpus import opinion_lexicon
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()

positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())


def calculate_sentiment_scores(text: str):
    words = text.lower().split()
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    total = len(lemmas) if lemmas else 1

    pos = sum(w in positive_words for w in lemmas)
    neg = sum(w in negative_words for w in lemmas)

    scores = {
        "nltk_opinion_lexicon_positive": pos / total,
        "nltk_opinion_lexicon_negative": neg / total,
        "nltk_opinion_lexicon_net": (pos - neg) / total,
    }

    scores.update({
        f"vader_{k}": v
        for k, v in vader.polarity_scores(text).items()
    })

    return scores
