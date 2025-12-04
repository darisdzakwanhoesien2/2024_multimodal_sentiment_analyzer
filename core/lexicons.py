import pandas as pd
import numpy as np
import requests
from io import StringIO
import nltk
import os
from nltk.stem import WordNetLemmatizer

# Ensure NLTK runtime downloads go to local folder to avoid permission issues
NLTK_LOCAL = os.environ.get("NLTK_DATA_DIR", "/tmp/nltk_data")
os.makedirs(NLTK_LOCAL, exist_ok=True)
nltk.data.path.append(NLTK_LOCAL)

# Ensure necessary NLTK corpora
for res in ("wordnet", "stopwords", "opinion_lexicon"):
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, download_dir=NLTK_LOCAL, quiet=True)

lemmatizer = WordNetLemmatizer()


def load_huliu():
    pos_url = "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt"
    neg_url = "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt"
    try:
        pos = pd.read_csv(pos_url, sep="\n", header=None, skiprows=35, names=["word"], encoding="utf-8")
        neg = pd.read_csv(neg_url, sep="\n", header=None, skiprows=35, names=["word"], encoding="utf-8")
    except Exception:
        pos = pd.read_csv(pos_url, sep="\n", header=None, skiprows=35, names=["word"], encoding="latin-1")
        neg = pd.read_csv(neg_url, sep="\n", header=None, skiprows=35, names=["word"], encoding="latin-1")

    pos["sentiment"] = "positive"
    neg["sentiment"] = "negative"
    df = pd.concat([pos, neg], ignore_index=True)
    df["dict"] = "huliu"
    df["word"] = df["word"].astype(str).str.strip().str.lower().map(lemmatizer.lemmatize)
    return df[["word", "sentiment", "dict"]]


def load_afinn():
    url = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
    df = pd.read_csv(url, sep="\t", names=["word", "score"], encoding="utf-8")
    df["sentiment"] = np.where(df["score"] < 0, "negative", "positive")
    df["dict"] = "afinn"
    df["word"] = df["word"].astype(str).str.strip().str.lower().map(lemmatizer.lemmatize)
    return df[["word", "sentiment", "dict"]]


def load_bing():
    # fallback attempt; this resource is less stable — handle failures
    try:
        url = "https://raw.githubusercontent.com/dinbav/LeXmo/master/R/sysdata.rda"
        content = requests.get(url, timeout=10).content.decode("latin-1")
        df = pd.read_csv(StringIO(content), sep="\t", on_bad_lines="skip")
        if "word" in df.columns and "sentiment" in df.columns:
            df = df[["word", "sentiment"]]
            df["dict"] = "bing"
            df["word"] = df["word"].astype(str).str.strip().str.lower().map(lemmatizer.lemmatize)
            return df
    except Exception:
        pass
    # return empty if not available
    return pd.DataFrame(columns=["word", "sentiment", "dict"])


def load_nltk_opinion():
    from nltk.corpus import opinion_lexicon
    pos = list(opinion_lexicon.positive())
    neg = list(opinion_lexicon.negative())
    df = pd.DataFrame({"word": pos + neg,
                       "sentiment": ["positive"] * len(pos) + ["negative"] * len(neg)})
    df["dict"] = "nltk_opinion"
    df["word"] = df["word"].astype(str).str.strip().str.lower().map(lemmatizer.lemmatize)
    return df[["word", "sentiment", "dict"]]


def load_all_lexicons():
    dfs = []
    try:
        dfs.append(load_huliu())
    except Exception:
        pass
    try:
        dfs.append(load_afinn())
    except Exception:
        pass
    try:
        b = load_bing()
        if not b.empty:
            dfs.append(b)
    except Exception:
        pass
    try:
        dfs.append(load_nltk_opinion())
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame(columns=["word", "sentiment", "dict"])
    df = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return df
