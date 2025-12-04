import pandas as pd
import numpy as np
import requests
from io import StringIO
from nltk.corpus import opinion_lexicon
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


def load_huliu():
    pos_url = "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt"
    neg_url = "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt"

    pos = pd.read_csv(pos_url, sep="\n", header=None, skiprows=35, names=["word"])
    neg = pd.read_csv(neg_url, sep="\n", header=None, skiprows=35, names=["word"])

    pos["sentiment"] = "positive"
    neg["sentiment"] = "negative"

    df = pd.concat([pos, neg])
    df["dict"] = "huliu"
    return df


def load_afinn():
    url = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
    df = pd.read_csv(url, sep="\t", names=["word", "score"])
    df["sentiment"] = np.where(df["score"] < 0, "negative", "positive")
    df["dict"] = "afinn"
    return df[["word", "sentiment", "dict"]]


def load_bing():
    url = "https://raw.githubusercontent.com/dinbav/LeXmo/master/R/sysdata.rda"
    content = requests.get(url).content.decode("latin-1")
    df = pd.read_csv(StringIO(content), sep="\t", on_bad_lines="skip")
    df = df[["word", "sentiment"]]
    df["dict"] = "bing"
    return df


def load_nltk_opinion():
    pos = list(opinion_lexicon.positive())
    neg = list(opinion_lexicon.negative())

    df = pd.DataFrame(
        {"word": pos + neg,
         "sentiment": ["positive"] * len(pos) + ["negative"] * len(neg)}
    )
    df["dict"] = "nltk"
    return df


def load_all_lexicons():
    dfs = [load_huliu(), load_afinn(), load_bing(), load_nltk_opinion()]
    df = pd.concat(dfs)
    df["word"] = df["word"].astype(str).apply(lemmatizer.lemmatize)
    df = df.drop_duplicates()
    return df
