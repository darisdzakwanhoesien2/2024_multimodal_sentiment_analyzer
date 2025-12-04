import json
import time
import os
import numpy as np
from openai import OpenAI
import streamlit as st

def _get_api_key():
    return os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

API_KEY = _get_api_key()
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Add it to environment or Streamlit secrets.")

client = OpenAI(api_key=API_KEY)


def get_sentiment_once(text: str, system_prompt: str = None):
    """
    Single call to OpenAI for JSON-only sentiment output.
    Expects the model to return JSON like: {"sentiment_score": 7}
    """
    system_prompt = system_prompt or "Return JSON only with a numeric 'sentiment_score' between 1 and 10."
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=200,
        )
        content = r.choices[0].message.content
        # attempt to extract JSON safely
        parsed = json.loads(content)
        return parsed.get("sentiment_score") if isinstance(parsed, dict) else None
    except Exception as e:
        # log and return None on failure
        print("OpenAI sentiment call failed:", e)
        return None


def run_iterative_sentiment(df, iterations=3, delay=1.0):
    """
    For each text row, call get_sentiment_once() 'iterations' times.
    Adds columns: openai_scores (list), openai_mean, openai_std
    """
    scores_all = []
    for text in df["text"].fillna("").tolist():
        scores = []
        for _ in range(iterations):
            s = get_sentiment_once(text)
            scores.append(s)
            time.sleep(delay)
        scores_all.append(scores)

    df = df.copy()
    df["openai_scores"] = scores_all
    df["openai_mean"] = df["openai_scores"].apply(
        lambda lst: np.mean([v for v in lst if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in lst) else None
    )
    df["openai_std"] = df["openai_scores"].apply(
        lambda lst: np.std([v for v in lst if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in lst) else None
    )
    return df
