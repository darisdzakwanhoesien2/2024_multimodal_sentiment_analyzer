import json
import numpy as np
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()


def get_sentiment(text):
    """Single OpenAI sentiment call, JSON-only output."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Respond ONLY with valid JSON containing {\"sentiment_score\": number}."},
                {"role": "user", "content": text}
            ]
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {"sentiment_score": None}


def run_iterative_sentiment(df, iterations=5, sleep=1):
    all_results = []

    for text in tqdm(df["text"], desc="OpenAI sentiment"):
        scores = []
        for _ in range(iterations):
            s = get_sentiment(text)
            scores.append(s.get("sentiment_score"))
            time.sleep(sleep)
        all_results.append(scores)

    df["openai_scores"] = all_results
    df["openai_mean"] = df["openai_scores"].apply(
        lambda x: np.mean([v for v in x if v is not None]) if x else None
    )
    df["openai_std"] = df["openai_scores"].apply(
        lambda x: np.std([v for v in x if v is not None]) if x else None
    )
    return df
