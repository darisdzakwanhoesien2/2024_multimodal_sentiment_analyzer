import numpy as np
import pandas as pd


def create_time_series(df):
    """
    Expand segment-level rows to second-by-second (integer seconds).
    Keeps numeric columns that look like lexicon metrics (contain 'positive', 'negative', 'net', 'openai')
    """
    metrics = [c for c in df.columns if any(k in c for k in ["positive", "negative", "net", "openai"])]
    rows = []
    for _, row in df.iterrows():
        start = int(row.get("start", 0) or 0)
        end = int(row.get("end", start) or start)
        if end < start:
            end = start
        for t in range(start, end + 1):
            r = {"time": t}
            for m in metrics:
                r[m] = row.get(m)
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["time"] + metrics)
    return pd.DataFrame(rows)
