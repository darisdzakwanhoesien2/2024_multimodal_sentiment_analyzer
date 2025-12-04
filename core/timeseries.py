import numpy as np
import pandas as pd


def create_time_series(df):
    rows = []

    metrics = [
        c for c in df.columns
        if any(key in c for key in ["vader", "pos", "neg", "net"])
    ]

    for _, row in df.iterrows():
        for t in np.arange(row["start"], row["end"] + 1, 1):
            d = {"time": t}
            for m in metrics:
                d[m] = row[m]
            rows.append(d)

    return pd.DataFrame(rows)
