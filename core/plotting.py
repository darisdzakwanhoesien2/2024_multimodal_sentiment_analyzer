# core/plotting.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_time_series(df):
    metrics = [
        "nltk_opinion_lexicon_positive",
        "nltk_opinion_lexicon_negative",
        "nltk_opinion_lexicon_net",
        "vader_neg", "vader_neu", "vader_pos", "vader_compound",
    ]

    rows = []
    for _, row in df.iterrows():
        start, end = row["start"], row["end"]
        duration = max(int(end - start), 1)
        times = np.linspace(start, end, duration)

        for t in times:
            rows.append({"time": t, **{m: row[m] for m in metrics}})

    return rows and __import__("pandas").DataFrame(rows)


def plot_metrics(ts_df, output_path):
    plt.figure(figsize=(14, 8))

    for col in ts_df.columns:
        if col != "time":
            sns.lineplot(data=ts_df, x="time", y=col, label=col)

    plt.title("Sentiment Metrics Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
