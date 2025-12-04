import seaborn as sns
import matplotlib.pyplot as plt


def plot_metrics(ts_df, output_path):
    plt.figure(figsize=(14, 8))
    metrics = [c for c in ts_df.columns if c not in ["time"]]

    for m in metrics:
        sns.lineplot(data=ts_df, x="time", y=m, label=m)

    plt.title("Sentiment Metrics Over Time")
    plt.savefig(output_path, dpi=300)
    plt.close()
