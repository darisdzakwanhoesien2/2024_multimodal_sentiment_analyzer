import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_metrics(ts_df, output_path):
    if ts_df is None or ts_df.empty:
        return None
    plt.figure(figsize=(12, 6))
    metrics = [c for c in ts_df.columns if c != "time"]
    for m in metrics:
        sns.lineplot(data=ts_df, x="time", y=m, label=m)
    plt.xlabel("Time (s)")
    plt.ylabel("Metric value")
    plt.title("Metrics Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path
