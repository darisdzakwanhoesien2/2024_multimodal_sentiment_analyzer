# pipeline/run_lexicon_pipeline.py
import os
import pandas as pd

from core.dataframe_utils import convert_json_to_dataframe
from core.lexicon_sentiment import calculate_sentiment_scores
from core.plotting import create_time_series, plot_metrics


def run_lexicon_pipeline(dataset_path):
    dataset_path = str(dataset_path)
    json_path = os.path.join(dataset_path, "transcription_result.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError("transcription_result.json not found")

    df = convert_json_to_dataframe(dataset_path, json_path)

    scores = df["text"].apply(calculate_sentiment_scores)
    scores_df = pd.DataFrame(scores.tolist())

    merged = pd.concat([df, scores_df], axis=1)

    csv_out = os.path.join(dataset_path, "transcription_result.csv")
    merged.to_csv(csv_out, index=False)

    ts_df = create_time_series(merged)
    plot_metrics(ts_df, os.path.join(dataset_path, "metrics_plot.png"))

    return merged
