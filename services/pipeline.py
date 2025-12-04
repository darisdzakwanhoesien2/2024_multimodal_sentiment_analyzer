from core.downloader import download_video
from core.audio import extract_audio
from core.whisper_api import transcribe, save_transcription
from core.segmentation import transcription_json_to_df
from core.sentiment_local import apply_local_sentiment
from core.timeseries import create_time_series
from core.plots import plot_metrics
import os


def process_youtube(url):
    folder, video = download_video(url)

    audio = extract_audio(video, folder)

    result = transcribe(audio)
    json_path = os.path.join(folder, "transcription.json")
    save_transcription(result, json_path)

    df = transcription_json_to_df(json_path)
    df = apply_local_sentiment(df)

    ts = create_time_series(df)
    plot_metrics(ts, os.path.join(folder, "metrics.png"))

    return folder, df
