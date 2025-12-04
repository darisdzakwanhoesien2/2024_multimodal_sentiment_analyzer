from core.downloader import download_video
from core.audio import extract_audio
from core.whisper_api import transcribe, save_transcription
from core.segmentation import transcription_json_to_df
from core.sentiment_local import apply_local_sentiment
from core.timeseries import create_time_series
from core.plots import plot_metrics

def process_youtube(url):
    folder, video = download_video(url)
    audio = extract_audio(video, folder)
    result = transcribe(audio)
    save_transcription(result, f"{folder}/transcription.json")

    df = transcription_json_to_df(f"{folder}/transcription.json")
    df = apply_local_sentiment(df)

    ts = create_time_series(df)
    plot_metrics(ts, f"{folder}/metrics.png")

    return folder, df