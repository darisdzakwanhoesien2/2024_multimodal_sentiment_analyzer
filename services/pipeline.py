from core.downloader import download_video
from core.audio import extract_audio
from core.whisper_api import transcribe, save_transcription
from core.segmentation import transcription_json_to_df
from core.sentiment_local import apply_local_sentiment
from core.sentiment_openai import run_iterative_sentiment
from core.timeseries import create_time_series
from core.plots import plot_metrics
from core.utils import make_dir, save_json
import os


def process_youtube(url: str, run_openai_sentiment: bool = False, openai_iters: int = 3):
    """
    Orchestrator: download -> extract audio -> whisper -> segmentation -> local sentiment -> optional OpenAI sentiment -> timeseries -> plots
    Returns: dict with folder path and DataFrame (pandas)
    """
    folder, video_path = download_video(url)
    if not folder:
        raise RuntimeError("Failed to download video info or create folder.")

    if not video_path:
        raise RuntimeError("Failed to download video file.")

    # Extract audio
    audio_path = extract_audio(video_path, folder)

    # Transcribe
    transcription = transcribe(audio_path)  # raw response
    json_path = os.path.join(folder, "transcription.json")
    save_transcription(transcription, json_path)

    # Convert to DataFrame
    df = transcription_json_to_df(json_path)
    if df is None:
        raise RuntimeError("Failed to convert transcription JSON to DataFrame.")

    # Local sentiment (lexicons)
    df = apply_local_sentiment(df)

    # Optional OpenAI iterative sentiment
    if run_openai_sentiment:
        df = run_iterative_sentiment(df, iterations=openai_iters, delay=1.0)

    # Time series expansion
    ts = create_time_series(df)

    # Plot metrics
    plot_path = os.path.join(folder, "metrics.png")
    plot_metrics(ts, plot_path)

    # Save CSV and JSON summaries
    try:
        df.to_csv(os.path.join(folder, "transcription_result.csv"), index=False)
    except Exception as e:
        print("Warning: could not save CSV:", e)

    try:
        save_json(df.fillna("").to_dict(orient="records"), os.path.join(folder, "transcription_result_records.json"))
    except Exception as e:
        print("Warning: could not save JSON records:", e)

    return {"folder": folder, "video": video_path, "audio": audio_path, "dataframe": df, "timeseries": ts, "plot": plot_path}
