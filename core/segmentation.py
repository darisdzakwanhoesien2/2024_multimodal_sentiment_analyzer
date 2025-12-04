import json
import pandas as pd


def transcription_json_to_df(json_path):
    """Convert Whisper JSON to DataFrame."""
    data = json.load(open(json_path))
    segments = data.get("segments", [])

    rows = []
    for s in segments:
        rows.append({
            "id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "tokens": s["tokens"],
            "avg_logprob": s.get("avg_logprob"),
            "compression_ratio": s.get("compression_ratio"),
            "no_speech_prob": s.get("no_speech_prob"),
        })

    return pd.DataFrame(rows)
