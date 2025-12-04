import json
import pandas as pd
from typing import Optional


def transcription_json_to_df(json_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Failed to load transcription JSON:", e)
        return None

    segments = data.get("segments", [])
    rows = []
    for s in segments:
        rows.append({
            "segment_id": s.get("id"),
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("text", "").strip(),
            "tokens": s.get("tokens"),
            "segment_avg_logprob": s.get("avg_logprob"),
            "segment_compression_ratio": s.get("compression_ratio"),
            "segment_no_speech_prob": s.get("no_speech_prob"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
