# core/dataframe_utils.py
import json
import pandas as pd


def convert_json_to_dataframe(folder_name, json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "segments" not in data:
        raise ValueError("No 'segments' key in transcription JSON")

    rows = []
    for seg in data["segments"]:
        rows.append({
            "segment_id": seg.get("id"),
            "seek": seg.get("seek"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
            "tokens": seg.get("tokens"),
            "temperature": seg.get("temperature"),
            "segment_avg_logprob": seg.get("avg_logprob"),
            "segment_compression_ratio": seg.get("compression_ratio"),
            "segment_no_speech_prob": seg.get("no_speech_prob"),
        })

    return pd.DataFrame(rows)
