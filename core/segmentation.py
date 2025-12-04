import json
import pandas as pd

def transcription_json_to_df(json_path):
    data = json.load(open(json_path))
    segments = data["segments"]

    df = pd.DataFrame([
        {
            "id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "tokens": s["tokens"],
            "avg_logprob": s.get("avg_logprob"),
            "compression_ratio": s.get("compression_ratio"),
            "no_speech_prob": s.get("no_speech_prob"),
        }
        for s in segments
    ])
    return df