def align_transcript_with_speakers(diar_rows: list, whisper_segments: list) -> list:
    """Align whisper segments to diarization speaker segments by time overlap."""
    if not diar_rows:
        return []

    def overlap(s1, e1, s2, e2):
        return max(0.0, min(e1, e2) - max(s1, s2))

    rows = []
    for row in diar_rows:
        s0, e0 = row["start"], row["end"]
        pieces = [w["text"] for w in whisper_segments if overlap(s0, e0, w["start"], w["end"]) > 0]
        rows.append({
            "start": row["start"],
            "end": row["end"],
            "duration": row["duration"],
            "speaker": row["speaker"],
            "text": " ".join(pieces).strip(),
        })
    rows.sort(key=lambda r: r["start"])
    return rows
