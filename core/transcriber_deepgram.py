# core/transcriber_deepgram.py
import os
import requests
import json

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?punctuate=true&utterances=false"

def deepgram_transcribe(audio_file: str, api_key: str = None):
    """
    Transcribe using Deepgram (REST). Requires API key.
    Returns dict: {"transcription_text": "...", "segments":[...]}
    """
    api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("Deepgram API key missing. Set DEEPGRAM_API_KEY in env or streamlit secrets.")

    headers = {"Authorization": f"Token {api_key}"}
    # send binary data - Deepgram supports streaming or single file
    with open(audio_file, "rb") as fh:
        resp = requests.post(DEEPGRAM_URL, headers=headers, data=fh, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Deepgram API error: {resp.status_code} - {resp.text}")

    data = resp.json()
    # deepgram returns a structure with 'results' or 'channels'
    # We'll collect alternatives if present
    transcript = ""
    segments = []
    # try generic path
    try:
        channels = data.get("results", {}).get("channels", [])
        if channels:
            alt = channels[0].get("alternatives", [])[0]
            transcript = alt.get("transcript", "")
            # deepgram may include words with timestamps
            words = alt.get("words", [])
            # form segments: group contiguous words into sentence-like segments (simple approach)
            if words:
                cur = {"id": 0, "start": words[0]['start'], "end": words[0]['end'], "text": words[0]['word'], "tokens": []}
                seg_id = 0
                for w in words[1:]:
                    # if large gap, create new segment
                    if w['start'] - cur['end'] > 1.0:
                        segments.append(cur)
                        seg_id += 1
                        cur = {"id": seg_id, "start": w['start'], "end": w['end'], "text": w['word'], "tokens": []}
                    else:
                        cur['text'] = cur['text'] + " " + w['word']
                        cur['end'] = w['end']
                segments.append(cur)
    except Exception:
        # fallback: try data.get('transcript')
        transcript = data.get("transcript", "") or ""
    return {"transcription_text": transcript, "segments": segments}
