from openai import OpenAI
import os
import json
import streamlit as st

def _get_api_key():
    return os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)


API_KEY = _get_api_key()
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Add it to environment or Streamlit secrets.")

client = OpenAI(api_key=API_KEY)


def transcribe(audio_file: str, model: str = "whisper-1"):
    """
    Transcribe an audio file using OpenAI Whisper API.
    Returns the raw response (parsed JSON-like dict).
    """
    with open(audio_file, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json"
        )
    return resp


def save_transcription(result, output_path: str):
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
