from openai import OpenAI
import json
import streamlit as st
import os

# Load key from secrets or env
api_key = (
    st.secrets.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in st.secrets
    else os.environ.get("OPENAI_API_KEY")
)

if not api_key:
    raise ValueError("❗ Missing OPENAI_API_KEY in Streamlit secrets or environment")

client = OpenAI(api_key=api_key)


def transcribe(audio_file):
    """Transcribe audio using Whisper API and return the raw response object."""
    with open(audio_file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    return result


def save_transcription(result, output_path):
    """
    Save Whisper output to JSON file.
    Must convert the response to a pure Python dict first.
    """
    # Convert API object → dict
    if hasattr(result, "model_dump"):
        result_dict = result.model_dump()
    else:
        # safety fallback
        result_dict = json.loads(result.json())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, ensure_ascii=False, indent=2)

# from openai import OpenAI
# import os
# import json
# import streamlit as st

# def _get_api_key():
#     return os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)


# API_KEY = _get_api_key()
# if not API_KEY:
#     raise ValueError("OPENAI_API_KEY is missing. Add it to environment or Streamlit secrets.")

# client = OpenAI(api_key=API_KEY)


# def transcribe(audio_file: str, model: str = "whisper-1"):
#     """
#     Transcribe an audio file using OpenAI Whisper API.
#     Returns the raw response (parsed JSON-like dict).
#     """
#     with open(audio_file, "rb") as f:
#         resp = client.audio.transcriptions.create(
#             model=model,
#             file=f,
#             response_format="verbose_json"
#         )
#     return resp


# def save_transcription(result, output_path: str):
#     with open(output_path, "w", encoding="utf-8") as fh:
#         json.dump(result, fh, ensure_ascii=False, indent=2)
