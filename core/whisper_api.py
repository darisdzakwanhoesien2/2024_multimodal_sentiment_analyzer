from openai import OpenAI
import json
import os

client = OpenAI()

def transcribe(audio_file):
    with open(audio_file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            response_format="verbose_json"
        )
    return result

def save_transcription(result, output_path):
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)