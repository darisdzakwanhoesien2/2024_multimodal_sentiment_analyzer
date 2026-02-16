# core/transcriber_google.py
import os
import tempfile
import json
import math
import ffmpeg
import speech_recognition as sr
from typing import Dict, List

CHUNK_SECONDS = 50  # safe chunk size for Google's free recognizer

def _wav_from_audio(input_audio_path: str, out_wav_path: str):
    # convert to 16k mono wav for best ASR results
    (
        ffmpeg
        .input(input_audio_path)
        .output(out_wav_path, ar=16000, ac=1, format='wav')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_wav_path

def _split_wav(wav_path: str, out_dir: str, chunk_seconds: int = CHUNK_SECONDS) -> List[str]:
    # use ffmpeg segment muxer to split into chunk_seconds segments
    template = os.path.join(out_dir, "chunk_%04d.wav")
    # create segments with start times resetting
    (
        ffmpeg
        .input(wav_path)
        .output(template, f='segment', segment_time=chunk_seconds, reset_timestamps=1, ar=16000, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )
    # collect produced chunk files
    chunks = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("chunk_") and f.endswith(".wav")])
    return chunks

def google_transcribe(audio_file: str) -> Dict:
    """
    Transcribe using SpeechRecognition (Google free web API).
    Returns a dict with structure similar to Whisper verbose_json:
    {
      "transcription_text": "...",
      "segments": [
         {"id":0,"start":0.0,"end":xx,"text": "...", "tokens": []},
         ...
      ]
    }
    """
    tmpdir = tempfile.mkdtemp(prefix="google_trans_")
    wav_path = os.path.join(tmpdir, "audio.wav")
    _wav_from_audio(audio_file, wav_path)
    chunks = _split_wav(wav_path, tmpdir)

    recognizer = sr.Recognizer()
    all_text = []
    segments = []
    start_time = 0.0

    for idx, chunk_path in enumerate(chunks):
        with sr.AudioFile(chunk_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            # Google service might be blocked or rate-limited
            raise RuntimeError(f"Google Speech API error: {e}")

        # estimate end time
        # get duration via ffmpeg probe
        probe = ffmpeg.probe(chunk_path)
        duration = float(next((s for s in probe['streams'] if s['codec_type']=='audio'), {}) .get('duration', 0.0) or 0.0)
        end_time = start_time + duration

        segments.append({
            "id": idx,
            "start": start_time,
            "end": end_time,
            "text": text,
            "tokens": []
        })

        all_text.append(text)
        start_time = end_time

    result = {
        "transcription_text": " ".join([t for t in all_text if t]),
        "segments": segments
    }
    return result
