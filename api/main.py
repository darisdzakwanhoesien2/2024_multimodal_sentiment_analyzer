from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routers import diarize, health

app = FastAPI(
    title="Speaker Diarization & Transcription API",
    description="Diarization (pyannote.audio) + transcription (faster-whisper) as a job-based API.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(diarize.router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
