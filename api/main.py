import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from . import jobs
from .routers import diarize, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # On SIGTERM (systemctl restart/stop), uvicorn calls this after it stops
    # accepting new connections. Block here until any in-flight diarization/
    # transcription job finishes, so a deploy doesn't kill a 15-45 minute job
    # partway through — see jobs.prepare_for_shutdown() and this service's
    # systemd unit (TimeoutStopSec must be raised to match, or systemd
    # SIGKILLs before this returns).
    logger.info("Shutting down: waiting for any in-flight job to finish...")
    jobs.prepare_for_shutdown()
    logger.info("Shutdown drain complete.")


app = FastAPI(
    title="Speaker Diarization & Transcription API",
    description="Diarization (pyannote.audio) + transcription (faster-whisper) as a job-based API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def no_edge_cache_for_frontend(request: Request, call_next):
    # Cloudflare caches static extensions (.js/.css) at the edge by default even
    # without an explicit Cache-Control from origin, which previously served a
    # stale app.js to visitors for hours after a deploy. Force revalidation.
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.include_router(health.router)
app.include_router(diarize.router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
