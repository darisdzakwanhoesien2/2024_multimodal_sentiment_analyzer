import os
import re

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import jobs
from ..schemas import JobCreatedResponse, JobStatusResponse, JobSummary

router = APIRouter(prefix="/api", tags=["diarization"])

ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".avi"}
YOUTUBE_URL_RE = re.compile(r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+")


@router.post("/jobs", response_model=JobCreatedResponse, status_code=202)
async def create_job(
    file: UploadFile | None = None,
    youtube_url: str = Form(""),
    hf_token: str = Form(...),
    model_choice: str = Form("pyannote/speaker-diarization-3.1"),
    num_speakers: int = Form(0),
    min_speakers: int = Form(0),
    max_speakers: int = Form(0),
    do_transcribe: bool = Form(False),
    whisper_model_size: str = Form("small"),
    whisper_device: str = Form("cpu"),
    whisper_compute_type: str = Form("int8"),
    whisper_language: str = Form(""),
):
    youtube_url = youtube_url.strip()
    has_file = file is not None and file.filename
    if has_file and youtube_url:
        raise HTTPException(400, "Provide either a file or a YouTube URL, not both.")
    if not has_file and not youtube_url:
        raise HTTPException(400, "Provide either a file or a YouTube URL.")

    file_name, file_bytes = None, None
    if has_file:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}")
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(400, "Uploaded file is empty.")
        file_name = file.filename
    else:
        if not YOUTUBE_URL_RE.match(youtube_url):
            raise HTTPException(400, "That doesn't look like a youtube.com or youtu.be video URL.")

    params = dict(
        hf_token=hf_token,
        model_choice=model_choice,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        do_transcribe=do_transcribe,
        whisper_model_size=whisper_model_size,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_language=whisper_language,
    )
    job_id = jobs.create_job(
        params, file_name=file_name, file_bytes=file_bytes,
        youtube_url=youtube_url or None,
    )
    return JobCreatedResponse(job_id=job_id, status="queued")


@router.get("/jobs", response_model=list[JobSummary])
def list_jobs():
    return jobs.list_jobs()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job["job_id"], status=job["status"], progress=job["progress"],
        error=job["error"], file_name=job["file_name"],
    )


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is not finished (status={job['status']})")
    return job["result"]


@router.get("/jobs/{job_id}/rttm")
def get_job_rttm(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is not finished (status={job['status']})")
    path = jobs.job_dir(job_id) / "result.rttm"
    if not path.exists():
        raise HTTPException(404, "RTTM file not found")
    return FileResponse(path, media_type="text/plain", filename=f"{job['file_name']}.rttm")


@router.get("/jobs/{job_id}/transcript.txt")
def get_job_transcript(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is not finished (status={job['status']})")
    result = job["result"]
    if not result.get("transcript_text"):
        raise HTTPException(404, "No transcript available for this job (transcription was not requested).")
    return PlainTextResponse(result["transcript_text"])
