import os
import re

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import jobs
from ..schemas import JobCreatedResponse, JobStatusResponse, JobSummary

router = APIRouter(prefix="/api", tags=["diarization"])

ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".avi"}
YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/|embed/|v/)|youtu\.be/)[\w-]+"
)


@router.get("/youtube/channel")
def list_channel(url: str, limit: int = 50, force_refresh: bool = False):
    from ..core import db
    from ..core.youtube import YoutubeDownloadError, is_channel_url, list_channel_videos, normalize_channel_url

    if not is_channel_url(url):
        raise HTTPException(400, "That doesn't look like a youtube.com channel URL (e.g. youtube.com/@handle).")

    normalized = normalize_channel_url(url)
    if not force_refresh:
        cached = db.get_cached_channel(normalized)
        if cached is not None:
            return cached

    try:
        result = list_channel_videos(url, limit=min(limit, 100))
    except YoutubeDownloadError as e:
        raise HTTPException(502, str(e)) from e

    db.save_channel_cache(result["channel_url"], result["channel_title"], result["videos"])
    fresh = db.get_cached_channel(result["channel_url"])
    fresh["cached"] = False  # just fetched live, not served from a pre-existing cache entry
    return fresh


@router.get("/youtube/channels")
def list_channel_history():
    from ..core import db
    return db.list_cached_channels()


@router.get("/youtube/videos")
def list_video_history():
    from ..core import db
    return db.list_all_cached_videos()


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
    try:
        job_id = jobs.create_job(
            params, file_name=file_name, file_bytes=file_bytes,
            youtube_url=youtube_url or None,
        )
    except jobs.DiskSpaceError as e:
        raise HTTPException(507, str(e)) from e
    except jobs.ShuttingDownError as e:
        raise HTTPException(503, str(e)) from e
    return JobCreatedResponse(job_id=job_id, status="queued")


@router.post("/downloads", response_model=JobCreatedResponse, status_code=202)
def create_download(youtube_url: str = Form(...)):
    youtube_url = youtube_url.strip()
    if not YOUTUBE_URL_RE.match(youtube_url):
        raise HTTPException(400, "That doesn't look like a youtube.com or youtu.be video URL.")
    try:
        job_id = jobs.create_download_job(youtube_url)
    except jobs.DiskSpaceError as e:
        raise HTTPException(507, str(e)) from e
    except jobs.ShuttingDownError as e:
        raise HTTPException(503, str(e)) from e
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
        error=job["error"], file_name=job["file_name"], kind=job.get("kind", "diarize"),
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


@router.get("/jobs/{job_id}/audio")
def get_job_audio(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    path = jobs.find_media(job_id)
    if path is None or path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        raise HTTPException(404, "No audio stored for this job.")
    media_type = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }[path.suffix.lower()]
    return FileResponse(path, media_type=media_type, filename=f"{job['file_name']}{path.suffix}")


@router.get("/jobs/{job_id}/video")
def get_job_video(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    path = jobs.find_media(job_id)
    video_exts = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
                  ".avi": "video/x-msvideo", ".webm": "video/webm"}
    if path is None or path.suffix.lower() not in video_exts:
        raise HTTPException(404, "No video stored for this job.")
    return FileResponse(path, media_type=video_exts[path.suffix.lower()],
                         filename=f"{job['file_name']}{path.suffix}")


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
