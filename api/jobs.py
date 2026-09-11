import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core.align import align_transcript_with_speakers
from .core.audio import extract_audio_if_video
from .core.diarization import DiarizationError, load_pipeline, run_diarization, speaker_stats
from .core.transcription import TranscriptionError, run_transcription
from .core.youtube import YoutubeDownloadError, download_audio

DATA_DIR = Path(__file__).resolve().parent / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Successful jobs now keep their audio permanently (previously deleted after
# processing) so results survive a restart. This VPS runs several other
# production services off the same disk with very little headroom, so refuse
# new jobs outright below this floor rather than risk filling it.
MIN_FREE_BYTES = 750 * 1024 * 1024


class DiskSpaceError(RuntimeError):
    pass


def check_disk_space():
    free = shutil.disk_usage(DATA_DIR).free
    if free < MIN_FREE_BYTES:
        raise DiskSpaceError(
            f"Only {free / 1e6:.0f}MB free on the server — refusing to start a new job "
            f"to protect other services sharing this disk. Ask the operator to free up space."
        )


_lock = threading.Lock()
_jobs: dict = {}
# Heavy CPU inference is serialized to one job at a time: this VPS runs several
# other shared services and loading two pyannote/whisper models concurrently
# risks exhausting memory.
_executor = ThreadPoolExecutor(max_workers=1)


def _set(job_id: str, **fields):
    with _lock:
        _jobs[job_id].update(fields)


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return [
        {"job_id": j["job_id"], "status": j["status"], "file_name": j["file_name"], "created_at": j["created_at"]}
        for j in jobs
    ]


def create_job(params: dict, file_name: str | None = None, file_bytes: bytes | None = None,
               youtube_url: str | None = None) -> str:
    if not file_bytes and not youtube_url:
        raise ValueError("Either a file or a youtube_url must be provided.")
    check_disk_space()

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = None
    if file_bytes is not None:
        ext = os.path.splitext(file_name or "")[1].lower() or ".wav"
        input_path = job_dir / f"input{ext}"
        input_path.write_bytes(file_bytes)

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": "queued",
            "error": None,
            "file_name": file_name or youtube_url,
            "source_url": youtube_url,
            "created_at": time.time(),
            "result": None,
        }

    _executor.submit(_process_job, job_id, str(input_path) if input_path else None, job_dir, params, youtube_url)
    return job_id


def _process_job(job_id: str, input_path: str | None, job_dir: Path, params: dict, youtube_url: str | None = None):
    audio_path = None
    succeeded = False
    try:
        if youtube_url:
            _set(job_id, status="running", progress="downloading audio from YouTube")
            audio_path, title = download_audio(youtube_url, job_dir)
            if title:
                _set(job_id, file_name=title)
            audio_path = str(audio_path)
        else:
            _set(job_id, status="running", progress="extracting audio")
            audio_path = extract_audio_if_video(input_path, str(job_dir / "audio.mp3"))

        _set(job_id, progress=f"loading diarization pipeline ({params['model_choice']})")
        pipeline = load_pipeline(params["hf_token"], params["model_choice"])

        _set(job_id, progress="running diarization")
        rows, rttm_str = run_diarization(
            pipeline, audio_path,
            params["num_speakers"], params["min_speakers"], params["max_speakers"],
        )

        transcript_text = None
        transcript_segments = None
        aligned = None
        detected_language = None
        language_probability = None

        if params["do_transcribe"]:
            _set(job_id, progress=f"transcribing ({params['whisper_model_size']})")
            transcript_text, transcript_segments, detected_language, language_probability = run_transcription(
                audio_path,
                model_size_or_path=params["whisper_model_size"],
                device=params["whisper_device"],
                compute_type=params["whisper_compute_type"],
                language=params["whisper_language"] or None,
            )
            aligned = align_transcript_with_speakers(rows, transcript_segments)

        rttm_path = job_dir / "result.rttm"
        rttm_path.write_text(rttm_str)

        if transcript_text is not None:
            (job_dir / "transcript.txt").write_text(transcript_text)

        result = {
            "segments": rows,
            "speaker_stats": speaker_stats(rows),
            "num_speakers": len({r["speaker"] for r in rows}),
            "transcript_text": transcript_text,
            "transcript_segments": transcript_segments,
            "speaker_aligned_transcript": aligned,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "has_audio": bool(audio_path and os.path.exists(audio_path)),
        }
        succeeded = True
        _set(job_id, status="done", progress="done", result=result)

    except (DiarizationError, TranscriptionError, YoutubeDownloadError) as e:
        _set(job_id, status="error", progress="failed", error=str(e))
    except Exception as e:
        _set(job_id, status="error", progress="failed", error=f"{type(e).__name__}: {e}")
    finally:
        # On success, keep audio_path (the file the pipeline actually used —
        # either the original upload, or an extracted/downloaded mp3 already
        # written inside job_dir) so results survive a restart. input_path is
        # only removed here when it's a separate raw file distinct from
        # audio_path (e.g. an uploaded video that got extracted to mp3) — no
        # reason to keep the raw video too. On failure, nothing is worth
        # keeping, so both are dropped.
        if input_path and input_path != audio_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass
        if not succeeded and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


_NON_AUDIO_NAMES = {"result.rttm", "transcript.txt"}


def find_audio(job_id: str) -> Path | None:
    d = job_dir(job_id)
    if not d.is_dir():
        return None
    for candidate in sorted(d.iterdir()):
        if candidate.is_file() and candidate.name not in _NON_AUDIO_NAMES:
            return candidate
    return None
