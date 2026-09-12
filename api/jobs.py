import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core.align import align_transcript_with_speakers
from .core.audio import VIDEO_EXTS as _VIDEO_EXTS
from .core.audio import extract_audio_if_video, get_media_duration
from .core.diarization import DiarizationError, load_pipeline, run_diarization, speaker_stats
from .core.transcription import TranscriptionError, run_transcription
from .core.youtube import YoutubeDownloadError, download_audio, download_video

logger = logging.getLogger(__name__)

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

_shutting_down = threading.Event()


class ShuttingDownError(RuntimeError):
    pass


def prepare_for_shutdown():
    """Called from main.py's shutdown handler. Stops accepting new jobs and
    blocks until whatever is currently running finishes, so a deploy restart
    doesn't kill a 15-45 minute job mid-run. Requires the systemd unit's
    TimeoutStopSec to be raised well above its default (~90s) — see
    video-diarization-api.service — or systemd will SIGKILL before this
    returns anyway."""
    _shutting_down.set()
    _executor.shutdown(wait=True)


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
        {"job_id": j["job_id"], "status": j["status"], "file_name": j["file_name"],
         "created_at": j["created_at"], "kind": j.get("kind", "diarize")}
        for j in jobs
    ]


def create_job(params: dict, file_name: str | None = None, file_bytes: bytes | None = None,
               youtube_url: str | None = None) -> str:
    if _shutting_down.is_set():
        raise ShuttingDownError("Server is restarting for a deploy — try again in a minute.")
    if not file_bytes and not youtube_url:
        raise ValueError("Either a file or a youtube_url must be provided.")
    check_disk_space()

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    created_at = time.time()
    display_name = file_name or youtube_url
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "file_name": display_name, "source_url": youtube_url,
        "created_at": created_at, "kind": "diarize",
    }))

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
            "file_name": display_name,
            "source_url": youtube_url,
            "created_at": created_at,
            "result": None,
            "kind": "diarize",
        }

    _executor.submit(_process_job, job_id, str(input_path) if input_path else None, job_dir, params, youtube_url)
    return job_id


def create_download_job(youtube_url: str) -> str:
    if _shutting_down.is_set():
        raise ShuttingDownError("Server is restarting for a deploy — try again in a minute.")
    check_disk_space()

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    created_at = time.time()
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "file_name": youtube_url, "source_url": youtube_url,
        "created_at": created_at, "kind": "download",
    }))

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": "queued",
            "error": None,
            "file_name": youtube_url,
            "source_url": youtube_url,
            "created_at": created_at,
            "result": None,
            "kind": "download",
        }

    _executor.submit(_process_download_job, job_id, job_dir, youtube_url)
    return job_id


def _process_download_job(job_id: str, job_dir: Path, youtube_url: str):
    video_path = None
    succeeded = False
    try:
        _set(job_id, status="running", progress="downloading video from YouTube")
        video_path, title = download_video(youtube_url, job_dir)
        if title:
            _set(job_id, file_name=title)
            _update_meta(job_dir, file_name=title)

        result = {"video_filename": video_path.name, "title": title}
        (job_dir / "result.json").write_text(json.dumps(result))
        succeeded = True
        _set(job_id, status="done", progress="done", result=result)

    except YoutubeDownloadError as e:
        _set(job_id, status="error", progress="failed", error=str(e))
    except Exception as e:
        _set(job_id, status="error", progress="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if not succeeded and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


def _process_job(job_id: str, input_path: str | None, job_dir: Path, params: dict, youtube_url: str | None = None):
    audio_path = None
    video_path = None
    succeeded = False
    try:
        if youtube_url:
            # Download the full video (not audio-only) so it survives as the
            # job's persisted media for in-page playback alongside the
            # transcript, same as an uploaded video does. Audio for the
            # pipeline is extracted from it via the same helper used for
            # video uploads below.
            _set(job_id, status="running", progress="downloading video from YouTube")
            video_path, title = download_video(youtube_url, job_dir)
            if title:
                _set(job_id, file_name=title)
                _update_meta(job_dir, file_name=title)
            video_path = str(video_path)

            _set(job_id, status="running", progress="extracting audio")
            audio_path = extract_audio_if_video(video_path, str(job_dir / "audio.mp3"))
        else:
            _set(job_id, status="running", progress="extracting audio")
            audio_path = extract_audio_if_video(input_path, str(job_dir / "audio.mp3"))
            if audio_path != input_path:
                # input_path was a video and got extracted to audio_path above;
                # keep the original video itself as the persisted media (it
                # already contains that audio) rather than the extraction.
                video_path = input_path

        _set(job_id, progress=f"loading diarization pipeline ({params['model_choice']})")
        pipeline = load_pipeline(params["hf_token"], params["model_choice"])

        _set(job_id, progress="running diarization")
        _t0 = time.time()
        rows, rttm_str = run_diarization(
            pipeline, audio_path,
            params["num_speakers"], params["min_speakers"], params["max_speakers"],
        )
        logger.info("Diarization took %.1fs for job %s (%s)", time.time() - _t0, job_id, params["model_choice"])

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

        media_for_duration = video_path if (video_path and os.path.exists(video_path)) else audio_path
        result = {
            "segments": rows,
            "speaker_stats": speaker_stats(rows),
            "num_speakers": len({r["speaker"] for r in rows}),
            "transcript_text": transcript_text,
            "transcript_segments": transcript_segments,
            "speaker_aligned_transcript": aligned,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "has_video": bool(video_path and os.path.exists(video_path)),
            "has_audio": bool(not video_path and audio_path and os.path.exists(audio_path)),
            "total_duration": get_media_duration(media_for_duration) if media_for_duration else None,
        }
        (job_dir / "result.json").write_text(json.dumps(result))
        succeeded = True
        _set(job_id, status="done", progress="done", result=result)

    except (DiarizationError, TranscriptionError, YoutubeDownloadError) as e:
        _set(job_id, status="error", progress="failed", error=str(e))
    except Exception as e:
        _set(job_id, status="error", progress="failed", error=f"{type(e).__name__}: {e}")
    finally:
        # On success: if a video exists (youtube_url, or an uploaded video),
        # it's the one persisted artifact — it already contains the audio, so
        # the extracted audio.mp3 intermediate is redundant and gets dropped.
        # Otherwise (audio-only source), audio_path is what's kept. On
        # failure, nothing is worth keeping — everything gets dropped.
        keep_path = video_path if (succeeded and video_path) else (audio_path if succeeded else None)
        for path in {input_path, audio_path, video_path}:
            if path and path != keep_path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


_NON_MEDIA_NAMES = {"result.rttm", "result.json", "transcript.txt", "meta.json"}


def find_media(job_id: str) -> Path | None:
    d = job_dir(job_id)
    if not d.is_dir():
        return None
    for candidate in sorted(d.iterdir()):
        if candidate.is_file() and candidate.name not in _NON_MEDIA_NAMES:
            return candidate
    return None


def _update_meta(dir_path: Path, **fields):
    meta_path = dir_path / "meta.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update(fields)
        meta_path.write_text(json.dumps(meta))
    except (OSError, json.JSONDecodeError):
        pass


def _parse_rttm(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = round(float(parts[3]), 3)
        dur = round(float(parts[4]), 3)
        rows.append({"start": start, "end": round(start + dur, 3), "duration": dur, "speaker": parts[7]})
    return rows


def _rehydrate_job(dir_path: Path) -> dict | None:
    job_id = dir_path.name
    meta_path = dir_path / "meta.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        meta = {}
    file_name = meta.get("file_name") or job_id
    source_url = meta.get("source_url")
    created_at = meta.get("created_at") or dir_path.stat().st_mtime

    kind = meta.get("kind", "diarize")
    base = {"job_id": job_id, "file_name": file_name, "source_url": source_url,
            "created_at": created_at, "kind": kind}

    result_json = dir_path / "result.json"
    if result_json.exists():
        try:
            result = json.loads(result_json.read_text())
        except (OSError, json.JSONDecodeError):
            result = None
        if result is not None:
            return {**base, "status": "done", "progress": "done", "error": None, "result": result}

    rttm_path = dir_path / "result.rttm"
    if rttm_path.exists():
        # Legacy job from before result.json existed: reconstruct what's
        # recoverable from rttm + transcript.txt. Word-level timing wasn't
        # persisted separately back then, so transcript_segments and
        # speaker_aligned_transcript can't be rebuilt and come back empty.
        rows = _parse_rttm(rttm_path.read_text())
        transcript_path = dir_path / "transcript.txt"
        transcript_text = transcript_path.read_text() if transcript_path.exists() else None
        media_path = find_media(job_id)
        is_video = media_path is not None and media_path.suffix.lower() in _VIDEO_EXTS
        result = {
            "segments": rows,
            "speaker_stats": speaker_stats(rows) if rows else [],
            "num_speakers": len({r["speaker"] for r in rows}),
            "transcript_text": transcript_text,
            "transcript_segments": None,
            "speaker_aligned_transcript": None,
            "detected_language": None,
            "language_probability": None,
            "has_video": is_video,
            "has_audio": media_path is not None and not is_video,
            "total_duration": get_media_duration(str(media_path)) if media_path else (rows[-1]["end"] if rows else None),
        }
        return {**base, "status": "done", "progress": "done", "error": None, "result": result}

    # No output artifacts: this job was still queued/running when the server
    # stopped. Nothing safe to resume from — drop any orphaned raw audio it
    # left behind and record the interruption instead of silently vanishing.
    for f in dir_path.iterdir():
        if f.is_file() and f.name != "meta.json":
            try:
                f.unlink()
            except OSError:
                pass
    return {**base, "status": "error", "progress": "failed",
            "error": f"Job was interrupted by a server restart before finishing. Please re-submit: {file_name}",
            "result": None}


def _rehydrate_from_disk():
    if not DATA_DIR.is_dir():
        return
    count = 0
    for entry in DATA_DIR.iterdir():
        if not entry.is_dir():
            continue
        try:
            job = _rehydrate_job(entry)
        except Exception as e:
            logger.warning("Failed to rehydrate job %s: %s", entry.name, e)
            continue
        if job:
            _jobs[job["job_id"]] = job
            count += 1
    if count:
        logger.info("Rehydrated %d job(s) from disk.", count)


_rehydrate_from_disk()
