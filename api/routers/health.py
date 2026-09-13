from fastapi import APIRouter, Form, HTTPException, UploadFile

from ..schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])

VERIFY_MODELS = [
    "pyannote/speaker-diarization-3.0",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]


@router.get("/health", response_model=HealthResponse)
def health():
    packages = {}
    try:
        import torch
        packages["torch"] = torch.__version__
    except Exception as e:
        packages["torch"] = f"unavailable ({e})"
    try:
        import pyannote.audio
        packages["pyannote.audio"] = pyannote.audio.__version__
    except Exception as e:
        packages["pyannote.audio"] = f"unavailable ({e})"
    try:
        import faster_whisper
        packages["faster_whisper"] = getattr(faster_whisper, "__version__", "unknown")
    except Exception as e:
        packages["faster_whisper"] = f"unavailable ({e})"
    try:
        import huggingface_hub
        packages["huggingface_hub"] = huggingface_hub.__version__
    except Exception as e:
        packages["huggingface_hub"] = f"unavailable ({e})"

    return HealthResponse(status="ok", packages=packages)


@router.get("/youtube/health")
def youtube_health():
    from ..core.youtube import check_cookies_health
    return check_cookies_health()


@router.post("/youtube/cookies")
def upload_cookies(file: UploadFile):
    # A plain (non-async) handler: FastAPI runs these in Starlette's own
    # thread pool, entirely separate from jobs.py's single-worker executor
    # that serializes diarization/download jobs. Uploading fresh cookies is
    # never blocked by (and never blocks) a job that's currently running —
    # it just won't retroactively affect a job already mid-download, since
    # that job already read the old file when it started.
    from ..core.youtube import check_cookies_health, save_cookies_file, validate_cookies_file

    content = file.file.read()
    error = validate_cookies_file(content)
    if error:
        raise HTTPException(400, error)

    save_cookies_file(content)
    return check_cookies_health()


@router.post("/hf/verify")
def verify_hf_token(hf_token: str = Form(...)):
    from huggingface_hub import login as hf_login
    from huggingface_hub import model_info, whoami

    try:
        hf_login(token=hf_token, add_to_git_credential=False)
        identity = whoami(token=hf_token)
    except Exception as e:
        return {"token_valid": False, "error": str(e), "models": []}

    results = []
    for repo_id in VERIFY_MODELS:
        try:
            info = model_info(repo_id, token=hf_token)
            results.append({
                "model": repo_id,
                "accessible": True,
                "detail": f"gated={getattr(info, 'gated', 'N/A')}",
            })
        except Exception as e:
            err = str(e)
            if "403" in err:
                detail = "Access denied (403) - accept the model license on huggingface.co"
            elif "404" in err:
                detail = "Not found (404)"
            else:
                detail = err[:200]
            results.append({"model": repo_id, "accessible": False, "detail": detail})

    return {
        "token_valid": True,
        "identity": identity.get("name"),
        "models": results,
    }
