from fastapi import APIRouter, Form

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
