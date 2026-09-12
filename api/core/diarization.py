import os

from .audio import load_waveform_for_diarization

_pipeline_cache: dict = {}

# pyannote defaults to embedding_batch_size=1 / segmentation_batch_size=1 —
# processing one chunk at a time through the neural nets with no batching.
# Raising this is a pure efficiency win (fewer, larger forward passes;
# better use of vectorized CPU math), not an accuracy trade-off — batching
# doesn't change what's computed, only how many chunks go through per call.
_BATCH_SIZE = 32


def _configure_torch_threads():
    import torch
    torch.set_num_threads(os.cpu_count() or 1)


class DiarizationError(RuntimeError):
    pass


def load_pipeline(token: str, repo_id: str):
    if not token:
        raise DiarizationError("A Hugging Face token is required.")

    if repo_id in _pipeline_cache:
        return _pipeline_cache[repo_id]

    _configure_torch_threads()

    from pyannote.audio import Pipeline

    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
    except Exception as e:
        raise DiarizationError(f"Hugging Face login failed: {e}") from e

    try:
        pipeline = Pipeline.from_pretrained(repo_id, token=token)
    except Exception as e:
        err = str(e)
        if "403" in err:
            raise DiarizationError(
                f"Access denied (403) for `{repo_id}`. Visit the model page on Hugging Face "
                "and click 'Agree and access repository'."
            ) from e
        if "404" in err:
            raise DiarizationError(f"Model `{repo_id}` not found (404).") from e
        raise DiarizationError(f"Failed to load `{repo_id}`: {e}") from e

    try:
        pipeline.embedding_batch_size = _BATCH_SIZE
        pipeline.segmentation_batch_size = _BATCH_SIZE
    except AttributeError:
        pass  # older/different pipeline variant without these knobs — harmless to skip

    _pipeline_cache[repo_id] = pipeline
    return pipeline


def _iter_diarization(diar):
    def _seg_obj(s):
        if hasattr(s, "start") and hasattr(s, "end"):
            return s
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            return type("Seg", (), {"start": float(s[0]), "end": float(s[1])})()
        raise TypeError("Unsupported segment type")

    if hasattr(diar, "itertracks"):
        yield from diar.itertracks(yield_label=True)
        return

    segs = getattr(diar, "segments", None) or getattr(diar, "segments_", None)
    labs = getattr(diar, "labels", None) or getattr(diar, "labels_", None)
    if segs is not None and labs is not None:
        for s, l in zip(segs, labs):
            yield _seg_obj(s), None, l
        return

    if isinstance(diar, dict):
        for key in ("annotation", "diarization", "output", "result"):
            if key in diar:
                yield from _iter_diarization(diar[key])
                return

    attrs = getattr(diar, "__dict__", {})
    for name, val in attrs.items():
        if name.startswith("_"):
            continue
        if hasattr(val, "itertracks"):
            yield from val.itertracks(yield_label=True)
            return
        if isinstance(val, (list, tuple)) and val:
            first = val[0]
            if hasattr(first, "start") and hasattr(first, "end"):
                for s in val:
                    label = getattr(s, "label", None) or getattr(s, "speaker", None) or "SPEAKER_0"
                    yield _seg_obj(s), None, label
                return

    raise RuntimeError(
        f"Unsupported diarization output (type={type(diar)}). "
        f"Attributes: {list(getattr(diar, '__dict__', {}).keys())}"
    )


def run_diarization(pipeline, audio_path: str, num_spk: int, min_spk: int, max_spk: int):
    """Returns (rows, rttm_str). rows: list of {start, end, duration, speaker} sorted by start."""
    kwargs = {}
    if num_spk > 0:
        kwargs["num_speakers"] = num_spk
    else:
        if min_spk > 0:
            kwargs["min_speakers"] = min_spk
        if max_spk > 0:
            kwargs["max_speakers"] = max_spk

    waveform, sample_rate = load_waveform_for_diarization(audio_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)

    rows = []
    rttm_lines = []
    for segment, _, label in _iter_diarization(diarization):
        start = round(float(segment.start), 3)
        end = round(float(segment.end), 3)
        dur = round(end - start, 3)
        rows.append({"start": start, "end": end, "duration": dur, "speaker": label})
        rttm_lines.append(f"SPEAKER file 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>")

    rows.sort(key=lambda r: r["start"])
    rttm_str = "\n".join(rttm_lines) + "\n"
    return rows, rttm_str


def speaker_stats(rows: list) -> list:
    totals: dict = {}
    counts: dict = {}
    for r in rows:
        totals[r["speaker"]] = totals.get(r["speaker"], 0.0) + r["duration"]
        counts[r["speaker"]] = counts.get(r["speaker"], 0) + 1
    grand_total = sum(totals.values()) or 1.0
    stats = [
        {
            "speaker": spk,
            "segments": counts[spk],
            "total_time": round(totals[spk], 2),
            "avg_segment": round(totals[spk] / counts[spk], 2),
            "percentage": round(totals[spk] / grand_total * 100, 1),
        }
        for spk in totals
    ]
    stats.sort(key=lambda s: s["total_time"], reverse=True)
    return stats
