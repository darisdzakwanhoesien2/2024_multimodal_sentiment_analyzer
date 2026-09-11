_whisper_cache: dict = {}


class TranscriptionError(RuntimeError):
    pass


def load_whisper_model(model_size_or_path: str, device: str, compute_type: str):
    key = (model_size_or_path, device, compute_type)
    if key in _whisper_cache:
        return _whisper_cache[key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise TranscriptionError("faster-whisper is not installed.") from e

    ct = compute_type if compute_type != "default" else None
    kwargs = dict(device=device)
    if ct:
        kwargs["compute_type"] = ct

    model = WhisperModel(model_size_or_path, **kwargs)
    _whisper_cache[key] = model
    return model


def run_transcription(
    audio_path: str,
    model_size_or_path: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
):
    """Returns (full_text, segments, detected_language, language_probability)."""
    model = load_whisper_model(model_size_or_path, device, compute_type)

    kwargs = dict(beam_size=5, vad_filter=True)
    if language:
        kwargs["language"] = language

    segments_gen, info = model.transcribe(audio_path, **kwargs)

    out_segments = []
    full_text_parts = []
    for seg in segments_gen:
        out_segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    full_text = " ".join(full_text_parts)
    return full_text, out_segments, info.language, info.language_probability
