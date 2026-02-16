# core/transcriber_whisper_local.py

def whisper_local_transcribe(audio_file: str, model_name: str = "large-v3"):
    """
    Transcribe audio using faster-whisper + Whisper-large-v3.
    Returns dict similar to Whisper API result:
      {"transcription_text": ..., "segments": [ {id, start, end, text, tokens=[]}, ... ]}
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("faster-whisper not installed. Install via 'pip install faster-whisper'") from e

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_file, beam_size=5, vad_filter=True,
                                      vad_parameters={"min_silence_duration_ms": 500})

    full_text = []
    out_segments = []
    for idx, seg in enumerate(segments):
        txt = seg.text.strip()
        if not txt:
            continue
        full_text.append(txt)
        out_segments.append({
            "id": idx,
            "start": seg.start,
            "end": seg.end,
            "text": txt,
            "tokens": []
        })

    return {"transcription_text": " ".join(full_text), "segments": out_segments}
