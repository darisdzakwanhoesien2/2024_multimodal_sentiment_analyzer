import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}


def extract_audio_if_video(input_path: str) -> str:
    """If input_path is a video file, extract mono 16kHz audio to a new wav and return its path.
    Otherwise return input_path unchanged."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in VIDEO_EXTS:
        return input_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", out_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return out_path


def load_waveform_for_diarization(audio_path: str):
    """Load, downmix to mono, resample to 16kHz and pad to a whole number of 10s
    chunks. Returns (waveform: torch.Tensor[1, T], sample_rate: int) for pyannote's
    dict input form, deliberately avoiding torchaudio.load()/torchcodec: this VPS
    has no CUDA runtime, and torchcodec's image decoder unconditionally tries to
    load libnvrtc and fails at import time without it."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    data, sample_rate = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)

    target_sr = 16000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    chunk_samples = target_sr * 10
    remainder = waveform.shape[1] % chunk_samples
    if remainder != 0:
        waveform = torch.nn.functional.pad(waveform, (0, chunk_samples - remainder))

    return waveform, sample_rate
