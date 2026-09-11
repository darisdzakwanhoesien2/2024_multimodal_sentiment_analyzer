import logging
import os
import subprocess

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}


def extract_audio_if_video(input_path: str, out_path: str) -> str:
    """If input_path is a video file, extract mono 16kHz audio as a compact mp3 at
    out_path (inside the job dir, so it survives as the persisted result) and
    return out_path. Otherwise return input_path unchanged — an already-audio
    upload is kept exactly as given, no re-encoding."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in VIDEO_EXTS:
        return input_path

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000",
         "-codec:a", "libmp3lame", "-qscale:a", "4", out_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return out_path


def get_media_duration(path: str) -> float | None:
    """Total duration in seconds via ffprobe, or None if it can't be read.
    Used to scale the frontend timeline to the real media length rather than
    the last detected speech segment's end time, which falls short whenever
    there's trailing silence/non-speech after the last diarized segment."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        return round(float(result.stdout.strip()), 3)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


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
