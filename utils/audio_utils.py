import numpy as np
import soundfile as sf
import librosa

def read_chunk_fixed(path, start_sec, end_sec, target_sr=44100):
    """
    Read audio chunk [start_sec, end_sec) from path, resample to target_sr if needed,
    and ensure the returned numpy array has exactly expected_samples = (end_sec-start_sec)*target_sr
    by padding with zeros or trimming.
    """
    expected_samples = int(round((end_sec - start_sec) * target_sr))

    # read entire file (soundfile gives exact frames)
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # mix to mono

    # resample if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # compute sample range
    start_sample = int(np.round(start_sec * sr))
    end_sample = int(np.round(end_sec * sr))

    # extract, pad or trim
    if start_sample >= len(data):
        chunk = np.zeros(expected_samples, dtype="float32")
    else:
        chunk = data[start_sample:end_sample]
        if len(chunk) < expected_samples:
            pad_len = expected_samples - len(chunk)
            chunk = np.concatenate([chunk, np.zeros(pad_len, dtype="float32")])
        elif len(chunk) > expected_samples:
            chunk = chunk[:expected_samples]

    return chunk