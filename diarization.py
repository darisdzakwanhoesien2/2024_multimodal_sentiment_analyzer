import numpy as np
import librosa

def get_chunk(audio_array, sample_rate, start_time, end_time):
    """
    Extract audio chunk with padding to ensure expected sample count.
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    expected_samples = int((end_time - start_time) * sample_rate)

    chunk = audio_array[start_sample:end_sample]
    actual_samples = len(chunk)

    # Pad with zeros if chunk is shorter than expected
    if actual_samples < expected_samples:
        padding = np.zeros(expected_samples - actual_samples, dtype=audio_array.dtype)
        chunk = np.concatenate([chunk, padding])
    # Trim if chunk is longer than expected
    elif actual_samples > expected_samples:
        chunk = chunk[:expected_samples]

    return chunk


def process_diarization_chunks(audio_path, chunk_duration=10.0):
    """
    Process audio in chunks for diarization with sample count validation.
    """
    audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    total_samples = len(audio_array)
    total_duration = total_samples / sample_rate
    expected_chunk_samples = int(chunk_duration * sample_rate)

    chunks = []
    start_time = 0.0

    while start_time < total_duration:
        end_time = min(start_time + chunk_duration, total_duration)

        chunk = get_chunk(audio_array, sample_rate, start_time, end_time)

        # Pad the last chunk if it's shorter than the full chunk duration
        if len(chunk) < expected_chunk_samples:
            padding = np.zeros(
                expected_chunk_samples - len(chunk), dtype=audio_array.dtype
            )
            chunk = np.concatenate([chunk, padding])

        chunks.append(chunk)
        start_time += chunk_duration

    return chunks, sample_rate