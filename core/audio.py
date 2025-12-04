import ffmpeg
import os


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract MP3 audio from a video file using ffmpeg-python.
    Returns path to audio file.
    """
    audio_path = os.path.join(output_dir, "audio.mp3")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format="mp3", acodec="libmp3lame", audio_bitrate="192k")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        # e.stderr is bytes in many installs
        msg = getattr(e, "stderr", None)
        if isinstance(msg, bytes):
            msg = msg.decode(errors="ignore")
        raise RuntimeError("ffmpeg failed extracting audio: " + (msg or str(e)))
    return audio_path
