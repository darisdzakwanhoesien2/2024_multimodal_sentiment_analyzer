import ffmpeg
import os


def extract_audio(video_path, output_dir):
    """
    Extract MP3 audio from a video file using FFmpeg.
    Fully compatible with Python 3.13 on Streamlit Cloud.
    """
    audio_path = os.path.join(output_dir, "audio.mp3")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format="mp3", acodec="libmp3lame", audio_bitrate="192k")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError("FFmpeg failed extracting audio: " + e.stderr.decode())

    return audio_path
