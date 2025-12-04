from pydub import AudioSegment
import os


def extract_audio(video_path, output_dir):
    """Extract audio from MP4 and save as MP3."""
    audio_path = os.path.join(output_dir, "audio.mp3")
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export(audio_path, format="mp3")
    return audio_path
