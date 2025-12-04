import yt_dlp
import os
import re
import json


def folder_name_from_title(title: str) -> str:
    return re.sub(r'\W+', '_', title.lower())


def get_video_info(url: str):
    try:
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        print("yt-dlp info error:", e)
        return None


def download_video(url: str, base_dir="public/packages"):
    """Download a YouTube video and return folder path and video file path."""
    info = get_video_info(url)
    if not info:
        return None, None

    title = info.get("title", "video")
    folder = folder_name_from_title(title)
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

    # Save metadata
    try:
        with open(os.path.join(folder_path, "video_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Could not write metadata:", e)

    video_output = os.path.join(folder_path, "video.mp4")
    ydl_opts = {"outtmpl": video_output, "format": "mp4"}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print("yt-dlp download error:", e)
        return folder_path, None

    return folder_path, video_output
