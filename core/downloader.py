import yt_dlp
import os
import re
import json

def folder_name_from_title(title: str):
    return re.sub(r'\W+', '_', title.lower())

def get_video_info(url: str):
    try:
        with yt_dlp.YoutubeDL() as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
        return None

def download_video(url: str, base_dir="public/packages"):
    info = get_video_info(url)
    if not info:
        return None, None

    title = info.get("title", "video")
    folder = folder_name_from_title(title)
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

    # Save info
    with open(os.path.join(folder_path, "video_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    outtmpl = os.path.join(folder_path, "video.mp4")

    opts = {"outtmpl": outtmpl, "format": "mp4"}

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:
        return folder_path, None

    return folder_path, outtmpl