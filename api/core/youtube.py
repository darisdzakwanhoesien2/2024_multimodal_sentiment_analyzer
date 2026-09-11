from pathlib import Path

# Server-side cookies.txt (Netscape format, exported from a logged-in browser
# session), read if present so yt-dlp can pass YouTube's bot check from this
# VPS's IP. Never sent over the API; the operator places it here manually.
# Lives under api/data/, which is gitignored.
COOKIES_FILE = Path(__file__).resolve().parent.parent / "data" / "youtube_cookies.txt"


class YoutubeDownloadError(RuntimeError):
    pass


def download_audio(url: str, out_dir: Path) -> tuple[Path, str | None]:
    """Download audio-only from a YouTube URL as mp3 (kept as the job's persisted
    audio afterward, so mp3 rather than wav to stay compact). Returns (mp3_path, video_title)."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "5",
        }],
    }
    if COOKIES_FILE.exists():
        ydl_opts["cookiefile"] = str(COOKIES_FILE)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        msg = str(e).lower()
        if "sign in to confirm" in msg or "cookies" in msg:
            raise YoutubeDownloadError(
                "YouTube is asking for an authenticated session (bot check) for this video, and "
                f"this server has no cookies configured to pass it. Export a cookies.txt from a "
                f"logged-in browser session and place it at {COOKIES_FILE} on the server, then retry."
            ) from e
        raise YoutubeDownloadError(f"Failed to download audio: {e}") from e

    mp3_path = out_dir / "source.mp3"
    if not mp3_path.exists():
        raise YoutubeDownloadError("Download succeeded but no audio file was produced.")
    return mp3_path, (info or {}).get("title")


def download_video(url: str, out_dir: Path) -> tuple[Path, str | None]:
    """Download the full video (best video+audio, merged to mp4). Returns (path, video_title)."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    if COOKIES_FILE.exists():
        ydl_opts["cookiefile"] = str(COOKIES_FILE)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        msg = str(e).lower()
        if "sign in to confirm" in msg or "cookies" in msg:
            raise YoutubeDownloadError(
                "YouTube is asking for an authenticated session (bot check) for this video, and "
                f"this server has no cookies configured to pass it. Export a cookies.txt from a "
                f"logged-in browser session and place it at {COOKIES_FILE} on the server, then retry."
            ) from e
        raise YoutubeDownloadError(f"Failed to download video: {e}") from e

    video_path = next(out_dir.glob("video.*"), None)
    if video_path is None:
        raise YoutubeDownloadError("Download succeeded but no video file was produced.")
    return video_path, (info or {}).get("title")
