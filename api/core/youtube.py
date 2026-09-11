from pathlib import Path

# Server-side cookies.txt (Netscape format, exported from a logged-in browser
# session), read if present so yt-dlp can pass YouTube's bot check from this
# VPS's IP. Never sent over the API; the operator places it here manually.
# Lives under api/data/, which is gitignored.
COOKIES_FILE = Path(__file__).resolve().parent.parent / "data" / "youtube_cookies.txt"


class YoutubeDownloadError(RuntimeError):
    pass


def _base_ydl_opts() -> dict:
    opts = {
        "noplaylist": True,
        "quiet": True,
        # YouTube's "n" parameter challenge now requires executing a JS
        # solver to unlock full-quality formats; without this, extraction
        # fails outright with "The page needs to be reloaded". node is
        # already present on this host, so no extra runtime install needed.
        # The solver script itself is a small yt-dlp-maintained component
        # fetched from GitHub on first use, hence remote_components.
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
    }
    if COOKIES_FILE.exists():
        opts["cookiefile"] = str(COOKIES_FILE)
    return opts


def _run_download(ydl_opts: dict, url: str, what: str):
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=True)
    except Exception as e:
        msg = str(e).lower()
        if "sign in to confirm" in msg or "cookies" in msg:
            raise YoutubeDownloadError(
                "YouTube is asking for an authenticated session (bot check) for this video, and "
                f"this server has no cookies configured to pass it. Export a cookies.txt from a "
                f"logged-in browser session and place it at {COOKIES_FILE} on the server, then retry."
            ) from e
        raise YoutubeDownloadError(f"Failed to download {what}: {e}") from e


def download_audio(url: str, out_dir: Path) -> tuple[Path, str | None]:
    """Download audio-only from a YouTube URL as mp3 (kept as the job's persisted
    audio afterward, so mp3 rather than wav to stay compact). Returns (mp3_path, video_title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        **_base_ydl_opts(),
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "5",
        }],
    }
    info = _run_download(ydl_opts, url, "audio")

    mp3_path = out_dir / "source.mp3"
    if not mp3_path.exists():
        raise YoutubeDownloadError("Download succeeded but no audio file was produced.")
    return mp3_path, (info or {}).get("title")


def download_video(url: str, out_dir: Path) -> tuple[Path, str | None]:
    """Download the full video (best video+audio, merged to mp4). Returns (path, video_title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        **_base_ydl_opts(),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "video.%(ext)s"),
    }
    info = _run_download(ydl_opts, url, "video")

    video_path = next(out_dir.glob("video.*"), None)
    if video_path is None:
        raise YoutubeDownloadError("Download succeeded but no video file was produced.")
    return video_path, (info or {}).get("title")
