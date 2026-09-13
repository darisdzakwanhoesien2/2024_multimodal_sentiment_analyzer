import re
from pathlib import Path

# Server-side cookies.txt (Netscape format, exported from a logged-in browser
# session), read if present so yt-dlp can pass YouTube's bot check from this
# VPS's IP. Never sent over the API; the operator places it here manually.
# Lives under api/data/, which is gitignored.
COOKIES_FILE = Path(__file__).resolve().parent.parent / "data" / "youtube_cookies.txt"


class YoutubeDownloadError(RuntimeError):
    pass


def validate_cookies_file(content: bytes) -> str | None:
    """Returns an error message if content doesn't look like a Netscape-format
    cookies.txt with YouTube entries, else None."""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        return "File isn't valid text."
    if not text.strip():
        return "File is empty."
    if "youtube.com" not in text:
        return "This doesn't look like a YouTube cookies file (no youtube.com entries found)."
    data_lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not data_lines:
        return "No cookie entries found in the file."
    if not all(len(line.split("\t")) == 7 for line in data_lines[:5]):
        return "This doesn't look like a Netscape-format cookies.txt (expected tab-separated fields)."
    return None


def save_cookies_file(content: bytes):
    """Atomic write (temp file + rename) so a job mid-read of the old cookies
    file (yt-dlp reads it fully when it starts) never sees a half-written
    file, regardless of exactly when this lands relative to that read."""
    import os

    COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = COOKIES_FILE.with_suffix(".txt.tmp")
    tmp_path.write_bytes(content)
    os.replace(tmp_path, COOKIES_FILE)


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


# "Me at the zoo" — the first video ever uploaded to YouTube. Always public,
# never taken down, and tiny — a stable, lightweight canary for checking
# whether cookies.txt still clears the bot-check, without downloading anything.
_CANARY_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def check_cookies_health() -> dict:
    """Probe whether cookies.txt is present and still valid, without
    downloading anything. Used by /api/youtube/health so staleness shows up
    proactively in the UI instead of only surfacing via a failed job."""
    import yt_dlp

    if not COOKIES_FILE.exists():
        return {"ok": False, "cookies_present": False, "detail": "No cookies.txt configured on the server."}

    opts = {**_base_ydl_opts(), "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(_CANARY_URL, download=False)
        return {"ok": True, "cookies_present": True, "detail": "Cookies are valid."}
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "sign in to confirm" in lower or "no longer valid" in lower or "cookies" in lower:
            return {
                "ok": False, "cookies_present": True,
                "detail": "Cookies are present but expired or rotated — export a fresh cookies.txt "
                          f"and replace it at {COOKIES_FILE} on the server.",
            }
        return {"ok": False, "cookies_present": True, "detail": f"Unexpected error while checking: {msg[:200]}"}


_CHANNEL_URL_RE = re.compile(
    r"^https?://(www\.)?youtube\.com/(@[\w.-]+|channel/[\w-]+|c/[\w.-]+|user/[\w.-]+)"
)
_CHANNEL_TAB_RE = re.compile(r"/(videos|shorts|streams|playlists)(/|$|\?)")


def normalize_channel_url(url: str) -> str:
    """Channels default to their Home tab, which yt-dlp can't flat-list the
    same way; append /videos when the URL doesn't already name a tab."""
    url = url.strip().rstrip("/")
    if not _CHANNEL_TAB_RE.search(url):
        url += "/videos"
    return url


def is_channel_url(url: str) -> bool:
    return bool(_CHANNEL_URL_RE.match(url.strip()))


def list_channel_videos(url: str, limit: int = 50) -> dict:
    """Flat-list a channel's videos (metadata only, nothing downloaded).
    Returns {channel_title, channel_url, videos: [{id, title, url, duration,
    thumbnail, view_count}, ...]}."""
    import yt_dlp

    opts = {**_base_ydl_opts(), "extract_flat": "in_playlist", "playlistend": limit}
    normalized = normalize_channel_url(url)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(normalized, download=False)
    except Exception as e:
        msg = str(e).lower()
        if "sign in to confirm" in msg or "cookies" in msg:
            raise YoutubeDownloadError(
                "YouTube is asking for an authenticated session (bot check) for this channel, and "
                f"this server has no cookies configured to pass it. Export a cookies.txt from a "
                f"logged-in browser session and place it at {COOKIES_FILE} on the server, then retry."
            ) from e
        raise YoutubeDownloadError(f"Failed to list channel videos: {e}") from e

    videos = []
    for entry in (info or {}).get("entries") or []:
        if not entry or not entry.get("id"):
            continue
        thumbs = entry.get("thumbnails") or []
        videos.append({
            "id": entry["id"],
            "title": entry.get("title") or entry["id"],
            "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}",
            "duration": entry.get("duration"),
            "thumbnail": thumbs[-1]["url"] if thumbs else None,
            "view_count": entry.get("view_count"),
        })

    channel_title = (info or {}).get("channel") or (info or {}).get("title") or normalized
    return {"channel_title": channel_title, "channel_url": normalized, "videos": videos}


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
