from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.downloader import download_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "downloads"
DEFAULT_FILENAME = "%(title)s.%(ext)s"
COOKIE_BROWSER_OPTIONS = {
    "None": None,
    "Chrome": "chrome",
    "Chromium": "chromium",
    "Brave": "brave",
    "Edge": "edge",
    "Firefox": "firefox",
    "Safari": "safari",
}


def resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


st.set_page_config(
    page_title="YouTube Downloader",
    page_icon="video_camera",
    layout="centered",
)

st.title("YouTube Video Downloader")
st.write("Paste a YouTube URL and save the video as an MP4 file in this app's downloads folder.")

with st.form("download-form"):
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    output_dir = st.text_input("Output folder", value=str(DEFAULT_OUTPUT_DIR))
    filename_template = st.text_input("Filename template", value=DEFAULT_FILENAME)
    cookies_from_browser_label = st.selectbox(
        "Use browser cookies",
        options=list(COOKIE_BROWSER_OPTIONS.keys()),
        help=(
            "If YouTube asks you to sign in, choose the browser where you're already "
            "logged into YouTube so yt-dlp can reuse that session."
        ),
    )
    cookies_file = st.text_input(
        "Or cookies.txt file path",
        placeholder="/path/to/cookies.txt",
        help="Optional exported cookies file. Leave blank if you're using browser cookies instead.",
    )
    submitted = st.form_submit_button("Download")

if submitted:
    if not url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        resolved_output_dir = resolve_output_dir(output_dir.strip() or str(DEFAULT_OUTPUT_DIR))
        resolved_cookies_file = Path(cookies_file).expanduser() if cookies_file.strip() else None
        try:
            with st.spinner("Downloading video..."):
                saved_path = download_video(
                    url=url.strip(),
                    output_dir=resolved_output_dir,
                    filename_template=filename_template.strip() or DEFAULT_FILENAME,
                    cookies_from_browser=COOKIE_BROWSER_OPTIONS[cookies_from_browser_label],
                    cookies_file=resolved_cookies_file,
                )
        except ModuleNotFoundError:
            st.error("Missing dependency: install packages from requirements.txt first.")
        except Exception as exc:
            st.error(f"Download failed: {exc}")
        else:
            st.success(f"Downloaded successfully: {saved_path}")
            if saved_path.exists():
                with saved_path.open("rb") as video_file:
                    st.download_button(
                        label="Download saved MP4",
                        data=video_file,
                        file_name=saved_path.name,
                        mime="video/mp4",
                    )
                st.video(str(saved_path))

st.caption(f"Default save location: {DEFAULT_OUTPUT_DIR}")
st.caption("Make sure you have permission to download the content you save.")
