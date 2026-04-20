from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.downloader import download_video


DEFAULT_OUTPUT_DIR = "downloads"
DEFAULT_FILENAME = "%(title)s.%(ext)s"


st.set_page_config(
    page_title="YouTube Downloader",
    page_icon="video_camera",
    layout="centered",
)

st.title("YouTube Video Downloader")
st.write("Paste a YouTube URL and save the video as an MP4 file.")

with st.form("download-form"):
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    output_dir = st.text_input("Output folder", value=DEFAULT_OUTPUT_DIR)
    filename_template = st.text_input("Filename template", value=DEFAULT_FILENAME)
    submitted = st.form_submit_button("Download")

if submitted:
    if not url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        try:
            with st.spinner("Downloading video..."):
                saved_path = download_video(
                    url=url.strip(),
                    output_dir=Path(output_dir.strip() or DEFAULT_OUTPUT_DIR),
                    filename_template=filename_template.strip() or DEFAULT_FILENAME,
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

st.caption("Make sure you have permission to download the content you save.")
