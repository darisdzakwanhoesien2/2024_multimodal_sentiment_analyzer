from pathlib import Path
import tempfile
import zipfile
import io

import streamlit as st

from new_app.utils.video_processing import extract_unique_frames
import os

# Determine project root and downloads directory (robust when running from repo root or via streamlit)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
# Ensure downloads dir exists (may be empty)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


st.title("Extract Slides from Video")

# Allow choosing between uploading or selecting an existing file
source = st.radio("Source", ["Upload new file", "Use existing resource"], index=0)

uploaded_file = None
selected_existing = None
existing_files = sorted(
    [p.name for p in DOWNLOADS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
)

if source == "Upload new file":
    uploaded_file = st.file_uploader("Upload a presentation video", type=["mp4", "mov", "mkv", "avi"])
else:
    if not existing_files:
        st.info(f"No video files found in {DOWNLOADS_DIR}. You can upload one instead.")
    else:
        selected_existing = st.selectbox("Select a file from downloads", existing_files)
        if selected_existing:
            st.write(f"Selected: {selected_existing}")
            st.video(str(DOWNLOADS_DIR / selected_existing))

threshold = st.slider(
    "Change threshold",
    min_value=0.01,
    max_value=0.30,
    value=0.08,
    step=0.01,
    help="Lower values capture more slide changes. Higher values are stricter.",
)

frame_step = st.slider(
    "Analyze every Nth frame",
    min_value=1,
    max_value=30,
    value=10,
    help="Higher values are faster but may miss very short slide transitions.",
)

min_gap_seconds = st.slider(
    "Minimum seconds between saved slides",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="Helps avoid saving duplicate frames during short playback overlays or transitions.",
)

cols_per_row = st.slider(
    "Slides per row",
    min_value=1,
    max_value=4,
    value=2,
    help="Number of slides to display per row.",
)

if source == "Upload new file" and uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)
    video_path_to_process = temp_video_path

elif source == "Use existing resource" and selected_existing:
    video_path_to_process = str(DOWNLOADS_DIR / selected_existing)
else:
    video_path_to_process = None

if st.button("Process video", type="primary"):
    if not video_path_to_process:
        st.error("No video selected. Upload a file or choose an existing resource.")
    else:
        with st.spinner("Detecting slide changes..."):
            results = extract_unique_frames(
                video_path_to_process,
                threshold=threshold,
                frame_step=frame_step,
                min_time_between_slides=min_gap_seconds,
            )

        if not results:
            st.warning("No slide changes were detected with the current settings.")
        else:
            st.success(f"Detected {len(results)} slide(s).")

            # --- Download All as ZIP ---
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, item in enumerate(results):
                    slide_path = Path(item["frame"])
                    zf.writestr(
                        f"slide_{idx + 1:03d}_t{item['timestamp']}s{slide_path.suffix}",
                        slide_path.read_bytes(),
                    )
            zip_buffer.seek(0)

            st.download_button(
                label=f"⬇️ Download All {len(results)} Slides as ZIP",
                data=zip_buffer,
                file_name=f"{Path(video_path_to_process).stem}_slides.zip",
                mime="application/zip",
            )

            st.divider()

            # --- Grid Display ---
            for row_start in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, item in enumerate(results[row_start: row_start + cols_per_row]):
                    slide_num = row_start + col_idx + 1
                    with cols[col_idx]:
                        st.image(
                            item["frame"],
                            caption=(
                                f"Slide {slide_num} | {item['timestamp']}s"
                                f" | Score: {item['score']:.3f}"
                            ),
                            use_column_width=True,
                        )
                        slide_path = Path(item["frame"])
                        st.download_button(
                            label=f"⬇️ Download Slide {slide_num}",
                            data=slide_path.read_bytes(),
                            file_name=f"slide_{slide_num:03d}_t{item['timestamp']}s{slide_path.suffix}",
                            mime="image/jpeg",
                            key=f"dl_{slide_num}",
                        )
