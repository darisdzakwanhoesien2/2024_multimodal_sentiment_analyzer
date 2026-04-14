from pathlib import Path
import tempfile
import zipfile
import io
import os
import shutil

import streamlit as st

from new_app.utils.video_processing import extract_unique_frames

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 200 MB in bytes — files larger than this use chunked streaming write
LARGE_FILE_THRESHOLD = 200 * 1024 * 1024   # 200 MB
CHUNK_SIZE           = 8 * 1024 * 1024      # 8 MB chunks

# Raise Streamlit's default 200 MB upload cap via config key (only works before set_page_config)
# Users should also set  server.maxUploadSize = 2048  in .streamlit/config.toml
# We set it here programmatically as a safety net.
try:
    st.config.set_option("server.maxUploadSize", 2048)   # 2 GB
except Exception:
    pass

# ── Page ───────────────────────────────────────────────────────────────────────
st.title("Extract Slides from Video")
st.caption(
    "Supports videos up to **2 GB**. "
    "For files > 200 MB the upload is written to disk in 8 MB chunks to avoid memory pressure."
)

# ── .streamlit/config.toml reminder ───────────────────────────────────────────
config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
if not config_path.exists():
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[server]\nmaxUploadSize = 2048\n\n[runner]\nfastReruns = true\n"
    )
    st.info(
        f"Created `{config_path}` with `maxUploadSize = 2048` (2 GB). "
        "Restart Streamlit for it to take effect."
    )

# ── Chunked write helper ───────────────────────────────────────────────────────
def write_uploaded_file_chunked(uploaded_file, dest_path: Path, chunk_size: int = CHUNK_SIZE) -> int:
    """
    Write an UploadedFile to dest_path in fixed-size chunks.
    Returns total bytes written.
    Shows a Streamlit progress bar.
    """
    total_size = uploaded_file.size  # bytes (Streamlit ≥ 1.18)
    uploaded_file.seek(0)

    progress = st.progress(0, text="Writing file to disk…")
    bytes_written = 0

    with open(dest_path, "wb") as f:
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
            pct = min(int(bytes_written / total_size * 100), 100) if total_size else 0
            progress.progress(pct, text=f"Writing… {bytes_written / 1_048_576:.1f} MB / {total_size / 1_048_576:.1f} MB")

    progress.empty()
    return bytes_written


# ── ZIP streaming helper (avoids loading all slides into RAM) ──────────────────
def stream_zip(results: list[dict], stem: str):
    """
    Yield a ZIP built from slide paths without loading everything into memory at once.
    Returns a BytesIO object (slides are typically small PNGs/JPEGs — safe).
    For very large slide counts (>500) consider writing to a temp file instead.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for idx, item in enumerate(results):
            slide_path = Path(item["frame"])
            zf.writestr(
                f"slide_{idx + 1:03d}_t{item['timestamp']}s{slide_path.suffix}",
                slide_path.read_bytes(),
            )
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# UI — Source selection
# ══════════════════════════════════════════════════════════════════════════════
source = st.radio("Source", ["Upload new file", "Use existing resource"], index=0)

uploaded_file     = None
selected_existing = None
existing_files = sorted(
    p.name
    for p in DOWNLOADS_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
)

if source == "Upload new file":
    uploaded_file = st.file_uploader(
        "Upload a presentation video (up to 2 GB)",
        type=["mp4", "mov", "mkv", "avi"],
    )
    if uploaded_file:
        size_mb = uploaded_file.size / 1_048_576
        st.caption(f"File size: **{size_mb:.1f} MB** — {'chunked write enabled ✅' if uploaded_file.size > LARGE_FILE_THRESHOLD else 'normal write'}")
else:
    if not existing_files:
        st.info(f"No video files found in `{DOWNLOADS_DIR}`. Upload one instead.")
    else:
        selected_existing = st.selectbox("Select a file from downloads", existing_files)
        if selected_existing:
            file_size_mb = (DOWNLOADS_DIR / selected_existing).stat().st_size / 1_048_576
            st.write(f"Selected: **{selected_existing}** ({file_size_mb:.1f} MB)")
            if file_size_mb < 500:          # st.video chokes on very large files
                st.video(str(DOWNLOADS_DIR / selected_existing))
            else:
                st.info("Preview disabled for files > 500 MB to avoid browser memory pressure.")

# ══════════════════════════════════════════════════════════════════════════════
# UI — Processing parameters
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("⚙️ Processing settings", expanded=True):
    threshold = st.slider(
        "Change threshold", 0.01, 0.30, 0.08, 0.01,
        help="Lower → more slides captured. Higher → stricter.",
    )
    frame_step = st.slider(
        "Analyze every Nth frame", 1, 30, 10,
        help="Higher = faster but may miss short transitions.",
    )
    min_gap_seconds = st.slider(
        "Minimum seconds between saved slides", 0.0, 5.0, 1.0, 0.5,
        help="Avoids duplicates during transitions.",
    )
    cols_per_row = st.slider("Slides per row", 1, 4, 2)

    # Extra controls for large video efficiency
    st.divider()
    st.markdown("**Large-video optimisations**")
    max_frames_to_scan = st.number_input(
        "Max frames to scan (0 = unlimited)",
        min_value=0, max_value=100_000, value=0, step=500,
        help="Limits scan to first N sampled frames. Useful to do a quick preview pass on a 1 GB+ file.",
    )
    delete_temp_after = st.checkbox(
        "Delete temp file after processing", value=True,
        help="Frees disk space immediately after slides are extracted.",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Resolve video path
# ══════════════════════════════════════════════════════════════════════════════
video_path_to_process = None
temp_video_path       = None          # track for cleanup

if source == "Upload new file" and uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    # Use a named temp file in the project downloads dir (survives the with-block)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=DOWNLOADS_DIR)
    os.close(tmp_fd)
    temp_video_path = Path(tmp_name)

    if uploaded_file.size > LARGE_FILE_THRESHOLD:
        st.info(f"Large file detected ({uploaded_file.size / 1_048_576:.0f} MB) — using chunked write.")
        write_uploaded_file_chunked(uploaded_file, temp_video_path)
    else:
        temp_video_path.write_bytes(uploaded_file.read())

    video_path_to_process = str(temp_video_path)

    size_mb = temp_video_path.stat().st_size / 1_048_576
    if size_mb < 500:
        st.video(video_path_to_process)

elif source == "Use existing resource" and selected_existing:
    video_path_to_process = str(DOWNLOADS_DIR / selected_existing)

# ══════════════════════════════════════════════════════════════════════════════
# Process
# ══════════════════════════════════════════════════════════════════════════════
if st.button("Process video", type="primary"):
    if not video_path_to_process:
        st.error("No video selected. Upload a file or choose an existing resource.")
    else:
        # Pass max_frames_to_scan only if > 0
        extra_kwargs = {}
        if max_frames_to_scan > 0:
            extra_kwargs["max_frames"] = int(max_frames_to_scan)

        with st.spinner("Detecting slide changes… (large files may take a few minutes)"):
            results = extract_unique_frames(
                video_path_to_process,
                threshold=threshold,
                frame_step=frame_step,
                min_time_between_slides=min_gap_seconds,
                **extra_kwargs,
            )

        # Cleanup temp file as early as possible to free disk space
        if delete_after := (delete_temp_after and temp_video_path and temp_video_path.exists()):
            try:
                temp_video_path.unlink()
            except Exception:
                pass

        if not results:
            st.warning("No slide changes detected. Try lowering the threshold.")
        else:
            st.success(f"Detected **{len(results)}** slide(s).")

            # --- Download All as ZIP (streamed build) ---
            zip_buf = stream_zip(results, Path(video_path_to_process).stem)
            st.download_button(
                label=f"⬇️ Download All {len(results)} Slides as ZIP",
                data=zip_buf,
                file_name=f"{Path(video_path_to_process).stem}_slides.zip",
                mime="application/zip",
            )

            st.divider()

            # --- Grid Display ---
            for row_start in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, item in enumerate(results[row_start: row_start + cols_per_row]):
                    slide_num  = row_start + col_idx + 1
                    slide_path = Path(item["frame"])
                    with cols[col_idx]:
                        st.image(
                            item["frame"],
                            caption=f"Slide {slide_num} | {item['timestamp']}s | Score: {item['score']:.3f}",
                            use_column_width=True,
                        )
                        st.download_button(
                            label=f"⬇️ Slide {slide_num}",
                            data=slide_path.read_bytes(),
                            file_name=f"slide_{slide_num:03d}_t{item['timestamp']}s{slide_path.suffix}",
                            mime="image/jpeg",
                            key=f"dl_{slide_num}",
                        )
