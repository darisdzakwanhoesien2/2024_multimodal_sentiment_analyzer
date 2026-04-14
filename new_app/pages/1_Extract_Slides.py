from pathlib import Path
import tempfile
import zipfile
import io
import os
import json
import time

import cv2
import numpy as np
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
SLIDES_DIR    = PROJECT_ROOT / "extracted_slides"
RESUME_DIR    = PROJECT_ROOT / ".resume_checkpoints"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
SLIDES_DIR.mkdir(parents=True, exist_ok=True)
RESUME_DIR.mkdir(parents=True, exist_ok=True)

LARGE_FILE_THRESHOLD = 200 * 1024 * 1024   # 200 MB
CHUNK_SIZE           = 8  * 1024 * 1024    # 8 MB

try:
    st.config.set_option("server.maxUploadSize", 2048)
except Exception:
    pass

# ── .streamlit/config.toml ────────────────────────────────────────────────────
config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
if not config_path.exists():
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[server]\nmaxUploadSize = 2048\n\n[runner]\nfastReruns = true\n"
    )

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(video_stem: str) -> Path:
    return RESUME_DIR / f"{video_stem}.json"

def save_checkpoint(video_stem: str, last_timestamp: float, slides: list[dict]) -> None:
    data = {"last_timestamp": last_timestamp, "slides": slides}
    checkpoint_path(video_stem).write_text(json.dumps(data, indent=2))

def load_checkpoint(video_stem: str) -> dict | None:
    p = checkpoint_path(video_stem)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def delete_checkpoint(video_stem: str) -> None:
    p = checkpoint_path(video_stem)
    if p.exists():
        p.unlink()

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKED WRITE
# ══════════════════════════════════════════════════════════════════════════════

def write_uploaded_file_chunked(uploaded_file, dest_path: Path) -> int:
    total_size = uploaded_file.size
    uploaded_file.seek(0)
    progress = st.progress(0, text="⬆️ Writing file to disk…")
    bytes_written = 0
    with open(dest_path, "wb") as f:
        while True:
            chunk = uploaded_file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
            pct = min(int(bytes_written / total_size * 100), 100) if total_size else 0
            progress.progress(
                pct,
                text=f"⬆️ Uploading… {bytes_written/1_048_576:.1f} MB / {total_size/1_048_576:.1f} MB"
            )
    progress.empty()
    return bytes_written

# ══════════════════════════════════════════════════════════════════════════════
# CORE: STREAMING FRAME EXTRACTOR  (yields results one-by-one)
# ══════════════════════════════════════════════════════════════════════════════

def frame_diff_score(prev: np.ndarray, curr: np.ndarray) -> float:
    """Mean absolute difference between two grayscale frames, normalised 0-1."""
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(p - c)) / 255.0)


def extract_frames_streaming(
    video_path: str,
    threshold: float        = 0.08,
    frame_step: int         = 10,
    min_gap_seconds: float  = 1.0,
    max_frames: int         = 0,          # 0 = unlimited
    resume_from: float      = 0.0,        # seconds — skip before this timestamp
    output_dir: Path        = SLIDES_DIR,
):
    """
    Generator — yields dict(frame_path, timestamp, score, frame_index) one at a time.
    Supports resume_from timestamp and max_frames limit.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    # Jump to resume point
    if resume_from > 0:
        resume_frame = int(resume_from * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
        start_frame = resume_frame
    else:
        start_frame = 0

    prev_frame        = None
    last_saved_time   = resume_from - min_gap_seconds   # allow saving at first resumed frame
    frames_scanned    = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = frame_idx / fps

            # Skip frames for speed
            if (frame_idx - start_frame) % frame_step != 0:
                continue

            frames_scanned += 1
            if max_frames > 0 and frames_scanned > max_frames:
                break

            if prev_frame is None:
                prev_frame = frame
                continue

            score = frame_diff_score(prev_frame, frame)

            if score >= threshold and (timestamp - last_saved_time) >= min_gap_seconds:
                # Save slide immediately
                slide_name = f"slide_{timestamp:.2f}s_score{score:.3f}.jpg"
                slide_path = output_dir / slide_name
                cv2.imwrite(str(slide_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

                last_saved_time = timestamp
                prev_frame      = frame

                yield {
                    "frame":       str(slide_path),
                    "timestamp":   round(timestamp, 2),
                    "score":       round(score, 4),
                    "frame_index": frame_idx,
                    "total_frames": total_frames,
                    "duration_s":   round(duration_s, 1),
                    "fps":          fps,
                }
            else:
                prev_frame = frame
    finally:
        cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# ZIP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_zip(slides: list[dict], stem: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for idx, item in enumerate(slides):
            sp = Path(item["frame"])
            if sp.exists():
                zf.writestr(
                    f"slide_{idx+1:03d}_t{item['timestamp']}s{sp.suffix}",
                    sp.read_bytes(),
                )
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# PAGE UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎞️ Extract Slides from Video")
st.caption("Real-time extraction · Resume support · Up to 2 GB")

# ── Source ────────────────────────────────────────────────────────────────────
source = st.radio("Source", ["Upload new file", "Use existing resource"], horizontal=True)

uploaded_file     = None
selected_existing = None
existing_files    = sorted(
    p.name for p in DOWNLOADS_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
)

if source == "Upload new file":
    uploaded_file = st.file_uploader(
        "Upload a video (up to 2 GB)",
        type=["mp4", "mov", "mkv", "avi"],
    )
    if uploaded_file:
        size_mb = uploaded_file.size / 1_048_576
        st.caption(
            f"📦 {size_mb:.1f} MB — "
            f"{'chunked write ✅' if uploaded_file.size > LARGE_FILE_THRESHOLD else 'normal write'}"
        )
else:
    if not existing_files:
        st.info(f"No videos in `{DOWNLOADS_DIR}`. Upload one instead.")
    else:
        selected_existing = st.selectbox("Select existing video", existing_files)
        if selected_existing:
            mb = (DOWNLOADS_DIR / selected_existing).stat().st_size / 1_048_576
            st.write(f"**{selected_existing}** — {mb:.1f} MB")
            if mb < 500:
                st.video(str(DOWNLOADS_DIR / selected_existing))
            else:
                st.info("Preview disabled for files > 500 MB.")

# ── Settings ──────────────────────────────────────────────────────────────────
with st.expander("⚙️ Processing settings", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Change threshold", 0.01, 0.30, 0.08, 0.01,
                              help="Lower → more slides. Higher → stricter.")
        frame_step = st.slider("Analyze every Nth frame", 1, 30, 10,
                               help="Higher = faster, may miss quick transitions.")
        min_gap_seconds = st.slider("Min seconds between slides", 0.0, 5.0, 1.0, 0.5)
    with col2:
        cols_per_row = st.slider("Slides per row (grid)", 1, 4, 3)
        max_frames_to_scan = st.number_input(
            "Max frames to scan (0 = all)", min_value=0, max_value=500_000, value=0, step=1000,
            help="Cap scan for a quick preview pass on 1 GB+ files."
        )
        delete_temp_after = st.checkbox("Delete temp file after processing", value=True)
        save_checkpoint_enabled = st.checkbox("Enable resume checkpoints", value=True)

# ══════════════════════════════════════════════════════════════════════════════
# Resolve video path + write to disk if needed
# ══════════════════════════════════════════════════════════════════════════════
video_path_to_process = None
temp_video_path       = None
video_stem            = None

if source == "Upload new file" and uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    video_stem = Path(uploaded_file.name).stem
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=DOWNLOADS_DIR)
    os.close(tmp_fd)
    temp_video_path = Path(tmp_name)

    if uploaded_file.size > LARGE_FILE_THRESHOLD:
        st.info(f"Large file ({uploaded_file.size/1_048_576:.0f} MB) — chunked write in progress…")
        write_uploaded_file_chunked(uploaded_file, temp_video_path)
    else:
        temp_video_path.write_bytes(uploaded_file.read())

    video_path_to_process = str(temp_video_path)

elif source == "Use existing resource" and selected_existing:
    video_path_to_process = str(DOWNLOADS_DIR / selected_existing)
    video_stem = Path(selected_existing).stem

# ── Resume checkpoint detection ───────────────────────────────────────────────
resume_from   = 0.0
prior_slides  = []
checkpoint    = load_checkpoint(video_stem) if video_stem else None

if checkpoint:
    st.info(
        f"⏩ Checkpoint found — last processed timestamp: **{checkpoint['last_timestamp']}s** "
        f"with **{len(checkpoint['slides'])}** slides saved.",
        icon="💾",
    )
    col_resume, col_fresh = st.columns(2)
    with col_resume:
        if st.button("▶️ Resume from checkpoint", use_container_width=True):
            resume_from  = checkpoint["last_timestamp"]
            prior_slides = checkpoint["slides"]
            st.session_state["resume_from"]  = resume_from
            st.session_state["prior_slides"] = prior_slides
    with col_fresh:
        if st.button("🔄 Start fresh (discard checkpoint)", use_container_width=True):
            delete_checkpoint(video_stem)
            st.session_state.pop("resume_from",  None)
            st.session_state.pop("prior_slides", None)
            st.rerun()

    # Restore from session state across reruns
    resume_from  = st.session_state.get("resume_from",  resume_from)
    prior_slides = st.session_state.get("prior_slides", prior_slides)

# ══════════════════════════════════════════════════════════════════════════════
# PROCESS BUTTON — Streaming extraction with live updates
# ══════════════════════════════════════════════════════════════════════════════
if st.button("🚀 Process video", type="primary", disabled=video_path_to_process is None):

    slides     = list(prior_slides)   # carry over resumed slides
    video_stem = video_stem or Path(video_path_to_process).stem

    # ── Live UI placeholders ──────────────────────────────────────────────────
    status_text   = st.empty()
    progress_bar  = st.progress(0, text="Starting…")
    slide_counter = st.empty()
    st.divider()
    grid_container = st.container()   # slides appear here in real time

    # Pre-fill grid with any resumed slides
    if prior_slides:
        status_text.info(f"Resuming from {resume_from}s — {len(prior_slides)} slides already saved.")
        with grid_container:
            cols = st.columns(cols_per_row)
            for idx, item in enumerate(prior_slides):
                sp = Path(item["frame"])
                with cols[idx % cols_per_row]:
                    if sp.exists():
                        st.image(str(sp), caption=f"Slide {idx+1} | {item['timestamp']}s", use_column_width=True)

    # ── Stream extraction ─────────────────────────────────────────────────────
    start_wall = time.time()
    try:
        gen = extract_frames_streaming(
            video_path     = video_path_to_process,
            threshold      = threshold,
            frame_step     = frame_step,
            min_gap_seconds= min_gap_seconds,
            max_frames     = int(max_frames_to_scan),
            resume_from    = resume_from,
            output_dir     = SLIDES_DIR / video_stem,
        )

        for result in gen:
            slides.append(result)
            idx = len(slides) - 1

            # ── Update progress bar ───────────────────────────────────────
            total_frames = result["total_frames"]
            frame_idx    = result["frame_index"]
            pct = min(int(frame_idx / total_frames * 100), 99) if total_frames else 0
            elapsed      = time.time() - start_wall
            progress_bar.progress(
                pct,
                text=(
                    f"🎞️ {pct}% · frame {frame_idx}/{total_frames} · "
                    f"t={result['timestamp']}s · {len(slides)} slides · "
                    f"⏱ {elapsed:.0f}s elapsed"
                )
            )
            slide_counter.markdown(f"**{len(slides)} slide(s) detected so far…**")

            # ── Render new slide immediately in grid ──────────────────────
            with grid_container:
                cols = st.columns(cols_per_row)
                sp   = Path(result["frame"])
                with cols[idx % cols_per_row]:
                    st.image(
                        str(sp),
                        caption=f"Slide {idx+1} | {result['timestamp']}s | Δ{result['score']:.3f}",
                        use_column_width=True,
                    )
                    st.download_button(
                        label    = f"⬇️ Slide {idx+1}",
                        data     = sp.read_bytes(),
                        file_name= sp.name,
                        mime     = "image/jpeg",
                        key      = f"dl_live_{idx}",
                    )

            # ── Save checkpoint every 10 new slides ───────────────────────
            if save_checkpoint_enabled and len(slides) % 10 == 0:
                save_checkpoint(video_stem, result["timestamp"], slides)

    except Exception as e:
        st.error(f"Extraction error: {e}")
        if save_checkpoint_enabled and slides:
            save_checkpoint(video_stem, slides[-1]["timestamp"], slides)
            st.warning("Checkpoint saved — you can resume later.")
        st.stop()

    # ── Finalise ──────────────────────────────────────────────────────────────
    progress_bar.progress(100, text="✅ Done!")
    elapsed_total = time.time() - start_wall

    if not slides:
        st.warning("No slide changes detected. Try lowering the threshold.")
    else:
        st.success(
            f"✅ Extracted **{len(slides)}** slide(s) in {elapsed_total:.1f}s "
            f"(resumed from {resume_from}s)" if resume_from else
            f"✅ Extracted **{len(slides)}** slide(s) in {elapsed_total:.1f}s"
        )

        # Save final checkpoint (mark as complete)
        if save_checkpoint_enabled:
            save_checkpoint(video_stem, slides[-1]["timestamp"], slides)

        # ── Download ZIP ──────────────────────────────────────────────────
        zip_buf = build_zip(slides, video_stem)
        st.download_button(
            label     = f"⬇️ Download all {len(slides)} slides as ZIP",
            data      = zip_buf,
            file_name = f"{video_stem}_slides.zip",
            mime      = "application/zip",
        )

    # ── Cleanup temp file ─────────────────────────────────────────────────
    if delete_temp_after and temp_video_path and temp_video_path.exists():
        try:
            temp_video_path.unlink()
        except Exception:
            pass

    # Clear resume session state after successful full run
    st.session_state.pop("resume_from",  None)
    st.session_state.pop("prior_slides", None)
