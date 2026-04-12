from pathlib import Path
import os
import io
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import shutil
import subprocess

# ── optional import ───────────────────────────────────────────────────────────
Pipeline = None
_import_error = None
try:
    from pyannote.audio import Pipeline
except Exception as e:
    _import_error = e

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Speaker Diarization", page_icon="🎙️", layout="wide")
st.title("🎙️ Speaker Diarization")
st.markdown("Upload an audio file and provide your Hugging Face token to identify who spoke when.")

if Pipeline is None:
    st.error(
        "❌ `pyannote.audio` is not installed in this Python environment.\n\n"
        "Run in your terminal:\n```\npip install pyannote.audio\n```\n\n"
        + (f"Import error: `{_import_error}`" if _import_error else "")
    )
    st.stop()

# --- Added: allow selecting existing files from project downloads ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_VIDEO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".avi"
}
existing_files = sorted(
    [p.name for p in DOWNLOADS_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_VIDEO_EXTS]
)

# ── session state defaults ────────────────────────────────────────────────────
defaults = {
    "diar_df":         None,
    "diar_rttm":       "",
    "diar_done":       False,
    "diar_error":      "",
    "diar_file_name":  "",
    "diar_transcript_text":     None,
    "diar_transcript_segments": None,
    "diar_segment_texts":       None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        help="Get your token at https://huggingface.co/settings/tokens (read scope required)",
        value=os.getenv("HF_TOKEN", ""),
    )
    if not hf_token:
        st.warning("⚠️ Token required. Go to **hf_token_test** to verify your token first.")

    st.markdown("**Diarization Model**")
    model_choice = st.selectbox(
        "Pipeline",
        options=[
            "pyannote/speaker-diarization-3.0",
            "pyannote/speaker-diarization-3.1",
        ],
        index=0,
    )

    st.markdown("**Optional constraints**")
    num_speakers = st.number_input("Number of speakers (0 = auto)", min_value=0, max_value=20, value=0)
    min_speakers = st.number_input("Min speakers (0 = auto)",        min_value=0, max_value=20, value=0)
    max_speakers = st.number_input("Max speakers (0 = auto)",        min_value=0, max_value=20, value=0)

    st.markdown("---")

    # ── Transcription options ─────────────────────────────────────────────────
    st.markdown("**Transcription (faster-whisper)**")
    do_transcribe = st.checkbox(
        "Transcribe audio", value=False,
        help="Run faster-whisper transcription alongside diarization."
    )

    whisper_model_size = st.selectbox(
        "Whisper model size",
        options=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=5,   # default to large-v3
        help="large-v3 = best accuracy. tiny/base = fastest.",
    )

    # Local model path (from hf download or git clone)
    local_model_path = st.text_input(
        "Local model path (optional)",
        value="",
        placeholder="e.g. /path/to/faster-whisper-large-v3",
        help=(
            "If you cloned/downloaded the model locally with:\n"
            "`hf download Systran/faster-whisper-large-v3`\n"
            "or `git clone https://huggingface.co/Systran/faster-whisper-large-v3`\n"
            "paste the folder path here. Leave blank to auto-download."
        ),
    )

    whisper_device = st.selectbox(
        "Compute device",
        options=["auto", "cpu", "cuda"],
        index=0,
        help="'auto' picks CUDA if available, else CPU.",
    )

    whisper_compute_type = st.selectbox(
        "Compute type",
        options=["default", "int8", "int8_float16", "float16", "float32"],
        index=1,   # int8 = fast on CPU
        help="int8 = fastest on CPU. float16 = best GPU speed. default = library decides.",
    )

    whisper_language = st.text_input(
        "Language code (optional)",
        value="",
        placeholder="e.g. en, ms, zh",
        help="Leave blank for auto-detect.",
    )

    st.page_link("pages/0_hf_token_test.py", label="🔑 Test your HF token first", icon="🔑")

    if st.button("🗑️ Clear results"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline…")
def load_pipeline(token: str, repo_id: str):
    if not token:
        raise RuntimeError("A Hugging Face token is required.")
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
    except Exception as e:
        raise RuntimeError(f"Hugging Face login failed: {e}") from e
    try:
        return Pipeline.from_pretrained(repo_id, token=token)
    except Exception as e:
        err = str(e)
        if "403" in err:
            raise RuntimeError(
                f"Access denied (403) for `{repo_id}`.\n"
                "Visit the model page on HF and click **Agree and access repository**."
            ) from e
        if "404" in err:
            raise RuntimeError(
                f"Model `{repo_id}` not found (404). Try the other model in the sidebar."
            ) from e
        raise RuntimeError(f"Failed to load `{repo_id}`:\n{e}") from e


@st.cache_resource(show_spinner="Loading faster-whisper model…")
def load_whisper_model(model_size_or_path: str, device: str, compute_type: str):
    """Load and cache a faster-whisper WhisperModel."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "Run: pip install faster-whisper"
        )
    ct = compute_type if compute_type != "default" else None
    kwargs = dict(device=device)
    if ct:
        kwargs["compute_type"] = ct
    return WhisperModel(model_size_or_path, **kwargs)


def run_diarization(pipeline, audio_path: str, num_spk: int, min_spk: int, max_spk: int):
    kwargs = {}
    if num_spk > 0:
        kwargs["num_speakers"] = num_spk
    else:
        if min_spk > 0:
            kwargs["min_speakers"] = min_spk
        if max_spk > 0:
            kwargs["max_speakers"] = max_spk

    # ── pre-process audio ─────────────────────────────────────────────────────
    cleaned_path = None
    try:
        import torchaudio
        import torch

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr

        chunk_samples = target_sr * 10
        remainder = waveform.shape[1] % chunk_samples
        if remainder != 0:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_samples - remainder))

        cleaned_path = audio_path + "_cleaned.wav"
        torchaudio.save(cleaned_path, waveform, sample_rate)
        audio_path = cleaned_path

    except Exception as e:
        st.warning(f"⚠️ Audio pre-processing skipped: {e}. Using original file.")

    # ── run pipeline ──────────────────────────────────────────────────────────
    try:
        diarization = pipeline(audio_path, **kwargs)
    finally:
        if cleaned_path and os.path.exists(cleaned_path):
            try:
                os.remove(cleaned_path)
            except OSError:
                pass

    # ── iterate diarization output ────────────────────────────────────────────
    def _iter_diarization(diar):
        def _seg_obj(s):
            if hasattr(s, "start") and hasattr(s, "end"):
                return s
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                return type("Seg", (), {"start": float(s[0]), "end": float(s[1])})()
            raise TypeError("Unsupported segment type")

        if hasattr(diar, "itertracks"):
            yield from diar.itertracks(yield_label=True)
            return

        segs = getattr(diar, "segments", None) or getattr(diar, "segments_", None)
        labs = getattr(diar, "labels", None) or getattr(diar, "labels_", None)
        if segs is not None and labs is not None:
            for s, l in zip(segs, labs):
                yield _seg_obj(s), None, l
            return

        if isinstance(diar, dict):
            for key in ("annotation", "diarization", "output", "result"):
                if key in diar:
                    yield from _iter_diarization(diar[key])
                    return

        attrs = getattr(diar, "__dict__", {})
        for name, val in attrs.items():
            if name.startswith("_"):
                continue
            if hasattr(val, "itertracks"):
                yield from val.itertracks(yield_label=True)
                return
            if isinstance(val, (list, tuple)) and val:
                first = val[0]
                if hasattr(first, "start") and hasattr(first, "end"):
                    for s in val:
                        label = getattr(s, "label", None) or getattr(s, "speaker", None) or "SPEAKER_0"
                        yield _seg_obj(s), None, label
                    return

        raise RuntimeError(
            f"Unsupported diarization output (type={type(diar)}).\n"
            f"Attributes: {list(getattr(diar, '__dict__', {}).keys())}"
        )

    rows = []
    rttm_lines = []
    for segment, _, label in _iter_diarization(diarization):
        start = round(float(segment.start), 3)
        end   = round(float(segment.end),   3)
        dur   = round(end - start,           3)
        rows.append({"start": start, "end": end, "duration": dur, "speaker": label})
        rttm_lines.append(
            f"SPEAKER file 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>"
        )

    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    rttm_str = "\n".join(rttm_lines) + "\n"
    return df, rttm_str


def run_transcription(
    audio_path: str,
    model_size_or_path: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str | None = None,
):
    """
    Transcribe audio with faster-whisper.
    Returns (full_text, segments) where segments is list of dicts: {start, end, text}.
    """
    whisper_model = load_whisper_model(model_size_or_path, device, compute_type)

    kwargs = dict(beam_size=5, vad_filter=True)
    if language:
        kwargs["language"] = language

    segments_gen, info = whisper_model.transcribe(audio_path, **kwargs)

    out_segments = []
    full_text_parts = []
    for seg in segments_gen:
        out_segments.append({
            "start": round(float(seg.start), 3),
            "end":   round(float(seg.end),   3),
            "text":  seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    full_text = " ".join(full_text_parts)
    return full_text, out_segments, info


def align_transcript_with_speakers(diar_df: pd.DataFrame, whisper_segments: list):
    """Align whisper segments to diarization speaker segments by overlap."""
    if diar_df is None or diar_df.empty:
        return pd.DataFrame(columns=["start", "end", "duration", "speaker", "text"])

    def overlap(s1, e1, s2, e2):
        return max(0.0, min(e1, e2) - max(s1, s2))

    rows = []
    for _, row in diar_df.iterrows():
        s0, e0 = float(row["start"]), float(row["end"])
        pieces = [w["text"] for w in whisper_segments if overlap(s0, e0, w["start"], w["end"]) > 0]
        rows.append({
            "start":    row["start"],
            "end":      row["end"],
            "duration": row["duration"],
            "speaker":  row["speaker"],
            "text":     " ".join(pieces).strip(),
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def plot_diarization(df: pd.DataFrame):
    if df.empty:
        st.info("No speech segments detected.")
        return
    speakers = sorted(df["speaker"].unique())
    n        = len(speakers)
    y_map    = {s: i for i, s in enumerate(speakers)}
    colours  = cm.tab10(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(figsize=(12, max(2, n * 0.8)))
    for _, row in df.iterrows():
        spk = row["speaker"]
        ax.broken_barh(
            [(row["start"], row["duration"])],
            (y_map[spk] - 0.4, 0.8),
            facecolors=colours[y_map[spk]],
            alpha=0.85,
        )
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()), fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title("Diarization Timeline", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def speaker_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("speaker")["duration"]
        .agg(segments="count", total_time="sum", avg_segment="mean")
        .round(2)
        .reset_index()
    )
    total = df["duration"].sum()
    stats["percentage"] = (stats["total_time"] / total * 100).round(1)
    return stats.sort_values("total_time", ascending=False).reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload audio / video file",
    type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mkv", "mov", "avi"],
    help="Mono or stereo. Long files (>30 min) may take several minutes.",
)

# --- Added: option to pick an existing file from the project's downloads folder ---
selected_existing = None
if existing_files:
    selected_existing = st.selectbox("Or choose an existing file from downloads", [""] + existing_files)
    if selected_existing:
        sel_path = DOWNLOADS_DIR / selected_existing
        st.write(f"Selected: {selected_existing}")
        # preview audio or video depending on extension
        if sel_path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}:
            st.video(str(sel_path))
        else:
            st.audio(str(sel_path))

if uploaded:
    st.audio(uploaded)
    if uploaded.name != st.session_state.diar_file_name:
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.session_state.diar_file_name = uploaded.name

# allow running when either an upload exists or an existing file is chosen
run_btn = st.button("▶ Run Diarization", type="primary", disabled=(uploaded is None and not selected_existing))

if run_btn and (uploaded or selected_existing):
    if not hf_token:
        st.error("❌ Please enter your Hugging Face token in the sidebar.")
        st.stop()

    st.session_state.diar_done  = False
    st.session_state.diar_error = ""
    st.session_state.diar_df    = None
    st.session_state.diar_rttm  = ""
    st.session_state.diar_transcript_text     = None
    st.session_state.diar_transcript_segments = None
    st.session_state.diar_segment_texts       = None

    orig_tmp_path  = None
    tmp_audio_path = None

    try:
        # If user chose an existing file, copy it to a temp file for processing.
        if selected_existing and not uploaded:
            selected_path = DOWNLOADS_DIR / selected_existing
            suffix = selected_path.suffix.lower() or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(selected_path.read_bytes())
                orig_tmp_path = tmp.name
            # preserve filename for downloads/outputs
            st.session_state.diar_file_name = selected_existing
        else:
            suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                orig_tmp_path = tmp.name
            st.session_state.diar_file_name = uploaded.name

        tmp_path = orig_tmp_path

        # ── extract audio from video ──────────────────────────────────────────
        video_exts = {".mp4", ".mkv", ".mov", ".avi"}
        if suffix in video_exts:
            if shutil.which("ffmpeg") is None:
                st.warning("⚠️ ffmpeg not found. Install: `brew install ffmpeg`")
            else:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ta:
                        tmp_audio_path = ta.name
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", orig_tmp_path,
                         "-vn", "-ac", "1", "-ar", "16000", tmp_audio_path],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    tmp_path = tmp_audio_path
                except subprocess.CalledProcessError as e:
                    st.warning(f"⚠️ ffmpeg failed: {e}. Using original file.")
                    tmp_audio_path = None

        # ── load diarization pipeline ─────────────────────────────────────────
        with st.status("Loading diarization pipeline…", expanded=True) as status:
            try:
                pipeline = load_pipeline(hf_token, model_choice)
                status.update(label=f"✅ Pipeline loaded: `{model_choice}`", state="complete")
            except RuntimeError as e:
                status.update(label="❌ Pipeline load failed", state="error")
                st.session_state.diar_error = str(e)
                st.stop()

        # ── diarization ───────────────────────────────────────────────────────
        with st.spinner("Running diarization…"):
            df, rttm_str = run_diarization(
                pipeline, tmp_path,
                int(num_speakers), int(min_speakers), int(max_speakers),
            )

        # ── transcription (faster-whisper) ────────────────────────────────────
        if do_transcribe:
            try:
                # resolve model: prefer local path if provided
                model_ref = local_model_path.strip() if local_model_path.strip() else whisper_model_size
                device_   = whisper_device if whisper_device != "auto" else (
                    "cuda" if __import__("torch").cuda.is_available() else "cpu"
                )
                lang = whisper_language.strip() or None

                with st.spinner(f"Transcribing with faster-whisper `{model_ref}` on `{device_}`…"):
                    full_text, w_segments, info = run_transcription(
                        tmp_path,
                        model_size_or_path=model_ref,
                        device=device_,
                        compute_type=whisper_compute_type,
                        language=lang,
                    )
                    aligned = align_transcript_with_speakers(df, w_segments)

                st.session_state.diar_transcript_text     = full_text
                st.session_state.diar_transcript_segments = w_segments
                st.session_state.diar_segment_texts       = aligned
                st.caption(
                    f"🌐 Detected language: **{info.language}** "
                    f"(probability {info.language_probability:.0%})"
                )
            except Exception as e:
                st.warning(f"⚠️ Transcription skipped: {e}")

        # ── persist results ───────────────────────────────────────────────────
        st.session_state.diar_df        = df
        st.session_state.diar_rttm      = rttm_str
        st.session_state.diar_done      = True
        st.session_state.diar_file_name = uploaded.name

    except Exception as e:
        st.session_state.diar_error = str(e)
    finally:
        for p in [orig_tmp_path, tmp_audio_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

# ── render results ────────────────────────────────────────────────────────────
if st.session_state.diar_error:
    st.error(f"❌ Diarization failed: {st.session_state.diar_error}")

if st.session_state.diar_done and st.session_state.diar_df is not None:
    df = st.session_state.diar_df
    st.success(f"✅ Done! Found **{df['speaker'].nunique()} speaker(s)**, **{len(df)} segments**.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Timeline", "📋 Segments", "📈 Statistics", "📝 Transcript"])

    with tab1:
        plot_diarization(df)

    with tab2:
        st.dataframe(
            df.style.format({"start": "{:.3f}", "end": "{:.3f}", "duration": "{:.3f}"}),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download CSV", data=df.to_csv(index=False).encode(),
            file_name=f"{st.session_state.diar_file_name}.csv", mime="text/csv",
        )

    with tab3:
        stats = speaker_stats(df)
        st.dataframe(stats, use_container_width=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(stats["speaker"], stats["total_time"],
                color=cm.tab10(np.linspace(0, 1, len(stats))))
        ax2.set_xlabel("Speaker")
        ax2.set_ylabel("Total speaking time (s)")
        ax2.set_title("Speaking time per speaker")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with tab4:
        tt        = st.session_state.get("diar_transcript_text")
        segs      = st.session_state.get("diar_transcript_segments")
        seg_texts = st.session_state.get("diar_segment_texts")

        if not tt:
            st.info("Enable **Transcribe audio** in the sidebar and re-run to see the transcript.")
        else:
            st.subheader("Full transcript")
            st.write(tt)
            st.download_button(
                "⬇ Download full transcript (.txt)", data=tt.encode(),
                file_name=f"{st.session_state.diar_file_name}.txt", mime="text/plain",
            )

            if segs:
                with st.expander("📄 Raw Whisper segments"):
                    st.dataframe(
                        pd.DataFrame(segs).style.format({"start": "{:.3f}", "end": "{:.3f}"}),
                        use_container_width=True,
                    )

            if seg_texts is not None and not seg_texts.empty:
                st.subheader("Speaker-aligned transcript")
                st.dataframe(
                    seg_texts.style.format(
                        {"start": "{:.3f}", "end": "{:.3f}", "duration": "{:.3f}"}
                    ),
                    use_container_width=True,
                )
                st.download_button(
                    "⬇ Download speaker-aligned transcript (.csv)",
                    data=seg_texts.to_csv(index=False).encode(),
                    file_name=f"{st.session_state.diar_file_name}.diar_transcript.csv",
                    mime="text/csv",
                )

    st.download_button(
        "⬇ Download RTTM", data=st.session_state.diar_rttm.encode(),
        file_name=f"{st.session_state.diar_file_name}.rttm", mime="text/plain",
    )