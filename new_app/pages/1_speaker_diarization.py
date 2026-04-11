import os
import io
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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

# ── session state defaults ────────────────────────────────────────────────────
defaults = {
    "diar_df":         None,
    "diar_rttm":       "",
    "diar_done":       False,
    "diar_error":      "",
    "diar_file_name":  "",
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

    st.markdown("**Model**")
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
    st.page_link("pages/0_hf_token_test.py", label="🔑 Test your HF token first", icon="🔑")

    # clear results button
    if st.button("🗑️ Clear results"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline…")
def load_pipeline(token: str, repo_id: str):
    """Load and cache the pyannote pipeline (pyannote.audio 4.x)."""
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


def run_diarization(pipeline, audio_path: str, num_spk: int, min_spk: int, max_spk: int):
    kwargs = {}
    if num_spk > 0:
        kwargs["num_speakers"] = num_spk
    else:
        if min_spk > 0:
            kwargs["min_speakers"] = min_spk
        if max_spk > 0:
            kwargs["max_speakers"] = max_spk

    # ── pre-process audio to fix sample count mismatch ────────────────────────
    try:
        import torchaudio
        import torch

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16000 Hz (pyannote's expected sample rate)
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
            waveform = resampler(waveform)
            sample_rate = target_sr

        # Pad to nearest chunk boundary (10s chunks = 160000 samples at 16kHz)
        chunk_samples = target_sr * 10  # 160000
        total_samples = waveform.shape[1]
        remainder = total_samples % chunk_samples
        if remainder != 0:
            pad_size = chunk_samples - remainder
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # Save the cleaned audio to a new temp file
        cleaned_path = audio_path + "_cleaned.wav"
        torchaudio.save(cleaned_path, waveform, sample_rate)
        audio_path = cleaned_path

    except Exception as e:
        # If pre-processing fails, fall back to original file
        st.warning(f"⚠️ Audio pre-processing skipped: {e}. Using original file.")
        cleaned_path = None

    # ── run pipeline ──────────────────────────────────────────────────────────
    try:
        diarization = pipeline(audio_path, **kwargs)
    finally:
        # Clean up the temporary cleaned file
        if "cleaned_path" in dir() and cleaned_path and os.path.exists(cleaned_path):
            try:
                os.remove(cleaned_path)
            except OSError:
                pass

    # ── compatibility: support both Annotation.itertracks and DiarizeOutput-like ─
    def _iter_diarization(diar):
        def _seg_obj(s):
            # return an object with .start and .end
            if hasattr(s, "start") and hasattr(s, "end"):
                return s
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                start, end = float(s[0]), float(s[1])
                return type("Seg", (), {"start": start, "end": end})()
            raise TypeError("Unsupported segment type")

        # pyannote.core.Annotation-like
        if hasattr(diar, "itertracks"):
            yield from diar.itertracks(yield_label=True)
            return

        # DiarizeOutput-like: common attributes
        segs = getattr(diar, "segments", None) or getattr(diar, "segments_", None)
        labs = getattr(diar, "labels", None) or getattr(diar, "labels_", None)
        if segs is not None and labs is not None:
            for s, l in zip(segs, labs):
                yield _seg_obj(s), None, l
            return

        # get_timeline() + labels()
        get_tl = getattr(diar, "get_timeline", None)
        if callable(get_tl):
            try:
                tl = get_tl()
                if hasattr(tl, "__iter__"):
                    # try paired labels first
                    labs = getattr(diar, "labels", None) or getattr(diar, "labels_", None)
                    if labs is not None:
                        for s, l in zip(tl, labs):
                            yield _seg_obj(s), None, l
                        return
                    # fallback: timeline may yield (segment, label) pairs
                    for item in tl:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            s, l = item[0], item[1]
                            yield _seg_obj(s), None, l
                    return
            except Exception:
                pass

        # iterable of (segment, label) or (segment, _, label)
        if isinstance(diar, (list, tuple)):
            for item in diar:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        s, l = item
                        yield _seg_obj(s), None, l
                    elif len(item) >= 3:
                        yield _seg_obj(item[0]), None, item[-1]
            return

        # dict-like wrappers
        if isinstance(diar, dict):
            for key in ("annotation", "diarization", "output", "result"):
                if key in diar:
                    yield from _iter_diarization(diar[key])
                    return

        # Inspect object's attributes / dataclass internals for lists/pairs
        try:
            attrs = getattr(diar, "__dict__", None)
            if isinstance(attrs, dict):
                # look for an attribute that is list/tuple of (segment, label) pairs
                for name, val in attrs.items():
                    if name.startswith("_"):
                        continue
                    if hasattr(val, "itertracks"):
                        yield from val.itertracks(yield_label=True)
                        return
                    if isinstance(val, (list, tuple)) and val:
                        first = val[0]
                        # list of (segment, label) pairs
                        if isinstance(first, (list, tuple)) and len(first) >= 2:
                            for item in val:
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    yield _seg_obj(item[0]), None, item[1]
                            return
                        # list of segments + separate labels attribute
                        for label_key in ("labels", "labels_", "speakers", "speaker_labels"):
                            lbls = attrs.get(label_key)
                            if isinstance(lbls, (list, tuple)) and len(lbls) == len(val):
                                for s, l in zip(val, lbls):
                                    yield _seg_obj(s), None, l
                                return
                        # list of segment-like objects only -> try to extract speaker field per item
                        if hasattr(first, "start") and hasattr(first, "end"):
                            for s in val:
                                label = getattr(s, "label", None) or getattr(s, "speaker", None) or "SPEAKER_0"
                                yield _seg_obj(s), None, label
                            return
        except Exception:
            pass

        # last attempt: attributes that look like pairs (informative failure)
        raise RuntimeError(
            f"Unsupported diarization output (type={type(diar)}). "
            "No 'itertracks', nor matching 'segments'/'labels', nor iterable pairs found."
        )

    rows = []
    for segment, _, label in _iter_diarization(diarization):
        rows.append({
            "start":    round(float(segment.start), 3),
            "end":      round(float(segment.end),   3),
            "duration": round(float(segment.end - segment.start), 3),
            "speaker":  label,
        })
    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

    # build RTTM in-memory
    rttm_lines = []
    for segment, _, label in _iter_diarization(diarization):
        rttm_lines.append(
            f"SPEAKER file 1 {segment.start:.3f} "
            f"{(segment.end - segment.start):.3f} "
            f"<NA> <NA> {label} <NA> <NA>"
        )
    rttm_str = "\n".join(rttm_lines) + "\n"

    return df, rttm_str


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
    "Upload audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    help="Mono or stereo audio. Long files (>30 min) may take several minutes.",
)

if uploaded:
    st.audio(uploaded)
    # clear old results when a new file is uploaded
    if uploaded.name != st.session_state.diar_file_name:
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.session_state.diar_file_name = uploaded.name

run_btn = st.button("▶ Run Diarization", type="primary", disabled=(uploaded is None))

if run_btn and uploaded:
    if not hf_token:
        st.error("❌ Please enter your Hugging Face token in the sidebar.")
        st.stop()

    # reset previous results before new run
    st.session_state.diar_done  = False
    st.session_state.diar_error = ""
    st.session_state.diar_df    = None
    st.session_state.diar_rttm  = ""

    tmp_path = None
    try:
        suffix = os.path.splitext(uploaded.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        # load pipeline
        with st.status("Loading pipeline…", expanded=True) as status:
            try:
                pipeline = load_pipeline(hf_token, model_choice)
                status.update(label=f"✅ Pipeline loaded: `{model_choice}`", state="complete")
            except RuntimeError as e:
                status.update(label="❌ Pipeline load failed", state="error")
                st.session_state.diar_error = str(e)
                st.stop()

        # run diarization
        with st.spinner("Running diarization… (this may take a while for long audio)"):
            df, rttm_str = run_diarization(
                pipeline, tmp_path,
                int(num_speakers), int(min_speakers), int(max_speakers),
            )

        # store in session state so results persist across reruns
        st.session_state.diar_df       = df
        st.session_state.diar_rttm     = rttm_str
        st.session_state.diar_done     = True
        st.session_state.diar_file_name = uploaded.name

    except Exception as e:
        st.session_state.diar_error = str(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# ── render results (persists across reruns) ───────────────────────────────────
if st.session_state.diar_error:
    st.error(f"❌ Diarization failed: {st.session_state.diar_error}")

if st.session_state.diar_done and st.session_state.diar_df is not None:
    df = st.session_state.diar_df

    st.success(
        f"✅ Done! Found **{df['speaker'].nunique()} speaker(s)**, **{len(df)} segments**."
    )

    tab1, tab2, tab3 = st.tabs(["📊 Timeline", "📋 Segments", "📈 Statistics"])

    with tab1:
        plot_diarization(df)

    with tab2:
        st.dataframe(
            df.style.format({"start": "{:.3f}", "end": "{:.3f}", "duration": "{:.3f}"}),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"{st.session_state.diar_file_name}.csv",
            mime="text/csv",
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

    st.download_button(
        "⬇ Download RTTM",
        data=st.session_state.diar_rttm.encode(),
        file_name=f"{st.session_state.diar_file_name}.rttm",
        mime="text/plain",
    )