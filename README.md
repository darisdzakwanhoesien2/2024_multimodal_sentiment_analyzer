Test code 3
Test code 2
Test code
Test

Code: https://chatgpt.com/c/694233c7-d5d4-8332-b9a1-aa76429ec0e7
Original Past Codes: https://github.com/darisdzakwanhoesien2/2025Code/blob/main/developing/oulu_courses/affcomp_old/python.py

ui/utils.py is the drive

Perfect — below is a **clean, complete, scalable solution** with:

1. ✅ **Final project directory structure (authoritative)**
2. ✅ **Complete working `app.py`**
3. ✅ **Centralized dataset discovery (`ui/utils.py`)**
4. ✅ **Clear scaling rules (why this won’t break later)**

This is **production-grade**, **research-grade**, and **future-proof**.

---

# ✅ 1. FINAL PROJECT DIRECTORY STRUCTURE

This is the structure you should now treat as **canonical**.

```
multimodal/
├── app.py                          # Main Streamlit entry point
├── README.md
│
├── data/                           # 📦 ALL DATASETS LIVE HERE
│   ├── data_2/
│   │   ├── downloaded_video.webm
│   │   ├── final_audio.mp3
│   │   ├── transcription_result.json
│   │   ├── transcription_result.csv
│   │   ├── transcription_result_new.json
│   │   ├── metrics_plot.png
│   │   ├── correlation_matrix.png
│   │   └── video_info.json
│   │
│   ├── data_3/
│   │   ├── *.mp4
│   │   ├── final_audio.mp3
│   │   ├── transcription_result.json
│   │   ├── transcription_result.csv
│   │   ├── metrics_plot.png
│   │   └── video_info.json
│   │
│   └── data_n/
│
├── core/                           # 🧠 Core logic (NO Streamlit)
│   ├── dataframe_utils.py
│   ├── lexicon_sentiment.py
│   └── plotting.py
│
├── pipeline/                       # ⚙️ Pipelines (generate artifacts)
│   └── run_lexicon_pipeline.py
│
├── ui/                             # 🖥 UI helpers
│   └── utils.py
│
├── pages/                          # 📑 Multi-page Streamlit app
│   ├── 01_Overview.py
│   ├── 02_Transcript.py
│   ├── 03_Sentiment.py
│   ├── 04_Explainability.py
│   ├── 05_Batch_Comparison.py
│   └── 06_Video_Sentiment_Player.py
│
└── project_directory.md
```

---

# ✅ 2. COMPLETE `ui/utils.py` (CENTRALIZED & SAFE)

This file is the **single source of truth** for dataset discovery.

📄 `ui/utils.py`

```python

```

---

# ✅ 3. COMPLETE `app.py` (SCALED, CLEAN, WORKING)

📄 `app.py`

```python

```

---

# ✅ 4. WHY THIS SCALES (IMPORTANT)

This design ensures:

| Concern                  | Solved                              |
| ------------------------ | ----------------------------------- |
| Many datasets            | `data/data_*` pattern               |
| No hardcoded paths       | Centralized `ui/utils.py`           |
| Clean separation         | Code ≠ Data                         |
| Batch analytics          | Easy to loop over `list_datasets()` |
| Docker / cloud           | Mount `data/` as volume             |
| Research reproducibility | Immutable dataset folders           |

---

# 🧠 Mental model (keep this)

> **Each `data_x/` folder is a self-contained experiment artifact**

Your app simply **reads**, never mutates.

---

## 🚀 If you want next

I can help you add:

* 📊 Cross-dataset comparison page
* 🧠 Dataset registry (index CSV/JSON)
* 🧪 Validation checks per dataset
* ☁️ Cloud-ready layout
* 📦 Zipped experiment export

Just tell me the next step.


Great, this is a **classic AffComp / multimodal pipeline** that just needs to be **re-wrapped into a clean Streamlit interface** instead of being a monolithic script.

Below I’ll do **3 things clearly and safely**:

1. **Explain how to reinterpret your existing code** (no logic lost)
2. **Propose a clean Streamlit app architecture** that works with your existing `data_2/`, `data_3/`
3. **Give you a complete `app.py`** that:

   * Browses processed videos
   * Loads `video_info.json`, `transcription_result.json`
   * Visualizes transcript + sentiment metrics
   * Displays saved plots (no re-running OpenAI by default ⚠️)

This avoids re-triggering Whisper / GPT unless you explicitly want it.

---

## 1️⃣ What we are NOT changing

We **do not touch**:

* Your yt-dlp logic
* Whisper transcription
* Lexicon + VADER sentiment
* OpenAI iterative sentiment
* CSV / JSON formats

We treat them as **offline preprocessing**.

Streamlit becomes a **viewer + controller**, not the processor.

---

## 2️⃣ Recommended project structure (minimal changes)

```text
multimodal/
│
├── app.py                  ← NEW Streamlit app
├── structure_code.py       ← keep (offline processing)
│
├── data_2/
│   ├── downloaded_video.webm
│   ├── final_audio.mp3
│   ├── transcription_result.json
│   ├── transcription_result_new.json
│   ├── transcription_result.csv
│   ├── metrics_plot.png
│   ├── correlation_matrix.png
│   └── video_info.json
│
├── data_3/
│   └── (same structure)
```

---

## 3️⃣ Streamlit app design (what you’ll see)

### Sidebar

* Select **dataset folder** (`data_2`, `data_3`)
* Toggle:

  * Show transcript
  * Show sentiment tables
  * Show plots

### Main panel

* 📄 Video metadata
* 📝 Transcript (scrollable)
* 📊 Segment-level sentiment
* 📈 Time-series sentiment plot
* 🔥 Correlation heatmap

---

## 4️⃣ Complete `app.py` (drop-in ready)

> ✅ This **does not call OpenAI**
> ✅ Safe for demos, grading, and review
> ⚠️ Uses already-generated files only

```python
import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(
    page_title="Multimodal Sentiment Explorer",
    layout="wide"
)

st.title("🎥 Multimodal Affect & Sentiment Explorer")
st.markdown("""
This app visualizes **video metadata, transcription, and sentiment analysis**
generated from your multimodal pipeline (audio + text).
""")

ROOT_DIR = Path(__file__).parent
DATA_DIRS = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.startswith("data_")])

# =====================================================
# Sidebar: dataset selection
# =====================================================
st.sidebar.header("📁 Dataset Selection")

if not DATA_DIRS:
    st.error("No data folders (data_*) found.")
    st.stop()

dataset = st.sidebar.selectbox(
    "Select processed dataset",
    DATA_DIRS,
    format_func=lambda p: p.name
)

# Toggle controls
show_transcript = st.sidebar.checkbox("Show full transcript", True)
show_tables = st.sidebar.checkbox("Show sentiment tables", True)
show_plots = st.sidebar.checkbox("Show plots", True)

# =====================================================
# Load helper
# =====================================================
def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# =====================================================
# Load files
# =====================================================
video_info = load_json(dataset / "video_info.json")
transcription = load_json(dataset / "transcription_result.json")
sentiment_json = load_json(dataset / "transcription_result_new.json")

csv_path = dataset / "transcription_result.csv"
metrics_plot_path = dataset / "metrics_plot.png"
corr_plot_path = dataset / "correlation_matrix.png"

df = pd.read_csv(csv_path) if csv_path.exists() else None

# =====================================================
# Video metadata
# =====================================================
st.subheader("📄 Video Metadata")

if video_info:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Title:** {video_info.get('title')}")
        st.markdown(f"**Uploader:** {video_info.get('uploader')}")
        st.markdown(f"**Upload Date:** {video_info.get('upload_date')}")

    with col2:
        st.markdown(f"**Duration:** {video_info.get('duration')} seconds")
        st.markdown("**Description:**")
        st.write(video_info.get("description", "")[:500] + "…")
else:
    st.warning("video_info.json not found.")

st.divider()

# =====================================================
# Transcript
# =====================================================
if show_transcript:
    st.subheader("📝 Transcription")

    if transcription and "segments" in transcription:
        transcript_text = transcription.get("transcription_text", "")
        st.text_area(
            "Full Transcript",
            transcript_text,
            height=250
        )

        with st.expander("🔍 Segment-level transcript"):
            for seg in transcription["segments"]:
                st.markdown(
                    f"**[{seg['start']:.2f}s – {seg['end']:.2f}s]** {seg['text']}"
                )
    else:
        st.warning("Transcription file missing or invalid.")

st.divider()

# =====================================================
# Sentiment tables
# =====================================================
if show_tables and df is not None:
    st.subheader("📊 Segment-level Sentiment Metrics")

    cols_to_show = [
        "start", "end",
        "nltk_opinion_lexicon_net",
        "vader_neg", "vader_neu", "vader_pos", "vader_compound",
        "openai_mean_sentiment", "std_sentiment"
    ]

    available_cols = [c for c in cols_to_show if c in df.columns]

    st.dataframe(
        df[available_cols],
        use_container_width=True,
        height=400
    )

st.divider()

# =====================================================
# Plots
# =====================================================
if show_plots:
    st.subheader("📈 Visual Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment Metrics Over Time**")
        if metrics_plot_path.exists():
            st.image(Image.open(metrics_plot_path), use_container_width=True)
        else:
            st.info("metrics_plot.png not found.")

    with col2:
        st.markdown("**Correlation Matrix**")
        if corr_plot_path.exists():
            st.image(Image.open(corr_plot_path), use_container_width=True)
        else:
            st.info("correlation_matrix.png not found.")

# =====================================================
# Summary
# =====================================================
if sentiment_json:
    st.subheader("🧠 Overall Sentiment Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Sentiment", round(sentiment_json["overall_mean_sentiment"], 3))
    col2.metric("Avg Std Dev", round(sentiment_json["overall_standard_deviation"], 3))
    col3.metric("Combined Std Dev", round(sentiment_json["combined_standard_deviation"], 3))
```

---

## 5️⃣ What we can do next (recommended)

Given your background in **multimodal affect & explainability**, next strong steps are:

1. ✅ **Add video playback synced with transcript**
2. ✅ **Segment-level sentiment timeline (interactive)**
3. ✅ **Explainability page** (lexicon vs VADER vs OpenAI disagreement)
4. ✅ **Batch comparison across data_2 / data_3**
5. ✅ **Export figures for paper-ready plots**

If you want, I can:

* Split this into **pages/** (`Overview`, `Transcript`, `Sentiment`, `Explainability`)
* Add **video + audio waveform**
* Add **attention-style highlighting of emotional segments**

Just tell me which direction you want to push 🚀
