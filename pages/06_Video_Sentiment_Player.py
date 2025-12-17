import streamlit as st
import pandas as pd
import json
from pathlib import Path

import plotly.graph_objects as go

from ui.utils import list_datasets, load_csv, load_json
from pipeline.run_lexicon_pipeline import run_lexicon_pipeline

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🎥 Video + Subtitle + Sentiment Timeline")

st.markdown("""
This page synchronizes:

• 🎥 Video playback  
• 📝 Whisper subtitles  
• 📈 Sentiment scores over time  

to support **multimodal affect analysis**.
""")

# =====================================================
# Dataset selection
# =====================================================
dataset = st.selectbox(
    "Select dataset",
    list_datasets(),
    format_func=lambda p: p.name
)

# =====================================================
# Locate files
# =====================================================
video_file = None
for ext in [".mp4", ".webm", ".mov"]:
    candidates = list(dataset.glob(f"*{ext}"))
    if candidates:
        video_file = candidates[0]
        break

json_path = dataset / "transcription_result.json"
csv_path = dataset / "transcription_result.csv"

if not json_path.exists():
    st.error("transcription_result.json not found.")
    st.stop()

# =====================================================
# Load or generate sentiment
# =====================================================
df = load_csv(csv_path)
if df is None:
    st.info("Sentiment not found — generating lexicon + VADER sentiment.")
    with st.spinner("Running sentiment pipeline..."):
        df = run_lexicon_pipeline(dataset)

# =====================================================
# Load transcription
# =====================================================
with open(json_path, "r", encoding="utf-8") as f:
    transcription = json.load(f)

segments = transcription["segments"]

# =====================================================
# Layout
# =====================================================
left, right = st.columns([2, 3])

# =====================================================
# LEFT: Video + subtitles
# =====================================================
with left:
    st.subheader("🎥 Video Playback")

    if video_file:
        st.video(str(video_file))
    else:
        st.warning("No video file found in dataset.")

    st.subheader("📝 Subtitles")

    subtitle_time = st.slider(
        "Jump to time (seconds)",
        min_value=0.0,
        max_value=float(df["end"].max()),
        value=0.0,
        step=0.5
    )

    active_segments = [
        seg for seg in segments
        if seg["start"] <= subtitle_time <= seg["end"]
    ]

    if active_segments:
        st.markdown(
            f"**[{active_segments[0]['start']:.2f}s – "
            f"{active_segments[0]['end']:.2f}s]**\n\n"
            f"{active_segments[0]['text']}"
        )
    else:
        st.info("No subtitle at this time.")

# =====================================================
# RIGHT: Sentiment timeline
# =====================================================
with right:
    st.subheader("📈 Sentiment Timeline")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["start"],
        y=df["nltk_opinion_lexicon_net"],
        mode="lines+markers",
        name="Lexicon Net",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df["start"],
        y=df["vader_compound"],
        mode="lines+markers",
        name="VADER Compound",
        line=dict(color="green")
    ))

    if "openai_mean_sentiment" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["start"],
            y=df["openai_mean_sentiment"],
            mode="lines+markers",
            name="OpenAI Sentiment",
            line=dict(color="red")
        ))

    fig.add_vline(
        x=subtitle_time,
        line_width=2,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Sentiment score",
        height=500,
        legend_title="Sentiment Model",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Segment table (optional)
# =====================================================
with st.expander("📊 Segment-level sentiment table"):
    cols = [
        "start", "end",
        "nltk_opinion_lexicon_net",
        "vader_compound"
    ]
    if "openai_mean_sentiment" in df.columns:
        cols.append("openai_mean_sentiment")

    st.dataframe(
        df[cols],
        use_container_width=True
    )
