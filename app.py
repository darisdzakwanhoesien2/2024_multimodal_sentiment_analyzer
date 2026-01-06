import streamlit as st
from PIL import Image

from ui.utils import list_datasets, load_csv, load_json

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
generated from your **multimodal pipeline (audio + text)**.
""")

# =====================================================
# Sidebar — dataset selection
# =====================================================
st.sidebar.header("📁 Dataset Selection")

DATASETS = list_datasets()

if not DATASETS:
    st.error("❌ No datasets found in data/")
    st.stop()

dataset = st.sidebar.selectbox(
    "Select processed dataset",
    DATASETS,
    format_func=lambda p: p.name
)

show_transcript = st.sidebar.checkbox("Show full transcript", True)
show_tables = st.sidebar.checkbox("Show sentiment tables", True)
show_plots = st.sidebar.checkbox("Show plots", True)

# =====================================================
# Load artifacts (safe)
# =====================================================
video_info = load_json(dataset / "video_info.json")
transcription = load_json(dataset / "transcription_result.json")
sentiment_json = load_json(dataset / "transcription_result_new.json")

df = load_csv(dataset / "transcription_result.csv")

metrics_plot_path = dataset / "metrics_plot.png"
corr_plot_path = dataset / "correlation_matrix.png"

# =====================================================
# Video metadata
# =====================================================
st.subheader("📄 Video Metadata")

if video_info:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**Title:** {video_info.get('title', 'N/A')}")
        st.markdown(f"**Uploader:** {video_info.get('uploader', 'N/A')}")
        st.markdown(f"**Upload Date:** {video_info.get('upload_date', 'N/A')}")

    with c2:
        st.markdown(f"**Duration:** {video_info.get('duration', 'N/A')} seconds")
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
        st.text_area(
            "Full Transcript",
            transcription.get("transcription_text", ""),
            height=250
        )

        with st.expander("🔍 Segment-level transcript"):
            for seg in transcription["segments"]:
                st.markdown(
                    f"**[{seg['start']:.2f}s – {seg['end']:.2f}s]** {seg['text']}"
                )
    else:
        st.warning("Transcription not found.")

st.divider()

# =====================================================
# Sentiment tables
# =====================================================
if show_tables and df is not None:
    st.subheader("📊 Segment-level Sentiment Metrics")

    cols = [
        "start", "end",
        "nltk_opinion_lexicon_net",
        "vader_neg", "vader_neu", "vader_pos", "vader_compound",
        "openai_mean_sentiment", "std_sentiment"
    ]

    available_cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df[available_cols],
        use_container_width=True,
        height=400
    )
elif show_tables:
    st.info("No sentiment CSV found for this dataset.")

st.divider()

# =====================================================
# Plots
# =====================================================
if show_plots:
    st.subheader("📈 Visual Analytics")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Sentiment Metrics Over Time**")
        if metrics_plot_path.exists():
            st.image(Image.open(metrics_plot_path), use_container_width=True)
        else:
            st.info("metrics_plot.png not found.")

    with c2:
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Sentiment", round(sentiment_json.get("overall_mean_sentiment", 0), 3))
    c2.metric("Avg Std Dev", round(sentiment_json.get("overall_standard_deviation", 0), 3))
    c3.metric("Combined Std Dev", round(sentiment_json.get("combined_standard_deviation", 0), 3))
else:
    st.info("No overall sentiment summary available.")

# import streamlit as st
# import os
# import json
# import pandas as pd
# from pathlib import Path
# from PIL import Image

# # =====================================================
# # Streamlit setup
# # =====================================================
# st.set_page_config(
#     page_title="Multimodal Sentiment Explorer",
#     layout="wide"
# )

# st.title("🎥 Multimodal Affect & Sentiment Explorer")
# st.markdown("""
# This app visualizes **video metadata, transcription, and sentiment analysis**
# generated from your multimodal pipeline (audio + text).
# """)

# ROOT_DIR = Path(__file__).parent
# DATA_DIRS = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.startswith("data_")])

# # =====================================================
# # Sidebar: dataset selection
# # =====================================================
# st.sidebar.header("📁 Dataset Selection")

# if not DATA_DIRS:
#     st.error("No data folders (data_*) found.")
#     st.stop()

# dataset = st.sidebar.selectbox(
#     "Select processed dataset",
#     DATA_DIRS,
#     format_func=lambda p: p.name
# )

# # Toggle controls
# show_transcript = st.sidebar.checkbox("Show full transcript", True)
# show_tables = st.sidebar.checkbox("Show sentiment tables", True)
# show_plots = st.sidebar.checkbox("Show plots", True)

# # =====================================================
# # Load helper
# # =====================================================
# def load_json(path):
#     if path.exists():
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return None

# # =====================================================
# # Load files
# # =====================================================
# video_info = load_json(dataset / "video_info.json")
# transcription = load_json(dataset / "transcription_result.json")
# sentiment_json = load_json(dataset / "transcription_result_new.json")

# csv_path = dataset / "transcription_result.csv"
# metrics_plot_path = dataset / "metrics_plot.png"
# corr_plot_path = dataset / "correlation_matrix.png"

# df = pd.read_csv(csv_path) if csv_path.exists() else None

# # =====================================================
# # Video metadata
# # =====================================================
# st.subheader("📄 Video Metadata")

# if video_info:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown(f"**Title:** {video_info.get('title')}")
#         st.markdown(f"**Uploader:** {video_info.get('uploader')}")
#         st.markdown(f"**Upload Date:** {video_info.get('upload_date')}")

#     with col2:
#         st.markdown(f"**Duration:** {video_info.get('duration')} seconds")
#         st.markdown("**Description:**")
#         st.write(video_info.get("description", "")[:500] + "…")
# else:
#     st.warning("video_info.json not found.")

# st.divider()

# # =====================================================
# # Transcript
# # =====================================================
# if show_transcript:
#     st.subheader("📝 Transcription")

#     if transcription and "segments" in transcription:
#         transcript_text = transcription.get("transcription_text", "")
#         st.text_area(
#             "Full Transcript",
#             transcript_text,
#             height=250
#         )

#         with st.expander("🔍 Segment-level transcript"):
#             for seg in transcription["segments"]:
#                 st.markdown(
#                     f"**[{seg['start']:.2f}s – {seg['end']:.2f}s]** {seg['text']}"
#                 )
#     else:
#         st.warning("Transcription file missing or invalid.")

# st.divider()

# # =====================================================
# # Sentiment tables
# # =====================================================
# if show_tables and df is not None:
#     st.subheader("📊 Segment-level Sentiment Metrics")

#     cols_to_show = [
#         "start", "end",
#         "nltk_opinion_lexicon_net",
#         "vader_neg", "vader_neu", "vader_pos", "vader_compound",
#         "openai_mean_sentiment", "std_sentiment"
#     ]

#     available_cols = [c for c in cols_to_show if c in df.columns]

#     st.dataframe(
#         df[available_cols],
#         use_container_width=True,
#         height=400
#     )

# st.divider()

# # =====================================================
# # Plots
# # =====================================================
# if show_plots:
#     st.subheader("📈 Visual Analytics")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("**Sentiment Metrics Over Time**")
#         if metrics_plot_path.exists():
#             st.image(Image.open(metrics_plot_path), use_container_width=True)
#         else:
#             st.info("metrics_plot.png not found.")

#     with col2:
#         st.markdown("**Correlation Matrix**")
#         if corr_plot_path.exists():
#             st.image(Image.open(corr_plot_path), use_container_width=True)
#         else:
#             st.info("correlation_matrix.png not found.")

# # =====================================================
# # Summary
# # =====================================================
# if sentiment_json:
#     st.subheader("🧠 Overall Sentiment Summary")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Mean Sentiment", round(sentiment_json["overall_mean_sentiment"], 3))
#     col2.metric("Avg Std Dev", round(sentiment_json["overall_standard_deviation"], 3))
#     col3.metric("Combined Std Dev", round(sentiment_json["combined_standard_deviation"], 3))
