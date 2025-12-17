import streamlit as st
from ui.utils import list_datasets, load_json

st.set_page_config(layout="wide")
st.title("📊 Project Overview")

datasets = list_datasets()
dataset = st.selectbox(
    "Select dataset",
    datasets,
    format_func=lambda p: p.name
)

video_info = load_json(dataset / "video_info.json")
sentiment = load_json(dataset / "transcription_result_new.json")

st.subheader("🎥 Video Metadata")

if video_info:
    st.json(video_info)
else:
    st.info("video_info.json not found")

if sentiment:
    st.subheader("🧠 Overall Sentiment Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Sentiment", round(sentiment["overall_mean_sentiment"], 3))
    c2.metric("Avg Std Dev", round(sentiment["overall_standard_deviation"], 3))
    c3.metric("Combined Std Dev", round(sentiment["combined_standard_deviation"], 3))
