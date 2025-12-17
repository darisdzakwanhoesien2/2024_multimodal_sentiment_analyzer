import streamlit as st
from ui.utils import list_datasets, load_json

st.set_page_config(layout="wide")
st.title("📝 Transcript Viewer")

dataset = st.selectbox(
    "Select dataset",
    list_datasets(),
    format_func=lambda p: p.name
)

transcription = load_json(dataset / "transcription_result.json")

if not transcription:
    st.warning("No transcription_result.json found")
    st.stop()

st.text_area(
    "Full Transcript",
    transcription.get("transcription_text", ""),
    height=250
)

st.subheader("📍 Segment-level Transcript")

for seg in transcription["segments"]:
    st.markdown(
        f"**[{seg['start']:.2f}s – {seg['end']:.2f}s]** {seg['text']}"
    )
