import streamlit as st
from services.pipeline import process_youtube

st.title("🎧 YouTube → Whisper → Sentiment Transcriber")

url = st.text_input("Enter a YouTube URL")

if st.button("Start Processing"):
    folder, df = process_youtube(url)
    st.success(f"Processing complete! Output stored in: {folder}")
    st.dataframe(df)
