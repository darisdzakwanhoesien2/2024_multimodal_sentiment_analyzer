import streamlit as st
from services.pipeline import process_youtube

st.title("YouTube → Whisper → Sentiment Dashboard")

url = st.text_input("Enter YouTube URL")

if st.button("Process"):
    folder, df = process_youtube(url)
    st.success(f"Completed! Output stored at: {folder}")
    st.dataframe(df)