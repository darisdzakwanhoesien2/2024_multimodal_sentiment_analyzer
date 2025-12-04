import streamlit as st
from services.pipeline import process_youtube
import os
import pandas as pd

st.set_page_config(page_title="YouTube → Whisper → Sentiment", layout="wide")
st.title("🎧 YouTube → Whisper → Sentiment Transcriber (VADER removed)")

st.markdown("""
Enter a YouTube URL and click `Process`.  
This will download the video, extract audio, transcribe with Whisper, run lexicon-based sentiment, expand to time series, and produce plots.  
You can optionally enable iterative OpenAI sentiment (costly).
""")

url = st.text_input("YouTube URL", value="https://www.youtube.com/watch?v=your_video_here")
run_openai = st.checkbox("Run iterative OpenAI sentiment (may cost API credits)", value=False)
openai_iters = st.number_input("OpenAI iterations per segment", min_value=1, max_value=10, value=3)

if st.button("Process"):
    if not url or url.strip() == "":
        st.error("Please provide a valid YouTube URL.")
    else:
        with st.spinner("Processing — this may take a while depending on video length..."):
            try:
                result = process_youtube(url.strip(), run_openai_sentiment=run_openai, openai_iters=int(openai_iters))
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.exception(e)
            else:
                st.success("Processing finished.")
                st.write("Output folder:", result["folder"])
                if result["plot"] and os.path.exists(result["plot"]):
                    st.image(result["plot"], use_column_width=True)
                df = result["dataframe"]
                if isinstance(df, pd.DataFrame):
                    st.dataframe(df.head(200))
                    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="transcription_result.csv", mime="text/csv")
