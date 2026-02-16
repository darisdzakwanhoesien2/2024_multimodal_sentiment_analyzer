import streamlit as st
from core.whisper_api import transcribe, save_transcription
from core.segmentation import transcription_json_to_df
import os
import pandas as pd
import time

BASE_DIR = "public/packages"

st.title("📝 Generate Transcription")

# List existing video folders
if not os.path.exists(BASE_DIR):
    st.error("No videos downloaded. Go to Page 1 first.")
    st.stop()

folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

if not folders:
    st.warning("No downloaded videos found.")
    st.stop()

selected = st.selectbox("Choose a video folder:", folders)
folder_path = os.path.join(BASE_DIR, selected)

audio_path = os.path.join(folder_path, "audio.mp3")
json_path = os.path.join(folder_path, "transcription.json")

st.write(f"📁 Selected Folder: `{selected}`")
st.write(f"🎧 Audio file: `{audio_path}`")

if not os.path.exists(audio_path):
    st.error("Audio file missing! Re-download the video in Page 1.")
    st.stop()

# UI: Choose transcription engine
st.subheader("Choose Transcription Engine")
engine = st.radio(
    "Transcription Engine",
    ["OpenAI Whisper (API)", "Whisper.cpp (local) – Coming Soon", "SpeechRecognition (Google)"],
    index=0
)

if st.button("Start Transcription"):
    st.markdown("### 🏃 Processing…")

    progress = st.progress(0)
    status = st.empty()

    try:
        # Step 1: Preparing
        status.info("Preparing transcription...")
        time.sleep(0.5)
        progress.progress(10)

        # Step 2: Run transcription
        status.info(f"Running transcription using: **{engine}**")

        if engine == "OpenAI Whisper (API)":
            result = transcribe(audio_path)

        elif engine == "SpeechRecognition (Google)":
            from core.transcriber_google import google_transcribe
            result = google_transcribe(audio_path)

        elif engine == "Whisper.cpp (local) – Coming Soon":
            st.warning("Whisper.cpp backend is not implemented yet.")
            st.stop()

        progress.progress(50)

        # Step 3: Save JSON
        status.info("Saving transcription results...")
        save_transcription(result, json_path)
        progress.progress(70)

        # Step 4: Create DataFrame
        status.info("Converting to DataFrame...")
        df = transcription_json_to_df(json_path)
        progress.progress(90)

        status.success("Transcription Completed!")
        progress.progress(100)

        st.subheader("📄 Transcription Segments")
        st.dataframe(df)

    except Exception as e:
        status.error(f"❌ Error: {str(e)}")
        st.exception(e)


# import streamlit as st
# from core.whisper_api import transcribe, save_transcription
# from core.segmentation import transcription_json_to_df
# import os
# import pandas as pd

# BASE_DIR = "public/packages"

# st.title("📝 Generate Transcription")

# # List available packages
# if not os.path.exists(BASE_DIR):
#     st.error("No videos downloaded yet.")
#     st.stop()

# folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

# if not folders:
#     st.warning("No downloaded videos found. Go to Page 1 to download first.")
#     st.stop()

# selected = st.selectbox("Choose a downloaded video folder:", folders)

# folder_path = os.path.join(BASE_DIR, selected)
# audio_path = os.path.join(folder_path, "audio.mp3")
# json_path = os.path.join(folder_path, "transcription.json")

# st.write(f"**Audio file:** {audio_path}")

# if not os.path.exists(audio_path):
#     st.error("Audio file not found. You must download again.")
#     st.stop()

# if st.button("Generate Transcription"):
#     with st.spinner("Transcribing audio..."):
#         result = transcribe(audio_path)
#         save_transcription(result, json_path)

#     st.success(f"Transcription saved to {json_path}")

#     # Load segments into dataframe
#     df = transcription_json_to_df(json_path)
#     st.dataframe(df)

#     st.info("Next: sentiment analysis (Page 3)")
