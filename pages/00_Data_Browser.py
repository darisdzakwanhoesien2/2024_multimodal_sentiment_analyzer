import streamlit as st
from pathlib import Path
import json
import pandas as pd
from PIL import Image

from ui.utils import list_datasets

# =====================================================
# Page setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🗂 Dataset & File Browser")

st.markdown("""
This page provides a **raw, transparent view** of everything inside your
`data/` directory.

No assumptions. No pipelines. Just **what exists**.
""")

# =====================================================
# Dataset selection
# =====================================================
datasets = list_datasets()

if not datasets:
    st.error("❌ No datasets found under data/")
    st.stop()

dataset = st.selectbox(
    "Select dataset folder",
    datasets,
    format_func=lambda p: p.name
)

st.divider()

# =====================================================
# List files
# =====================================================
st.subheader("📁 Files in dataset")

files = sorted([p for p in dataset.iterdir() if p.is_file()])

if not files:
    st.info("No files found in this dataset.")
    st.stop()

file_names = [f.name for f in files]

selected_file = st.selectbox(
    "Select a file to preview",
    file_names
)

file_path = dataset / selected_file

st.markdown(f"**Full path:** `{file_path}`")
st.divider()

# =====================================================
# File preview logic
# =====================================================
suffix = file_path.suffix.lower()

# -----------------------------
# VIDEO
# -----------------------------
if suffix in [".mp4", ".webm", ".mov", ".m4a"]:
    st.subheader("🎥 Video / Media Preview")
    st.video(str(file_path))

# -----------------------------
# AUDIO
# -----------------------------
elif suffix in [".mp3", ".wav"]:
    st.subheader("🎧 Audio Preview")
    st.audio(str(file_path))

# -----------------------------
# IMAGE
# -----------------------------
elif suffix in [".png", ".jpg", ".jpeg"]:
    st.subheader("🖼 Image Preview")
    st.image(Image.open(file_path), use_container_width=True)

# -----------------------------
# JSON
# -----------------------------
elif suffix == ".json":
    st.subheader("📝 JSON Preview")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.json(data)
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")

# -----------------------------
# CSV
# -----------------------------
elif suffix == ".csv":
    st.subheader("📊 CSV Preview")
    try:
        df = pd.read_csv(file_path)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

# -----------------------------
# OTHER FILE TYPES
# -----------------------------
else:
    st.subheader("📄 File Info")
    st.info(f"Preview not supported for `{suffix}` files.")
    st.markdown(f"**File size:** {file_path.stat().st_size / 1024:.2f} KB")
