import streamlit as st
from ui.utils import list_datasets, load_json

st.set_page_config(layout="wide")
st.title("📦 Batch Comparison")

rows = []

for ds in list_datasets():
    sentiment = load_json(ds / "transcription_result_new.json")
    if sentiment:
        rows.append({
            "dataset": ds.name,
            "mean_sentiment": sentiment["overall_mean_sentiment"],
            "std_sentiment": sentiment["overall_standard_deviation"],
            "combined_std": sentiment["combined_standard_deviation"]
        })

if not rows:
    st.info("No sentiment results found")
else:
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.subheader("📊 Mean Sentiment Comparison")
    st.bar_chart(df.set_index("dataset")["mean_sentiment"])
