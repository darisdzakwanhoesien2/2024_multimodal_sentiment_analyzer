import streamlit as st
from datasets import load_dataset

st.title("📊 Explore Streaming Data (OpenFoodFacts)")

st.markdown(
    "Preview a small streamed sample from the OpenFoodFacts product dataset. "
    "Streaming mode avoids local downloads — results are limited by the slider below."
)

@st.cache_data
def get_streaming_dataset():
    # streaming=True ensures no local dataset storage
    return load_dataset("openfoodfacts/product-database", split="food", streaming=True)

dataset = get_streaming_dataset()

num_samples = st.slider("Number of samples to preview", min_value=5, max_value=50, value=10, step=1)

data = []
for i, item in enumerate(dataset):
    try:
        data.append({
            "product": item.get("product_name") or "",
            "brand": item.get("brands") or "",
            "country": item.get("countries") or "",
            "nutriscore": item.get("nutriscore_grade") or "",
            "categories": item.get("categories") or "",
            "image": item.get("image_small_url") or ""
        })
    except Exception:
        # skip malformed records
        continue

    if i >= (num_samples - 1):
        break

st.write(f"Previewing {len(data)} items (streamed)")
st.dataframe(data)

with st.expander("Show raw JSON of first item"):
    if data:
        st.json(data[0])
    else:
        st.write("No data to show. Try increasing the sample slider or check your network.")