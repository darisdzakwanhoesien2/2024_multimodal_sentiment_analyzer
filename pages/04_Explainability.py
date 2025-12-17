import streamlit as st
import pandas as pd

from ui.utils import list_datasets, load_csv

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Explainability & Model Disagreement")

st.markdown("""
This page helps you **understand why sentiment models disagree**  
by comparing **lexicon-based**, **VADER**, and **OpenAI-based** sentiment scores
at the **segment level**.
""")

# =====================================================
# Dataset selection
# =====================================================
dataset = st.selectbox(
    "Select dataset",
    list_datasets(),
    format_func=lambda p: p.name
)

csv_path = dataset / "transcription_result.csv"
df = load_csv(csv_path)

if df is None:
    st.warning("Sentiment CSV not found. Please run sentiment generation first.")
    st.stop()

# =====================================================
# Detect available sentiment sources
# =====================================================
HAS_OPENAI = "openai_mean_sentiment" in df.columns

# =====================================================
# Explainability metrics
# =====================================================

# Lexicon vs VADER disagreement
df["lexicon_vs_vader"] = (
    df["nltk_opinion_lexicon_net"] - df["vader_compound"]
)

# Lexicon vs OpenAI (if available)
if HAS_OPENAI:
    df["lexicon_vs_openai"] = (
        df["nltk_opinion_lexicon_net"] - df["openai_mean_sentiment"]
    )

# VADER vs OpenAI (if available)
if HAS_OPENAI:
    df["vader_vs_openai"] = (
        df["vader_compound"] - df["openai_mean_sentiment"]
    )

# =====================================================
# Segment-level table
# =====================================================
st.subheader("📊 Segment-Level Sentiment Disagreement")

BASE_COLS = [
    "start", "end",
    "nltk_opinion_lexicon_net",
    "vader_compound",
    "lexicon_vs_vader"
]

OPTIONAL_COLS = []
if HAS_OPENAI:
    OPTIONAL_COLS += [
        "openai_mean_sentiment",
        "lexicon_vs_openai",
        "vader_vs_openai"
    ]

display_cols = BASE_COLS + OPTIONAL_COLS

st.dataframe(
    df[display_cols],
    use_container_width=True,
    height=450
)

# =====================================================
# Aggregate disagreement analysis
# =====================================================
st.subheader("📉 Aggregate Disagreement Statistics")

summary_rows = []

summary_rows.append({
    "comparison": "Lexicon vs VADER",
    "mean_abs_diff": df["lexicon_vs_vader"].abs().mean(),
    "std_diff": df["lexicon_vs_vader"].std()
})

if HAS_OPENAI:
    summary_rows.append({
        "comparison": "Lexicon vs OpenAI",
        "mean_abs_diff": df["lexicon_vs_openai"].abs().mean(),
        "std_diff": df["lexicon_vs_openai"].std()
    })
    summary_rows.append({
        "comparison": "VADER vs OpenAI",
        "mean_abs_diff": df["vader_vs_openai"].abs().mean(),
        "std_diff": df["vader_vs_openai"].std()
    })

summary_df = pd.DataFrame(summary_rows)

st.dataframe(
    summary_df,
    use_container_width=True
)

# =====================================================
# Interpretation guide
# =====================================================
st.markdown("---")
st.subheader("🧠 How to Interpret These Results")

st.markdown("""
### What disagreement means

- **Lexicon-based models**  
  Count emotionally charged words → sensitive to vocabulary

- **VADER**  
  Rule-based with valence + punctuation → better for short emotional bursts

- **OpenAI**  
  Context-aware → understands narrative, irony, and discourse

### Large disagreement often indicates:
- Ambiguous language
- Narrative framing (storytelling vs emotion words)
- Moral or ideological language
- Non-literal expressions

### Research insight
Segments with **high disagreement** are often the **most interesting**  
for qualitative analysis, annotation, or model improvement.
""")

# =====================================================
# Flags for analyst attention
# =====================================================
st.subheader("🚩 High-Disagreement Segments")

threshold = st.slider(
    "Absolute disagreement threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05
)

flagged = df[df["lexicon_vs_vader"].abs() >= threshold]

if HAS_OPENAI:
    flagged = flagged[
        (flagged["lexicon_vs_openai"].abs() >= threshold) |
        (flagged["vader_vs_openai"].abs() >= threshold)
    ]

if flagged.empty:
    st.info("No segments exceed the selected threshold.")
else:
    st.dataframe(
        flagged[display_cols],
        use_container_width=True
    )
