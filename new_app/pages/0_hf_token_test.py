import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="HF Token Test", page_icon="🔑", layout="wide")
st.title("🔑 Hugging Face Token Tester")
st.markdown("Use this page to verify your token and model access before running diarization.")

MODELS_TO_CHECK = [
    "pyannote/speaker-diarization-3.0",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]

# ── local model cache directory ───────────────────────────────────────────────
DEFAULT_LOCAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "wespeaker-voxceleb-resnet34-LM"
)
DEFAULT_LOCAL_DIR = os.path.normpath(DEFAULT_LOCAL_DIR)

# ── initialise session state ──────────────────────────────────────────────────
defaults = {
    "token_tested":    False,
    "token_valid":     False,
    "login_error":     "",
    "whoami_info":     None,
    "model_results":   [],
    "pipeline_loaded": False,
    "pipeline_repr":   "",
    "pipeline_error":  "",
    "pipeline_model":  "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── token input ───────────────────────────────────────────────────────────────
hf_token = st.text_input(
    "Hugging Face Token",
    type="password",
    value=os.getenv("HF_TOKEN", ""),
    help="Get your token at https://huggingface.co/settings/tokens (read scope required)",
    key="hf_token_input",
)

st.markdown("---")

# ── Steps 1–4: Token test ─────────────────────────────────────────────────────
st.subheader("Steps 1–4 — Token validation & model access")

if st.button("🔍 Test Token", type="primary", disabled=not hf_token):
    st.session_state.token_tested  = True
    st.session_state.token_valid   = False
    st.session_state.login_error   = ""
    st.session_state.whoami_info   = None
    st.session_state.model_results = []

    # Step 2: login
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token, add_to_git_credential=False)
        st.session_state.token_valid = True
    except Exception as e:
        st.session_state.login_error = str(e)

    # Step 3: whoami
    if st.session_state.token_valid:
        try:
            from huggingface_hub import whoami
            st.session_state.whoami_info = whoami(token=hf_token)
        except Exception as e:
            st.session_state.whoami_info = {"error": str(e)}

    # Step 4: model access
    from huggingface_hub import model_info
    results = []
    for repo_id in MODELS_TO_CHECK:
        try:
            info = model_info(repo_id, token=hf_token)
            results.append({
                "Model":             repo_id,
                "Status":            "✅ Accessible",
                "Gated":             str(getattr(info, "gated", "N/A")),
                "Downloads (month)": getattr(info, "downloads", "N/A"),
            })
        except Exception as e:
            err = str(e)
            if "403" in err:
                status = "❌ Access denied (403) — accept model on HF"
            elif "404" in err:
                status = "⚠️ Not found (404)"
            else:
                status = f"❌ {err[:80]}"
            results.append({"Model": repo_id, "Status": status,
                            "Gated": "N/A", "Downloads (month)": "N/A"})
    st.session_state.model_results = results

# ── render Steps 1–4 results ──────────────────────────────────────────────────
if st.session_state.token_tested:
    # Step 1
    st.markdown("**Step 1 — Token format**")
    if hf_token.startswith("hf_") and len(hf_token) > 20:
        st.success("✅ Token format looks valid.")
    else:
        st.warning("⚠️ Token doesn't start with `hf_` — double-check you copied it correctly.")

    # Step 2
    st.markdown("**Step 2 — Hugging Face login**")
    if st.session_state.token_valid:
        st.success("✅ `hf_login()` succeeded.")
    else:
        st.error(f"❌ Login failed: {st.session_state.login_error or 'unknown error'}")

    # Step 3
    st.markdown("**Step 3 — Token identity**")
    info = st.session_state.whoami_info
    if info and "error" not in info:
        st.success(f"✅ Logged in as **{info['name']}** (`{info['type']}`)")
        with st.expander("Full whoami response"):
            st.json(info)
    elif info:
        st.error(f"❌ whoami failed: {info['error']}")

    # Step 4
    st.markdown("**Step 4 — Model access**")
    if st.session_state.model_results:
        st.dataframe(pd.DataFrame(st.session_state.model_results), use_container_width=True)

st.markdown("---")

# ── Step 5: Load pipeline ─────────────────────────────────────────────────────
st.subheader("Step 5 — Load pipeline (dry run)")
st.info("This will actually download and load the model weights. It may take 1–2 minutes the first time.")

selected_model = st.selectbox(
    "Choose model to test-load",
    MODELS_TO_CHECK[:2],
    key="test_load_model",
)

if st.button("🚀 Load pipeline now", disabled=not hf_token):
    # reset previous result
    st.session_state.pipeline_loaded = False
    st.session_state.pipeline_repr   = ""
    st.session_state.pipeline_error  = ""
    st.session_state.pipeline_model  = selected_model

    try:
        from pyannote.audio import Pipeline as PyPipeline
        with st.spinner(f"Loading `{selected_model}`…"):
            pipe = PyPipeline.from_pretrained(selected_model, token=hf_token)
        st.session_state.pipeline_loaded = True
        st.session_state.pipeline_repr   = repr(pipe)[:500]
    except Exception as e:
        st.session_state.pipeline_error = str(e)

# render Step 5 result
if st.session_state.pipeline_loaded:
    st.success(f"✅ Pipeline `{st.session_state.pipeline_model}` loaded successfully!")
    st.code(st.session_state.pipeline_repr)
elif st.session_state.pipeline_error:
    st.error(f"❌ Failed to load pipeline: {st.session_state.pipeline_error}")
    st.markdown(
        "**Possible fixes:**\n"
        "- Make sure you accepted the model on Hugging Face\n"
        "- Check your token has `read` scope\n"
        "- Run `pip install -U pyannote.audio huggingface_hub`"
    )

st.markdown("---")

# ── Step 6: Environment info ──────────────────────────────────────────────────
st.subheader("Step 6 — Environment info")
try:
    import pyannote.audio
    import pyannote.core
    import torch
    import huggingface_hub

    st.dataframe(pd.DataFrame([
        {"Package": "pyannote.audio",  "Version": pyannote.audio.__version__},
        {"Package": "pyannote.core",   "Version": pyannote.core.__version__},
        {"Package": "torch",           "Version": torch.__version__},
        {"Package": "huggingface_hub", "Version": huggingface_hub.__version__},
    ]), use_container_width=True)

    major, minor = (int(x) for x in huggingface_hub.__version__.split(".")[:2])
    if (major, minor) >= (0, 17):
        st.success("✅ `huggingface_hub` supports `token=` kwarg (v0.17+).")
    else:
        st.warning("⚠️ Old `huggingface_hub`. Run: `pip install -U huggingface_hub`")
except Exception as e:
    st.warning(f"Could not retrieve environment info: {e}")