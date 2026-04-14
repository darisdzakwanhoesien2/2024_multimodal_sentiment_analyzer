import streamlit as st
import json
import os
import re
import time
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import requests

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pear AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
# ORIGINAL:
# DATA_DIR         = Path(__file__).parent.parent / "data"
# CHAT_HISTORY_DIR = DATA_DIR / "chat_history"
# CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# REPLACED: robust data directory locator (searches common locations and creates fallback)
def locate_data_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data",               # halal_product/data
        here.parent.parent.parent / "data",        # multimodal/data
        here.parent.parent.parent.parent / "data", # computer_vision/data
        Path.cwd() / "data",                       # current working dir /data
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    # fallback: create data next to this package
    fallback = here.parent.parent / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

DATA_DIR = locate_data_dir()
CHAT_HISTORY_DIR = DATA_DIR / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

THESIS_DATASET_DIR = DATA_DIR / "thesis_dataset"   # data/thesis_dataset/<doc>_pdf/images/
IMAGES_DIR         = DATA_DIR / "images"            # data/images/<subfolder>/

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# Models known to support vision — used as fallback when API doesn't expose modality
VISION_MODEL_KEYWORDS = [
    "gpt-4o", "gpt-4-vision", "claude-3", "claude-3.5",
    "gemini", "llava", "vision", "pixtral", "qwen-vl",
    "intern-vl", "minicpm-v", "phi-3-vision",
]

# ── RAG / LLM constants ────────────────────────────────────────────────────────
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONTEXT_CHARS  = 12_000   # max total chars injected into system prompt
CHUNK_PREVIEW_LEN  = 500      # chars shown in reference preview expanders


# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    if st.session_state.get("api_key", "").strip():
        return st.session_state["api_key"].strip()
    try:
        from config.settings import settings
        for attr in ("OPENROUTER_API_KEY", "openrouter_api_key", "api_key"):
            val = getattr(settings, attr, None)
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return os.getenv("OPENROUTER_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# OPENROUTER MODEL FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def _FALLBACK_MODELS() -> list[dict]:
    return [
        {"id": "meta-llama/llama-3.1-8b-instruct:free",  "label": "Llama 3.1 8B",      "free": True,  "vision": False, "notes": "free · 131,072 ctx", "ctx": 131072},
        {"id": "meta-llama/llama-3.3-70b-instruct:free",  "label": "Llama 3.3 70B",      "free": True,  "vision": False, "notes": "free · 131,072 ctx", "ctx": 131072},
        {"id": "google/gemma-3-27b-it:free",              "label": "Gemma 3 27B",         "free": True,  "vision": False, "notes": "free · 131,072 ctx", "ctx": 131072},
        {"id": "deepseek/deepseek-r1:free",               "label": "DeepSeek R1",         "free": True,  "vision": False, "notes": "free · 65,536 ctx",  "ctx": 65536},
        {"id": "openai/gpt-4o-mini",                      "label": "GPT-4o Mini",         "free": False, "vision": True,  "notes": "$0.150/1M · 128,000 ctx", "ctx": 128000},
        {"id": "openai/gpt-4o",                           "label": "GPT-4o",              "free": False, "vision": True,  "notes": "$2.500/1M · 128,000 ctx", "ctx": 128000},
        {"id": "anthropic/claude-3.5-sonnet",             "label": "Claude 3.5 Sonnet",   "free": False, "vision": True,  "notes": "$3.000/1M · 200,000 ctx", "ctx": 200000},
        {"id": "anthropic/claude-3.5-haiku",              "label": "Claude 3.5 Haiku",    "free": False, "vision": True,  "notes": "$0.800/1M · 200,000 ctx", "ctx": 200000},
        {"id": "google/gemini-flash-1.5",                 "label": "Gemini 1.5 Flash",    "free": False, "vision": True,  "notes": "$0.075/1M · 1,000,000 ctx", "ctx": 1000000},
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openrouter_models() -> list[dict]:
    api_key = _get_api_key()
    if not api_key:
        return _FALLBACK_MODELS()
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer":  "https://pear-edtech.app",
                "X-Title":       "Pear EdTech Chatbot",
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])

        models = []
        for m in raw:
            mid      = m.get("id", "")
            name     = m.get("name", mid)
            ctx      = m.get("context_length", 0)
            pricing  = m.get("pricing", {})

            # ── Free / paid ────────────────────────────────────────────────
            try:
                p_cost  = float(pricing.get("prompt", 1))
                c_cost  = float(pricing.get("completion", 1))
                is_free = p_cost == 0.0 and c_cost == 0.0
            except (ValueError, TypeError):
                is_free = str(pricing.get("prompt", "1")) == "0"

            if is_free:
                cost_str = "free"
            else:
                try:
                    cost_str = f"${float(pricing.get('prompt', 0)) * 1_000_000:.3f}/1M"
                except Exception:
                    cost_str = "paid"

            # ── Vision support ─────────────────────────────────────────────
            # OpenRouter exposes architecture.modality or architecture.input_modalities
            arch          = m.get("architecture", {})
            modality      = arch.get("modality", "")
            input_mods    = arch.get("input_modalities", [])
            has_vision = (
                "image" in modality
                or "image" in input_mods
                or "multimodal" in modality
                or any(kw in mid.lower() for kw in VISION_MODEL_KEYWORDS)
                or any(kw in name.lower() for kw in VISION_MODEL_KEYWORDS)
            )

            ctx_str = f"{ctx:,} ctx" if ctx else ""
            notes   = " · ".join(filter(None, [cost_str, ctx_str]))
            models.append({
                "id":     mid,
                "label":  name,
                "free":   is_free,
                "vision": has_vision,
                "notes":  notes,
                "ctx":    ctx,
            })

        models.sort(key=lambda x: (not x["free"], not x["vision"], x["label"].lower()))
        return models if models else _FALLBACK_MODELS()

    except Exception:
        return _FALLBACK_MODELS()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def encode_image_to_base64(uploaded_file) -> tuple[str, str]:
    """Return (base64_string, mime_type) for an uploaded file."""
    raw       = uploaded_file.read()
    b64       = base64.b64encode(raw).decode("utf-8")
    mime_type = uploaded_file.type or "image/jpeg"
    return b64, mime_type


def build_image_content_part(b64: str, mime_type: str) -> dict:
    """OpenRouter / OpenAI vision message part."""
    return {
        "type":      "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def is_vision_model(model_id: str, all_models: list[dict]) -> bool:
    m = next((x for x in all_models if x["id"] == model_id), None)
    if m:
        return m.get("vision", False)
    # Keyword fallback
    return any(kw in model_id.lower() for kw in VISION_MODEL_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE BROWSER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def list_thesis_documents() -> list[str]:
    """Return sorted list of *_pdf folders inside data/thesis_dataset/."""
    if not THESIS_DATASET_DIR.exists():
        return []
    return sorted([p.name for p in THESIS_DATASET_DIR.iterdir() if p.is_dir()])


def list_thesis_images(doc_folder: str) -> list[Path]:
    """Return all images inside data/thesis_dataset/<doc_folder>/images/."""
    img_dir = THESIS_DATASET_DIR / doc_folder / "images"
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def list_general_image_folders() -> list[str]:
    """Return subfolders inside data/images/ (e.g. 'Thesis Diagram')."""
    if not IMAGES_DIR.exists():
        return []
    return sorted([p.name for p in IMAGES_DIR.iterdir() if p.is_dir()])


def list_general_images(subfolder: str) -> list[Path]:
    """Return all images inside data/images/<subfolder>/."""
    folder = IMAGES_DIR / subfolder
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def load_ocr_page_for_image(doc_folder: str, img_name: str) -> str | None:
    """Find the markdown page that references this image name."""
    pages_dir = THESIS_DATASET_DIR / doc_folder / "pages"
    if not pages_dir.exists():
        return None
    for md_file in sorted(pages_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        if img_name in content:
            return content
    return None


def image_path_to_base64(img_path: Path) -> tuple[str, str]:
    """Read image from disk and return (base64_string, mime_type)."""
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
    }
    mime = mime_map.get(img_path.suffix.lower(), "image/jpeg")
    b64  = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return b64, mime


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _session_path(session_id: str) -> Path:
    return CHAT_HISTORY_DIR / f"{session_id}.json"


def save_conversation(session_id: str, messages: list[dict], metadata: dict) -> None:
    # Strip raw image bytes before saving — keep only the text representation
    safe_messages = []
    for msg in messages:
        safe_msg = {k: v for k, v in msg.items() if k != "_image_parts"}
        safe_messages.append(safe_msg)
    data = {
        "session_id": session_id,
        "metadata":   metadata,
        "updated_at": datetime.now().isoformat(),
        "messages":   safe_messages,
    }
    _session_path(session_id).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_conversation(session_id: str) -> dict | None:
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_conversations() -> list[dict]:
    sessions = []
    for fp in sorted(CHAT_HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", fp.stem),
                "title":      data.get("metadata", {}).get("title", fp.stem),
                "model":      data.get("metadata", {}).get("model", ""),
                "updated_at": data.get("updated_at", ""),
                "msg_count":  len(data.get("messages", [])),
            })
        except Exception:
            continue
    return sessions


def delete_conversation(session_id: str) -> None:
    p = _session_path(session_id)
    if p.exists():
        p.unlink()


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def derive_title(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            content = m["content"]
            # content may be a list (multimodal) or a string
            text = content if isinstance(content, str) else next(
                (p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"), ""
            )
            text = text.strip().replace("\n", " ")
            return text[:60] + "…" if len(text) > 60 else text
    return "Untitled conversation"


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

def load_json_file(filepath: Path) -> dict | list | None:
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_html_file(filepath: Path) -> str:
    try:
        soup = BeautifulSoup(filepath.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def flatten_json(obj, prefix="") -> str:
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(flatten_json(v, f"{prefix}{k} > " if prefix else f"{k} > "))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.append(flatten_json(v, f"{prefix}[{i}] "))
    else:
        lines.append(f"{prefix.rstrip(' > ')}: {obj}")
    return "\n".join(lines)


def extract_text_from_file(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    if ext == ".json":
        data = load_json_file(filepath)
        return flatten_json(data) if data is not None else ""
    elif ext in (".html", ".htm"):
        return load_html_file(filepath)
    elif ext == ".txt":
        try:
            return filepath.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def friendly_label(filepath: Path) -> str:
    return str(filepath.relative_to(DATA_DIR))


@st.cache_data(show_spinner="Loading knowledge base…")
def load_all_documents() -> list[dict]:
    docs, seen = [], set()
    for pattern in ["**/*.json", "**/*.html", "**/*.htm", "**/*.txt"]:
        for fp in sorted(DATA_DIR.glob(pattern)):
            if CHAT_HISTORY_DIR in fp.parents:
                continue
            if fp in seen:
                continue
            seen.add(fp)
            text = extract_text_from_file(fp).strip()
            if not text:
                continue
            rel_parts = fp.relative_to(DATA_DIR).parts
            category  = rel_parts[0] if len(rel_parts) > 1 else "general"
            docs.append({"label": friendly_label(fp), "path": str(fp), "text": text, "category": category})
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def simple_keyword_score(query: str, text: str) -> float:
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return 0.0
    text_lower = text.lower()
    return sum(text_lower.count(t) for t in tokens) / len(tokens)


def retrieve_top_docs(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    scored = [(simple_keyword_score(query, d["text"]), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k] if score > 0]


def build_context(retrieved: list[dict]) -> str:
    parts, budget = [], MAX_CONTEXT_CHARS
    for d in retrieved:
        snippet = d["text"][:budget]
        parts.append(f"--- Reference: {d['label']} ---\n{snippet}")
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# OPENROUTER CALL  (multimodal-aware)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """\
You are Pear, a helpful education advisor assistant.
You answer questions about universities, scholarships, programs, and study abroad opportunities.

You have access to a knowledge base. Relevant excerpts are provided below.

INSTRUCTIONS:
1. Base your answer primarily on the provided references.
2. You MUST cite EVERY reference block you use. Use this exact inline format: [REF: <label>]
   where <label> is copied EXACTLY from the "Reference: <label>" header of that block.
   Example: [REF: output/page_1_uow.json]
3. Place the citation immediately after the sentence that uses the information.
4. If images are provided, analyze them and incorporate findings into your answer.
5. If the references do not contain enough information, say so honestly.
6. Be concise, friendly, and structured (use bullet points where helpful).

KNOWLEDGE BASE:
{context}
"""


def build_user_message_content(
    text: str,
    image_parts: list[dict] | None = None,
) -> str | list:
    """
    Returns a plain string for text-only models,
    or a list of content parts for vision models.
    """
    if not image_parts:
        return text
    parts = [{"type": "text", "text": text}]
    parts.extend(image_parts)
    return parts


def call_openrouter(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float = 0.3,
) -> tuple[str, list[str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://pear-edtech.app",
        "X-Title":       "Pear EdTech Chatbot",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp    = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"]
    cited = re.findall(r"\[REF:\s*(.+?)\]", reply)
    return reply, cited


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def render_message_with_citations(content: str, docs_index: dict):
    parts = re.split(r"(\[REF:\s*.+?\])", content)
    for part in parts:
        m = re.match(r"\[REF:\s*(.+?)\]", part)
        if m:
            label = m.group(1).strip()
            doc   = docs_index.get(label)
            with st.expander(f"📄 {label}", expanded=False):
                if doc:
                    st.caption(f"**Category:** {doc['category']}")
                    st.code(doc["text"][:CHUNK_PREVIEW_LEN] + "…", language=None)
                else:
                    st.write("_Reference not found in loaded documents._")
        else:
            if part.strip():
                st.markdown(part)


def render_user_message(msg: dict):
    """Render user message — show text + thumbnail(s) if images were attached."""
    content = msg["content"]
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                st.markdown(part["text"])
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    # base64 embedded image — display thumbnail
                    st.image(url, width=220, caption="📎 Attached image")
    else:
        st.markdown(content)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "api_key":          "",
    "messages":         [],
    "session_id":       new_session_id(),
    "active_model_id":  "meta-llama/llama-3.1-8b-instruct:free",
    "pending_images":   [],   # list of {"b64": str, "mime": str, "name": str}
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🤖 Pear AI Chatbot")
    st.caption("Ask anything about universities, scholarships, and study abroad — text & image supported.")

    # ── Load models ────────────────────────────────────────────────────────────
    with st.spinner("🔄 Loading models from OpenRouter…"):
        all_models  = fetch_openrouter_models()

    free_models    = [m for m in all_models if     m["free"]]
    paid_models    = [m for m in all_models if not m["free"]]
    vision_models  = [m for m in all_models if     m.get("vision")]
    id_to_model    = {m["id"]: m for m in all_models}

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:

        # ── API Key ────────────────────────────────────────────────────────────
        st.header("🔑 API Key")
        api_key_input = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Get your key at https://openrouter.ai/keys",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input

        effective_key = _get_api_key()
        # ❌ WRONG — ternary causes st.write() to receive the DeltaGenerator return value
        # st.success("✅ API key set") if effective_key else st.error("❌ API key missing")

        # ✅ CORRECT — plain if/else
        if effective_key:
            st.success("✅ API key set")
        else:
            st.error("❌ API key missing")

        if st.button("🔄 Refresh Model List", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(
            f"**{len(all_models)}** models · {len(free_models)} 🆓 free · "
            f"{len(paid_models)} 💳 paid · {len(vision_models)} 👁 vision"
        )

        st.divider()

        # ── Model selector ─────────────────────────────────────────────────────
        st.header("🤖 Model")

        tier = st.radio("Show:", ["🆓 Free Only", "💳 Paid Only", "👁 Vision Only", "🔀 All"], horizontal=True)
        visible = (
            free_models   if tier == "🆓 Free Only"   else
            paid_models   if tier == "💳 Paid Only"   else
            vision_models if tier == "👁 Vision Only" else
            all_models
        )

        search = st.text_input("🔍 Search", placeholder="llama, claude, gemini…")
        if search.strip():
            visible = [m for m in visible if search.lower() in m["label"].lower() or search.lower() in m["id"].lower()]

        visible_labels = [m["label"] for m in visible]
        current_label  = id_to_model.get(st.session_state.active_model_id, {}).get("label", "")
        default_idx    = visible_labels.index(current_label) if current_label in visible_labels else 0

        selected_label = st.selectbox(
            f"Select model ({len(visible)} shown)",
            options=visible_labels,
            index=default_idx,
            key="model_selectbox",
        )

        selected_model = next((m for m in all_models if m["label"] == selected_label), None)
        if selected_model:
            st.session_state.active_model_id = selected_model["id"]
            tier_badge    = "🆓 Free" if selected_model["free"] else "💳 Paid"
            vision_badge  = " · 👁 Vision" if selected_model.get("vision") else " · 📝 Text only"
            st.caption(f"{tier_badge}{vision_badge} · {selected_model['notes']}\n\n`{selected_model['id']}`")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        top_k       = st.slider("Top K references", 1, 10, 5)

        st.divider()

        # ── Knowledge base ─────────────────────────────────────────────────────
        st.header("📂 Knowledge Base")
        docs = load_all_documents()

        # Debug / user help: show resolved DATA_DIR and top-level contents so it's
        # obvious why no documents were loaded (common cause: wrong path or empty folder).
        try:
            top_level = sorted([p.name for p in DATA_DIR.iterdir()]) if DATA_DIR.exists() else []
        except Exception as _e:
            top_level = [f"error: {_e}"]

        st.caption(f"Resolved DATA_DIR: `{DATA_DIR}` · exists={DATA_DIR.exists()}")
        if top_level:
            st.caption("Top-level entries: " + (", ".join(top_level[:10]) + ("…" if len(top_level) > 10 else "")))
        else:
            st.caption("DATA_DIR is empty or not readable.")

        all_cats = sorted({d["category"] for d in docs})
        selected_cats = st.multiselect("Filter categories", all_cats, default=all_cats)
        filtered_docs = [d for d in docs if d["category"] in selected_cats]
        st.metric("Documents loaded", len(filtered_docs))

        # If no documents were found, show actionable guidance
        if not docs:
            st.warning(
                "No documents were found in DATA_DIR. Make sure you have .json/.html/.txt files under:\n\n"
                f"`{DATA_DIR}`\n\n"
                "Example (Mac):\n"
                "1) create sample: `mkdir -p data && echo \"Example content\" > data/example.txt`\n"
                "2) then refresh the app (Refresh Model List -> Refresh cache).",
                icon="⚠️",
            )
         with st.expander("Browse documents"):
             for d in filtered_docs[:50]:
                 st.text(f"• {d['label']}")
             if len(filtered_docs) > 50:
                 st.caption(f"…and {len(filtered_docs) - 50} more")

        st.divider()

        # ── Conversation history ───────────────────────────────────────────────
        st.header("💬 Conversations")

        if st.button("➕ New Conversation", use_container_width=True):
            if st.session_state.messages:
                save_conversation(
                    st.session_state.session_id,
                    st.session_state.messages,
                    {"title": derive_title(st.session_state.messages), "model": st.session_state.active_model_id},
                )
            st.session_state.messages    = []
            st.session_state.session_id  = new_session_id()
            st.session_state.pending_images = []
            st.rerun()

        saved = list_conversations()
        if saved:
            with st.expander(f"📁 Saved ({len(saved)})", expanded=True):
                for s in saved:
                    col_a, col_b = st.columns([5, 1])
                    is_active    = s["session_id"] == st.session_state.session_id
                    lbl          = f"{'▶ ' if is_active else ''}{s['title']}"
                    col_a.caption(f"**{lbl}**\n\n_{s['msg_count']} msgs · {s['updated_at'][:16]}_")
                    if col_a.button("Load", key=f"load_{s['session_id']}", use_container_width=True):
                        if st.session_state.messages:
                            save_conversation(
                                st.session_state.session_id,
                                st.session_state.messages,
                                {"title": derive_title(st.session_state.messages), "model": st.session_state.active_model_id},
                            )
                        conv = load_conversation(s["session_id"])
                        if conv:
                            st.session_state.messages   = conv["messages"]
                            st.session_state.session_id = s["session_id"]
                            if conv.get("metadata", {}).get("model"):
                                st.session_state.active_model_id = conv["metadata"]["model"]
                        st.session_state.pending_images = []
                        st.rerun()
                    if col_b.button("🗑", key=f"del_{s['session_id']}"):
                        delete_conversation(s["session_id"])
                        if s["session_id"] == st.session_state.session_id:
                            st.session_state.messages   = []
                            st.session_state.session_id = new_session_id()
                        st.rerun()
        else:
            st.caption("No saved conversations yet.")

    # ── Chat area ──────────────────────────────────────────────────────────────
    docs_index = {d["label"]: d for d in filtered_docs}
    active_m   = id_to_model.get(st.session_state.active_model_id)

    # Active model banner
    if active_m:
        tier_icon   = "🆓" if active_m["free"] else "💳"
        vision_icon = " · 👁 Vision" if active_m.get("vision") else " · 📝 Text only"
        st.caption(f"{tier_icon} **{active_m['label']}**{vision_icon} · `{active_m['id']}` · {active_m['notes']}")

    # ADD THIS LINE ────────────────────────────────────────────────────────────
    model_has_vision = is_vision_model(st.session_state.active_model_id, all_models)

    # ── Image uploader section ─────────────────────────────────────────────────
    if model_has_vision:
        with st.expander("📎 Attach images to next message", expanded=bool(st.session_state.pending_images)):

            tab_upload, tab_thesis, tab_general = st.tabs([
                "⬆️ Upload",
                "📄 Thesis Dataset",
                "🖼 General Images",
            ])

            # ── Tab 1: Upload ──────────────────────────────────────────────────
            with tab_upload:
                uploaded_files = st.file_uploader(
                    "Upload images (JPEG, PNG, GIF, WEBP)",
                    type=["jpg", "jpeg", "png", "gif", "webp"],
                    accept_multiple_files=True,
                    key="image_uploader",
                )
                if uploaded_files:
                    st.session_state.pending_images = []
                    for uf in uploaded_files:
                        b64, mime = encode_image_to_base64(uf)
                        st.session_state.pending_images.append({"b64": b64, "mime": mime, "name": uf.name})

            # ── Tab 2: Thesis Dataset ──────────────────────────────────────────
            with tab_thesis:
                thesis_docs = list_thesis_documents()

                if not thesis_docs:
                    st.info("No thesis dataset found. Run Bulk OCR first.", icon="ℹ️")
                else:
                    selected_doc = st.selectbox(
                        "📁 Select thesis document",
                        thesis_docs,
                        format_func=lambda x: x.replace("_pdf", "").replace("_", " "),
                        key="thesis_doc_selector",
                    )

                    thesis_images = list_thesis_images(selected_doc)

                    if not thesis_images:
                        st.warning(f"No images found in `{selected_doc}/images/`")
                    else:
                        st.caption(f"📂 `data/thesis_dataset/{selected_doc}/images/` — {len(thesis_images)} image(s)")

                        img_names = [p.name for p in thesis_images]
                        selected_names = st.multiselect(
                            "Select image(s) to attach",
                            img_names,
                            key="thesis_img_multiselect",
                        )

                        if selected_names:
                            cols = st.columns(min(len(selected_names), 4))
                            for i, name in enumerate(selected_names):
                                img_path = THESIS_DATASET_DIR / selected_doc / "images" / name
                                cols[i % 4].image(str(img_path), caption=name, width=150)

                            # ✅ REPLACED inner expander with toggle + container
                            show_ocr = st.checkbox("📄 Show associated OCR page text", value=False, key="show_ocr_toggle")
                            if show_ocr:
                                for name in selected_names:
                                    page_text = load_ocr_page_for_image(selected_doc, name)
                                    if page_text:
                                        st.markdown(f"**Page referencing `{name}`:**")
                                        st.text_area("", value=page_text, height=200, key=f"thesis_page_{name}")
                                    else:
                                        st.caption(f"`{name}` — no matching OCR page found.")

                        if st.button("✅ Attach thesis image(s)", key="attach_thesis", use_container_width=True):
                            if not selected_names:
                                st.warning("Please select at least one image.")
                            else:
                                st.session_state.pending_images = []
                                for name in selected_names:
                                    img_path = THESIS_DATASET_DIR / selected_doc / "images" / name
                                    b64, mime = image_path_to_base64(img_path)
                                    st.session_state.pending_images.append({
                                        "b64": b64, "mime": mime,
                                        "name": f"{selected_doc}/{name}",
                                    })
                                st.success(f"✅ {len(selected_names)} image(s) attached!")
                                st.rerun()

            # ── Tab 3: General Images ──────────────────────────────────────────
            with tab_general:
                img_folders = list_general_image_folders()

                if not img_folders:
                    st.info("No folders found in `data/images/`.", icon="ℹ️")
                else:
                    selected_folder = st.selectbox(
                        "📁 Select image folder",
                        img_folders,
                        key="general_img_folder_selector",
                    )

                    general_images = list_general_images(selected_folder)

                    if not general_images:
                        st.warning(f"No images found in `data/images/{selected_folder}/`")
                    else:
                        st.caption(f"📂 `data/images/{selected_folder}/` — {len(general_images)} image(s)")

                        img_names = [p.name for p in general_images]
                        selected_names = st.multiselect(
                            "Select image(s) to attach",
                            img_names,
                            key="general_img_multiselect",
                        )

                        if selected_names:
                            cols = st.columns(min(len(selected_names), 4))
                            for i, name in enumerate(selected_names):
                                img_path = IMAGES_DIR / selected_folder / name
                                cols[i % 4].image(str(img_path), caption=name, width=150)

                        if st.button("✅ Attach general image(s)", key="attach_general", use_container_width=True):
                            if not selected_names:
                                st.warning("Please select at least one image.")
                            else:
                                st.session_state.pending_images = []
                                for name in selected_names:
                                    img_path = IMAGES_DIR / selected_folder / name
                                    b64, mime = image_path_to_base64(img_path)
                                    st.session_state.pending_images.append({
                                        "b64": b64, "mime": mime,
                                        "name": f"{selected_folder}/{name}",
                                    })
                                st.success(f"✅ {len(selected_names)} image(s) attached!")
                                st.rerun()

            # ── Pending images preview (shared) ───────────────────────────────
            if st.session_state.pending_images:
                st.divider()
                st.caption(f"✅ {len(st.session_state.pending_images)} image(s) ready to send:")
                img_cols = st.columns(min(len(st.session_state.pending_images), 4))
                for i, img in enumerate(st.session_state.pending_images):
                    img_cols[i % 4].image(
                        f"data:{img['mime']};base64,{img['b64']}",
                        caption=img["name"],
                        width=120,
                    )
                if st.button("🗑 Clear images", key="clear_images"):
                    st.session_state.pending_images = []
                    st.rerun()

    else:
        if st.session_state.pending_images:
            st.session_state.pending_images = []
        st.info("💡 Select a **👁 Vision** model in the sidebar to attach images.", icon="ℹ️")

    st.divider()

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_message_with_citations(msg["content"], docs_index)

                cited_refs     = msg.get("references_cited",    msg.get("references", []))
                retrieved_refs = msg.get("references_retrieved", [])

                if retrieved_refs:
                    with st.expander(f"🔍 {len(retrieved_refs)} reference(s) used as context", expanded=False):
                        for lbl in retrieved_refs:
                            doc         = docs_index.get(lbl)
                            cited_badge = "✅ cited" if lbl in cited_refs else "📄 retrieved"
                            st.markdown(f"**{cited_badge} · {lbl}**")
                            if doc:
                                st.code(doc["text"][:CHUNK_PREVIEW_LEN] + "…", language=None)

                if cited_refs:
                    st.caption(f"📎 LLM cited: {', '.join(cited_refs)}")
                elif retrieved_refs:
                    st.caption("⚠️ LLM did not emit explicit citations — see retrieved references above.")

                if msg.get("model_id"):
                    m_info  = id_to_model.get(msg["model_id"])
                    m_label = m_info["label"] if m_info else msg["model_id"]
                    vision_tag = " · 👁 multimodal" if msg.get("had_images") else ""
                    st.caption(f"🤖 `{m_label}`{vision_tag} · ⏱ {msg.get('elapsed_s', '?')}s")
            else:
                render_user_message(msg)

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about universities, scholarships, programs…"):
        if not effective_key:
            st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
            st.stop()
        if not filtered_docs:
            st.warning("⚠️ No documents loaded.")
            st.stop()

        # Snapshot & clear pending images
        pending = list(st.session_state.pending_images)
        st.session_state.pending_images = []

        # Warn if images attached to non-vision model
        if pending and not model_has_vision:
            st.warning("⚠️ Images ignored — selected model does not support vision.")
            pending = []

        # Build image content parts
        image_parts = [build_image_content_part(img["b64"], img["mime"]) for img in pending]

        # Build user message content (multimodal or plain text)
        user_content = build_user_message_content(prompt, image_parts if image_parts else None)

        user_msg = {
            "role":       "user",
            "content":    user_content,
            "had_images": bool(image_parts),
            "image_meta": [{"name": img["name"], "mime": img["mime"]} for img in pending],
        }
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            render_user_message(user_msg)

        # RAG retrieval
        retrieved        = retrieve_top_docs(prompt, filtered_docs, top_k=top_k)
        context          = build_context(retrieved)
        retrieved_labels = [d["label"] for d in retrieved]

        # Build API messages
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        api_messages  = [{"role": "system", "content": system_prompt}]

        # History: last 10 turns — serialize multimodal content correctly
        for m in st.session_state.messages[-10:]:
            api_messages.append({"role": m["role"], "content": m["content"]})

        # Call LLM
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {active_m['label'] if active_m else 'model'}…"):
                t0 = time.time()
                try:
                    reply, cited = call_openrouter(
                        api_messages,
                        model=st.session_state.active_model_id,
                        api_key=effective_key,
                        temperature=temperature,
                    )
                    elapsed = round(time.time() - t0, 1)
                except requests.HTTPError as e:
                    st.error(f"API error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.stop()

            render_message_with_citations(reply, docs_index)

            # Show image thumbnails analysed
            if pending:
                with st.expander(f"🖼 {len(pending)} image(s) analysed", expanded=False):
                    img_cols = st.columns(min(len(pending), 4))
                    for i, img in enumerate(pending):
                        img_cols[i % 4].image(
                            f"data:{img['mime']};base64,{img['b64']}",
                            caption=img["name"],
                            width=150,
                        )

            # Retrieved references
            with st.expander(f"🔍 Retrieved {len(retrieved)} reference(s) from knowledge base", expanded=False):
                for d in retrieved:
                    badge = "✅ cited" if d["label"] in cited else "📄 retrieved"
                    st.markdown(f"**{badge} · {d['label']}** _(category: {d['category']})_")
                    st.code(d["text"][:CHUNK_PREVIEW_LEN] + "…", language=None)

            if cited:
                st.caption(f"📎 LLM cited: {', '.join(cited)}")
            else:
                st.caption("⚠️ LLM did not emit explicit citations — see retrieved references above.")

            vision_tag = " · 👁 multimodal" if pending else ""
            if active_m:
                st.caption(f"🤖 `{active_m['label']}`{vision_tag} · ⏱ {elapsed}s")

        # Save assistant message
        assistant_msg = {
            "role":                  "assistant",
            "content":               reply,
            "references_cited":      cited,
            "references_retrieved":  retrieved_labels,
            "references":            list(dict.fromkeys(cited + retrieved_labels)),
            "model_id":              st.session_state.active_model_id,
            "elapsed_s":             elapsed,
            "had_images":            bool(pending),
            "timestamp":             datetime.now().isoformat(),
        }
        st.session_state.messages.append(assistant_msg)

        save_conversation(
            st.session_state.session_id,
            st.session_state.messages,
            {"title": derive_title(st.session_state.messages), "model": st.session_state.active_model_id},
        )


if __name__ == "__main__":
    main()