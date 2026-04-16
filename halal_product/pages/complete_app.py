import base64
import io
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv9 Product Detection", layout="wide")
st.title("YOLOv9 Product Detection + Vision LM Lookup")

# ── Result storage directories ─────────────────────────────────────────────────
RESULTS_DIR       = Path("results")
SUCCESSFUL_DIR    = RESULTS_DIR / "successful"
UNSUCCESSFUL_DIR  = RESULTS_DIR / "unsuccessful"
for _d in (SUCCESSFUL_DIR, UNSUCCESSFUL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL_KEYWORDS = [
    "gpt-4o", "gpt-4-vision", "claude-3", "claude-3.5",
    "gemini", "llava", "vision", "pixtral", "qwen-vl",
    "intern-vl", "minicpm-v", "phi-3-vision",
]

# ══════════════════════════════════════════════════════════════════════════════
# RESULT STORAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_result(
    original_image: Image.Image,
    annotated_array: np.ndarray,
    enrichment_rows: list[dict],
    source_name: str,
    successful: bool,
) -> Path:
    """
    Save original + annotated image and a metadata JSON into
    results/successful/ or results/unsuccessful/.
    Returns the folder where files were saved.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name  = re.sub(r"[^\w\-.]", "_", source_name)
    folder     = (SUCCESSFUL_DIR if successful else UNSUCCESSFUL_DIR) / f"{timestamp}_{safe_name}"
    folder.mkdir(parents=True, exist_ok=True)

    # Save original
    original_image.save(folder / "original.jpg", format="JPEG")

    # Save annotated
    annotated_pil = Image.fromarray(annotated_array)
    annotated_pil.save(folder / "annotated.jpg", format="JPEG")

    # Save each crop
    for row in enrichment_rows:
        crop: Image.Image = row.get("crop")
        if crop:
            crop.save(folder / f"crop_{row['index']:03d}_{row['yolo_class']}.jpg", format="JPEG")

    # Save metadata JSON (exclude PIL crops — not JSON-serialisable)
    meta = {
        "source":        source_name,
        "timestamp":     timestamp,
        "successful":    successful,
        "total_detections": sum(1 for r in enrichment_rows if r.get("product_name")),
        "enrichments": [
            {k: v for k, v in row.items() if k != "crop"}
            for row in enrichment_rows
        ],
    }
    (folder / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return folder


def _list_result_folders(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir()],
        reverse=True,
    )


def _delete_result_folder(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)

# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    try:
        from config.settings import settings
        for attr in ("OPENROUTER_API_KEY", "openrouter_api_key", "api_key"):
            val = getattr(settings, attr, None)
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return st.session_state.get("openrouter_api_key", "").strip() or os.getenv("OPENROUTER_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# NMS / DRAW HELPERS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter   = inter_w * inter_h
    area_a  = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
    area_b  = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) * np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
    union   = np.maximum(area_a + area_b - inter, 1e-9)
    return inter / union


def _classwise_nms(boxes, scores, classes, iou_thr, max_det):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    keep = []
    for cls in np.unique(classes):
        cls_idx    = np.where(classes == cls)[0]
        order      = cls_idx[np.argsort(-scores[cls_idx])]
        while len(order) > 0:
            cur = order[0]
            keep.append(cur)
            if len(order) == 1:
                break
            rest = order[1:]
            order = rest[_compute_iou(boxes[cur], boxes[rest]) < iou_thr]
    keep = np.array(keep, dtype=np.int64)
    keep = keep[np.argsort(-scores[keep])]
    return keep[:max_det]


def _draw_detections(image_rgb, boxes, scores, classes, names):
    canvas = image_rgb.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id  = int(classes[i])
        score   = float(scores[i])
        color   = (int((37*(cls_id+1))%255), int((17*(cls_id+7))%255), int((29*(cls_id+13))%255))
        label   = f"{names.get(cls_id, str(cls_id))} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(y1 - 8, th + 4)
        cv2.rectangle(canvas, (x1, y_text-th-6), (x1+tw+6, y_text+2), color, -1)
        cv2.putText(canvas, label, (x1+3, y_text-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    return canvas

# ══════════════════════════════════════════════════════════════════════════════
# YOLO INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _predict_full_image(model, img_array, conf, iou, imgsz, max_det, augment):
    result = model.predict(source=img_array, conf=conf, iou=iou, imgsz=imgsz,
                           max_det=max_det, augment=augment, verbose=False)[0]
    if len(result.boxes) == 0:
        return np.empty((0,4),np.float32), np.empty((0,),np.float32), np.empty((0,),np.int32)
    return (result.boxes.xyxy.cpu().numpy().astype(np.float32),
            result.boxes.conf.cpu().numpy().astype(np.float32),
            result.boxes.cls.cpu().numpy().astype(np.int32))


def _predict_sliced(model, img_array, conf, iou, imgsz, max_det, augment, tile_size, overlap):
    h, w   = img_array.shape[:2]
    stride = max(32, int(tile_size*(1.0-overlap)))
    xs = list(range(0, max(1, w-tile_size+1), stride))
    ys = list(range(0, max(1, h-tile_size+1), stride))
    if not xs or xs[-1] != max(0, w-tile_size): xs.append(max(0, w-tile_size))
    if not ys or ys[-1] != max(0, h-tile_size): ys.append(max(0, h-tile_size))
    all_b, all_s, all_c = [], [], []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(w, x0+tile_size), min(h, y0+tile_size)
            tr = model.predict(source=img_array[y0:y1, x0:x1], conf=conf, iou=iou,
                               imgsz=imgsz, max_det=max(200, max_det//4),
                               augment=augment, verbose=False)[0]
            if len(tr.boxes) == 0: continue
            b = tr.boxes.xyxy.cpu().numpy().astype(np.float32)
            b[:, [0,2]] += x0;  b[:, [1,3]] += y0
            all_b.append(b)
            all_s.append(tr.boxes.conf.cpu().numpy().astype(np.float32))
            all_c.append(tr.boxes.cls.cpu().numpy().astype(np.int32))
    if not all_b:
        return np.empty((0,4),np.float32), np.empty((0,),np.float32), np.empty((0,),np.int32)
    boxes   = np.concatenate(all_b)
    scores  = np.concatenate(all_s)
    classes = np.concatenate(all_c)
    keep    = _classwise_nms(boxes, scores, classes, iou, max_det)
    return boxes[keep], scores[keep], classes[keep]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY & LOAD
# ══════════════════════════════════════════════════════════════════════════════

def _discover_model_paths():
    candidates = []
    for pattern in ["*.pt", "models/*.pt"]:
        for p in Path(".").glob(pattern):
            candidates.append(str(p))
    if not candidates:
        return ["yolov9c.pt", "yolov8n.pt"]
    preferred = ["yolov9c.pt", "yolov8n.pt"]
    ordered   = [m for m in preferred if m in candidates]
    ordered  += sorted([m for m in candidates if m not in ordered])
    return ordered


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ══════════════════════════════════════════════════════════════════════════════
# OPENROUTER / VISION MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_models() -> list[dict]:
    return [
        {"id": "openai/gpt-4o-mini",              "label": "GPT-4o Mini",        "vision": True, "free": False, "notes": "$0.150/1M"},
        {"id": "openai/gpt-4o",                   "label": "GPT-4o",             "vision": True, "free": False, "notes": "$2.500/1M"},
        {"id": "anthropic/claude-3.5-sonnet",     "label": "Claude 3.5 Sonnet",  "vision": True, "free": False, "notes": "$3.000/1M"},
        {"id": "google/gemini-flash-1.5",         "label": "Gemini 1.5 Flash",   "vision": True, "free": False, "notes": "$0.075/1M"},
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openrouter_models(api_key: str) -> list[dict]:
    if not api_key:
        return _fallback_models()
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://pear-edtech.app"},
            timeout=10,
        )
        resp.raise_for_status()
        raw    = resp.json().get("data", [])
        models = []
        for m in raw:
            mid       = m.get("id", "")
            name      = m.get("name", mid)
            pricing   = m.get("pricing", {})
            arch      = m.get("architecture", {})
            modality  = arch.get("modality", "")
            in_mods   = arch.get("input_modalities", [])
            has_vision = (
                "image" in modality or "image" in in_mods or "multimodal" in modality
                or any(kw in mid.lower() for kw in VISION_MODEL_KEYWORDS)
                or any(kw in name.lower() for kw in VISION_MODEL_KEYWORDS)
            )
            if not has_vision:
                continue
            try:
                is_free = float(pricing.get("prompt",1))==0.0 and float(pricing.get("completion",1))==0.0
            except (TypeError, ValueError):
                is_free = False
            models.append({"id": mid, "label": name, "vision": has_vision,
                           "free": is_free, "notes": "free" if is_free else "paid"})
        models.sort(key=lambda x: (not x["free"], x["label"].lower()))
        return models or _fallback_models()
    except Exception:
        return _fallback_models()


def build_image_content_part(b64: str, mime_type: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}


def build_user_message_content(text: str, image_parts: list[dict] | None = None):
    if not image_parts:
        return text
    return [{"type": "text", "text": text}] + image_parts


def call_openrouter(messages, model, api_key, temperature=0.1) -> str:
    """
    Call OpenRouter with automatic retry on 429 (rate-limit) and
    clear error messages on 404 (model not found).
    """
    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://pear-edtech.app",
        "X-Title":        "Pear EdTech Chatbot",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}

    max_retries  = 3
    base_delay   = 2.0   # seconds

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_API_URL, headers=headers,
                json=payload, timeout=90,
            )

            if resp.status_code == 429:
                # Respect Retry-After header if present, else exponential back-off
                retry_after = float(resp.headers.get("Retry-After", base_delay * (2 ** (attempt - 1))))
                if attempt < max_retries:
                    time.sleep(retry_after)
                    continue
                else:
                    resp.raise_for_status()   # raise after final attempt

            if resp.status_code == 404:
                raise ValueError(
                    f"Model '{model}' not found on OpenRouter (404). "
                    "Please select a different model in the sidebar."
                )

            resp.raise_for_status()

            # ── Parse response safely ─────────────────────────────────────────
            data    = resp.json()
            choice  = data["choices"][0]
            content = choice["message"]["content"]

            # content can be a str or a list of parts — normalise here too
            if isinstance(content, list):
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                return " ".join(p for p in parts if p).strip()

            return str(content)

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            if attempt < max_retries:
                time.sleep(base_delay * attempt)
                continue
            raise exc

    raise RuntimeError("call_openrouter: exhausted all retries.")


def pil_to_base64(image: Image.Image, image_format="JPEG") -> tuple[str, str]:
    buf  = io.BytesIO()
    mime = "image/png" if image_format.upper()=="PNG" else "image/jpeg"
    image.save(buf, format="PNG" if mime=="image/png" else "JPEG")
    return base64.b64encode(buf.getvalue()).decode(), mime


def extract_product_crop(image: Image.Image, box: np.ndarray, padding_ratio=0.08) -> Image.Image:
    w, h    = image.size
    x1,y1,x2,y2 = box.tolist()
    px, py  = max(8, int((x2-x1)*padding_ratio)), max(8, int((y2-y1)*padding_ratio))
    return image.crop((max(0,int(x1)-px), max(0,int(y1)-py),
                       min(w,int(x2)+px), min(h,int(y2)+py)))


def extract_product_name(raw_reply) -> str:
    """
    Safely extract a plain product name string from the Vision LM reply.
    Handles: plain str, list of content parts (OpenAI/OpenRouter multipart),
    dict with 'text' key, or anything else.
    """
    # ── Normalise to plain string first ───────────────────────────────────────
    if isinstance(raw_reply, list):
        # OpenAI-style content parts: [{"type": "text", "text": "..."}, ...]
        parts = []
        for part in raw_reply:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "") or part.get("content", "")))
            else:
                parts.append(str(part))
        text = " ".join(p for p in parts if p).strip()
    elif isinstance(raw_reply, dict):
        text = str(raw_reply.get("text", "") or raw_reply.get("content", "") or raw_reply).strip()
    else:
        text = str(raw_reply).strip()

    if not text:
        return ""

    # ── Pull out product_name: <value> if present ─────────────────────────────
    match = re.search(r"product_name\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = re.sub(r"^[-*\s]+", "", text)
    return re.sub(r"[\r\n]+", " ", text).strip(" '\"`")


def identify_product_name(crop, model_id, api_key, yolo_label) -> tuple[str, str]:
    b64, mime = pil_to_base64(crop)
    prompt = (
        f"You are identifying a packaged food or drink product. YOLO label: '{yolo_label}'. "
        "Return only the most likely market-facing product name. "
        "Output format: product_name: <answer>"
    )
    messages = [{"role": "user", "content": build_user_message_content(
        prompt, [build_image_content_part(b64, mime)])}]

    # raw_reply is now guaranteed to be a plain str from call_openrouter
    raw_reply = call_openrouter(messages, model=model_id, api_key=api_key, temperature=0.0)
    return extract_product_name(raw_reply), raw_reply

# ══════════════════════════════════════════════════════════════════════════════
# OPENFOODFACTS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(value) -> str:
    return "" if value is None else str(value).strip().lower()


def score_openfoodfacts_match(item, query) -> int:
    tokens   = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
    name     = normalize_text(item.get("product_name"))
    brand    = normalize_text(item.get("brands"))
    combined = " ".join(p for p in [name, brand, normalize_text(item.get("categories"))] if p)
    if not combined: return -1
    score = 0
    if query.lower() == name:      score += 200
    if query.lower() in name:      score += 80
    if query.lower() in combined:  score += 40
    for t in tokens:
        score += 15 if t in name else (6 if t in combined else 0)
    return score


@st.cache_data(show_spinner=False, ttl=3600)
def search_openfoodfacts(query: str, max_results=5, max_scan=3000) -> list[dict]:
    if not query.strip(): return []
    dataset = load_dataset("openfoodfacts/product-database", split="food", streaming=True)
    matches = []
    for idx, item in enumerate(dataset):
        if idx >= max_scan: break
        score = score_openfoodfacts_match(item, query)
        if score <= 0: continue
        matches.append({
            "score": score,
            "product_name":     item.get("product_name") or "",
            "brands":           item.get("brands") or "",
            "categories":       item.get("categories") or "",
            "countries":        item.get("countries") or "",
            "nutriscore_grade": item.get("nutriscore_grade") or "",
            "image_small_url":  item.get("image_small_url") or "",
            "url":              item.get("url") or "",
            "code":             item.get("code") or "",
        })
    matches.sort(key=lambda x: (-x["score"], x["product_name"].lower()))
    return matches[:max_results]

# ══════════════════════════════════════════════════════════════════════════════
# LIVE WEBCAM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _list_available_cameras(max_test=5) -> list[int]:
    """Return indices of cameras that OpenCV can open."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available or [0]


def _capture_single_frame(camera_index: int) -> np.ndarray | None:
    """Grab one frame from the camera and return as RGB numpy array."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DETECTION PIPELINE  (reused for every input source)
# ══════════════════════════════════════════════════════════════════════════════

def run_detection_pipeline(
    image: Image.Image,
    source_name: str,
    model,
    conf_threshold, iou_threshold, img_size, max_detections, use_tta,
    use_sliced_inference, tile_size, tile_overlap,
    run_product_identification, effective_key, selected_vision_model,
    max_products_to_enrich, openfoodfacts_results, openfoodfacts_scan_limit,
    auto_save: bool = True,
):
    img_array = np.array(image)

    with st.spinner("Detecting objects…"):
        if use_sliced_inference:
            boxes, scores, classes = _predict_sliced(
                model, img_array, conf_threshold, iou_threshold,
                img_size, max_detections, use_tta, tile_size, tile_overlap)
        else:
            boxes, scores, classes = _predict_full_image(
                model, img_array, conf_threshold, iou_threshold,
                img_size, max_detections, use_tta)

    annotated = _draw_detections(img_array, boxes, scores, classes, model.names)
    col1, col2 = st.columns(2)
    col1.image(image,     caption="Input",    use_container_width=True)
    col2.image(annotated, caption="Detected", use_container_width=True)

    enrichment_rows: list[dict] = []
    successful = len(boxes) > 0

    if len(boxes) == 0:
        st.warning("No objects detected.")
    else:
        st.success(f"**{len(boxes)}** detection(s) found.")
        ordered_indices = list(np.argsort(-scores))

        with st.expander("Raw detections", expanded=False):
            for rank, i in enumerate(ordered_indices, 1):
                st.write(f"{rank}. **{model.names[int(classes[i])]}** — conf {scores[i]:.3f}")

        if run_product_identification:
            if not effective_key:
                st.warning("OpenRouter API key missing — product identification skipped.")
            elif not selected_vision_model:
                st.warning("No vision model selected.")
            else:
                # ── Guard: warn if model id looks invalid ─────────────────────
                model_id = selected_vision_model.get("id", "")
                if not model_id or "/" not in model_id:
                    st.error(
                        f"⚠️ Selected model id `{model_id!r}` looks invalid. "
                        "Pick a valid model from the sidebar (format: `provider/model-name`)."
                    )
                else:
                    st.subheader("Vision LM + OpenFoodFacts")
                    st.caption(f"Using model: `{model_id}`")
                    top_indices = ordered_indices[:min(max_products_to_enrich, len(ordered_indices))]
                    progress    = st.progress(0.0)

                    for step, idx in enumerate(top_indices, 1):
                        crop       = extract_product_crop(image, boxes[idx])
                        class_name = model.names[int(classes[idx])]
                        confidence = float(scores[idx])
                        try:
                            product_name, raw_reply = identify_product_name(
                                crop, model_id, effective_key, class_name)
                            matches = search_openfoodfacts(
                                product_name, max_results=openfoodfacts_results,
                                max_scan=openfoodfacts_scan_limit)
                            enrichment_rows.append({
                                "index": int(idx)+1, "yolo_class": class_name,
                                "confidence": round(confidence, 3),
                                "product_name": product_name,
                                "matches_found": len(matches),
                                "raw_reply": raw_reply,
                                "matches": matches,
                                "crop": crop,
                            })
                        except ValueError as exc:
                            # 404 model-not-found — stop enriching, show once
                            st.error(str(exc))
                            enrichment_rows.append({
                                "index": int(idx)+1, "yolo_class": class_name,
                                "confidence": round(confidence, 3),
                                "product_name": "", "matches_found": 0,
                                "raw_reply": f"Stopped: {exc}", "matches": [], "crop": crop,
                            })
                            break   # no point retrying other crops with invalid model
                        except Exception as exc:
                            enrichment_rows.append({
                                "index": int(idx)+1, "yolo_class": class_name,
                                "confidence": round(confidence, 3),
                                "product_name": "", "matches_found": 0,
                                "raw_reply": f"Error: {exc}", "matches": [], "crop": crop,
                            })
                        progress.progress(step / len(top_indices))

                    if enrichment_rows:
                        st.dataframe([{
                            "detection": r["index"], "yolo_class": r["yolo_class"],
                            "confidence": r["confidence"],
                            "vision_product_name": r["product_name"],
                            "openfoodfacts_matches": r["matches_found"],
                        } for r in enrichment_rows], use_container_width=True)

                        for row in enrichment_rows:
                            with st.expander(
                                f"Det {row['index']} | {row['yolo_class']} | {row['product_name'] or 'no name'}",
                                expanded=False):
                                st.image(row["crop"], caption=f"Crop {row['index']}", width=220)
                                st.caption(f"Confidence: {row['confidence']}")
                                st.code(row["raw_reply"], language=None)
                                if row["matches"]:
                                    st.dataframe(row["matches"], use_container_width=True)
                                else:
                                    st.write("No OpenFoodFacts match found.")

    # ── Auto-save ──────────────────────────────────────────────────────────────
    if auto_save:
        saved_folder = _save_result(
            original_image=image,
            annotated_array=annotated,
            enrichment_rows=enrichment_rows,
            source_name=source_name,
            successful=successful,
        )
        cluster = "✅ successful" if successful else "❌ unsuccessful"
        st.success(f"Result saved to `{saved_folder}` → cluster: **{cluster}**")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

available_models = _discover_model_paths()
selected_model_path = st.sidebar.selectbox(
    "YOLO weights", options=available_models, index=0,
    help="Use custom trained weights for best results.")

try:
    model = load_model(selected_model_path)
except Exception as e:
    st.error(f"Failed to load model '{selected_model_path}': {e}")
    st.stop()

st.sidebar.subheader("Detection settings")
conf_threshold    = st.sidebar.slider("Confidence threshold", 0.01, 0.90, 0.10, 0.01)
iou_threshold     = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, 0.50, 0.01)
img_size          = st.sidebar.select_slider("Inference image size", [640,960,1280,1536,1920], value=1536)
max_detections    = st.sidebar.slider("Max detections", 50, 3000, 1000, 50)
use_tta           = st.sidebar.checkbox("Test-time augmentation", True)
use_sliced_inference = st.sidebar.checkbox("Sliced inference for tiny objects", True)
tile_size         = st.sidebar.select_slider("Tile size", [512,640,768,896,1024], value=768,
                                              disabled=not use_sliced_inference)
tile_overlap      = st.sidebar.slider("Tile overlap", 0.10, 0.50, 0.30, 0.05,
                                      disabled=not use_sliced_inference)

st.sidebar.divider()

# ── API key + Vision model selector ───────────────────────────────────────────
st.sidebar.subheader("Vision LM + OpenFoodFacts")
api_key_input = st.sidebar.text_input(
    "OpenRouter API Key", type="password",
    value=st.session_state.get("openrouter_api_key", ""),
    help="https://openrouter.ai/keys")
if api_key_input:
    st.session_state["openrouter_api_key"] = api_key_input

effective_key        = _get_api_key()
all_openrouter_models = fetch_openrouter_models(effective_key)

if effective_key:
    st.sidebar.success("✅ API key set")
else:
    st.sidebar.error("❌ API key missing")

if st.sidebar.button("🔄 Refresh model list"):
    st.cache_data.clear()
    st.rerun()

# ── Tier / search filter ───────────────────────────────────────────────────────
free_models   = [m for m in all_openrouter_models if     m.get("free")]
paid_models   = [m for m in all_openrouter_models if not m.get("free")]
vision_models = [m for m in all_openrouter_models if     m.get("vision")]

tier = st.sidebar.radio(
    "Show models:",
    ["🆓 Free Only", "💳 Paid Only", "👁 Vision Only", "🔀 All"],
    index=3, horizontal=True,
)
visible_models = (
    free_models   if tier == "🆓 Free Only"   else
    paid_models   if tier == "💳 Paid Only"   else
    vision_models if tier == "👁 Vision Only" else
    all_openrouter_models
)

search_q = st.sidebar.text_input("🔍 Search models", placeholder="llama, gpt, gemini…")
if search_q.strip():
    q = search_q.strip().lower()
    visible_models = [m for m in visible_models
                      if q in m.get("label","").lower() or q in m.get("id","").lower()]

st.sidebar.caption(
    f"**{len(all_openrouter_models)}** total · {len(free_models)} 🆓 · "
    f"{len(paid_models)} 💳 · {len(vision_models)} 👁"
)

vision_model_labels = [f"{m['label']}  ({m['id']})" for m in visible_models]
selected_vision_label = st.sidebar.selectbox(
    f"Vision model ({len(visible_models)} shown)",
    options=vision_model_labels,
    index=0 if vision_model_labels else None,
)
selected_vision_model = next(
    (m for m, lbl in zip(visible_models, vision_model_labels) if lbl == selected_vision_label),
    None,
)

run_product_identification = st.sidebar.checkbox(
    "Identify products with Vision LM", value=True)
max_products_to_enrich    = st.sidebar.slider("Detections to enrich", 1, 100, 10)
openfoodfacts_scan_limit  = st.sidebar.slider("OpenFoodFacts streamed rows", 100, 10000, 3000, 100)
openfoodfacts_results     = st.sidebar.slider("OpenFoodFacts matches per product", 1, 10, 5)
auto_save_results         = st.sidebar.checkbox("💾 Auto-save results to disk", value=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_upload, tab_camera_snap, tab_live, tab_results = st.tabs([
    "📁 Upload Image",
    "📷 Camera Snapshot",
    "🎥 Live Webcam",
    "🗂 Saved Results",
])

# ── Tab 1 : Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload an image for detection")
    uploaded_file = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        run_detection_pipeline(
            image=image,
            source_name=Path(uploaded_file.name).stem,
            model=model,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            img_size=img_size, max_detections=max_detections, use_tta=use_tta,
            use_sliced_inference=use_sliced_inference, tile_size=tile_size,
            tile_overlap=tile_overlap,
            run_product_identification=run_product_identification,
            effective_key=effective_key, selected_vision_model=selected_vision_model,
            max_products_to_enrich=max_products_to_enrich,
            openfoodfacts_results=openfoodfacts_results,
            openfoodfacts_scan_limit=openfoodfacts_scan_limit,
            auto_save=auto_save_results,
        )

# ── Tab 2 : Camera snapshot (st.camera_input) ─────────────────────────────────
with tab_camera_snap:
    st.subheader("Take a photo with your camera")
    st.info("Click **Take Photo** below — uses your device's front/back camera via the browser.", icon="📷")
    camera_photo = st.camera_input("Take photo")
    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_detection_pipeline(
            image=image,
            source_name=f"camera_snap_{ts}",
            model=model,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            img_size=img_size, max_detections=max_detections, use_tta=use_tta,
            use_sliced_inference=use_sliced_inference, tile_size=tile_size,
            tile_overlap=tile_overlap,
            run_product_identification=run_product_identification,
            effective_key=effective_key, selected_vision_model=selected_vision_model,
            max_products_to_enrich=max_products_to_enrich,
            openfoodfacts_results=openfoodfacts_results,
            openfoodfacts_scan_limit=openfoodfacts_scan_limit,
            auto_save=auto_save_results,
        )

# ── Tab 3 : Live webcam (OpenCV) ───────────────────────────────────────────────
with tab_live:
    st.subheader("Live Webcam Detection")
    st.info(
        "Captures frames from your local webcam via OpenCV. "
        "Click **▶ Start** to begin streaming, **⏹ Stop** to end. "
        "Each captured frame can be saved to disk.", icon="🎥"
    )

    available_cams = _list_available_cameras()
    cam_index = st.selectbox(
        "Camera index", options=available_cams,
        format_func=lambda i: f"Camera {i}", index=0,
    )

    col_start, col_stop, col_capture = st.columns(3)
    if col_start.button("▶ Start stream", use_container_width=True):
        st.session_state["live_running"] = True
    if col_stop.button("⏹ Stop stream", use_container_width=True):
        st.session_state["live_running"] = False

    live_frame_placeholder    = st.empty()
    live_annotated_placeholder = st.empty()
    live_status               = st.empty()

    if st.session_state.get("live_running", False):
        frame_rgb = _capture_single_frame(cam_index)
        if frame_rgb is None:
            live_status.error(f"Could not open camera {cam_index}.")
            st.session_state["live_running"] = False
        else:
            # Run YOLO on frame
            if use_sliced_inference:
                boxes, scores, classes = _predict_sliced(
                    model, frame_rgb, conf_threshold, iou_threshold,
                    img_size, max_detections, use_tta, tile_size, tile_overlap)
            else:
                boxes, scores, classes = _predict_full_image(
                    model, frame_rgb, conf_threshold, iou_threshold,
                    img_size, max_detections, use_tta)

            annotated = _draw_detections(frame_rgb, boxes, scores, classes, model.names)
            live_frame_placeholder.image(frame_rgb,  caption="Raw frame",      use_container_width=True)
            live_annotated_placeholder.image(annotated, caption=f"Detected — {len(boxes)} object(s)", use_container_width=True)
            live_status.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')} · {len(boxes)} detection(s)")

            # Store latest frame in session for capture button
            st.session_state["live_latest_frame"]    = frame_rgb
            st.session_state["live_latest_annotated"] = annotated
            st.session_state["live_latest_boxes"]    = boxes
            st.session_state["live_latest_scores"]   = scores
            st.session_state["live_latest_classes"]  = classes

            # Auto-refresh every 0.5 s while running
            time.sleep(0.5)
            st.rerun()

    # ── Capture + save current frame ──────────────────────────────────────────
    if col_capture.button("📸 Capture & analyse frame", use_container_width=True,
                          disabled="live_latest_frame" not in st.session_state):
        frame_rgb = st.session_state["live_latest_frame"]
        image     = Image.fromarray(frame_rgb)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.subheader("Captured frame — full analysis")
        run_detection_pipeline(
            image=image,
            source_name=f"webcam_{cam_index}_{ts}",
            model=model,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            img_size=img_size, max_detections=max_detections, use_tta=use_tta,
            use_sliced_inference=use_sliced_inference, tile_size=tile_size,
            tile_overlap=tile_overlap,
            run_product_identification=run_product_identification,
            effective_key=effective_key, selected_vision_model=selected_vision_model,
            max_products_to_enrich=max_products_to_enrich,
            openfoodfacts_results=openfoodfacts_results,
            openfoodfacts_scan_limit=openfoodfacts_scan_limit,
            auto_save=auto_save_results,
        )

# ── Tab 4 : Saved Results browser ─────────────────────────────────────────────
with tab_results:
    st.subheader("🗂 Saved Results Browser")

    cluster_choice = st.radio(
        "Cluster", ["✅ Successful", "❌ Unsuccessful", "📋 All"], horizontal=True
    )

    def _show_cluster(base: Path, label: str):
        folders = _list_result_folders(base)
        if not folders:
            st.info(f"No {label} results yet.")
            return
        st.caption(f"{len(folders)} session(s) in **{label}**")
        for folder in folders:
            meta_path = folder / "metadata.json"
            meta      = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            title     = f"{folder.name} · {meta.get('total_detections', '?')} product(s) identified"
            with st.expander(title, expanded=False):
                col_img, col_meta = st.columns([2, 1])
                ann_path = folder / "annotated.jpg"
                if ann_path.exists():
                    col_img.image(str(ann_path), caption="Annotated", use_container_width=True)
                orig_path = folder / "original.jpg"
                if orig_path.exists():
                    col_img.image(str(orig_path), caption="Original", use_container_width=True)
                col_meta.json(meta)

                # Show crops
                crop_files = sorted(folder.glob("crop_*.jpg"))
                if crop_files:
                    st.write(f"**{len(crop_files)} crop(s)**")
                    crop_cols = st.columns(min(len(crop_files), 5))
                    for ci, cf in enumerate(crop_files[:10]):
                        crop_cols[ci % 5].image(str(cf), caption=cf.stem, use_container_width=True)

                # Download annotated image
                if ann_path.exists():
                    with open(ann_path, "rb") as f:
                        st.download_button(
                            "⬇ Download annotated image",
                            data=f,
                            file_name=f"{folder.name}_annotated.jpg",
                            mime="image/jpeg",
                            key=f"dl_{folder.name}",
                        )

                # Delete
                if st.button("🗑 Delete this result", key=f"del_{folder.name}"):
                    _delete_result_folder(folder)
                    st.success("Deleted.")
                    st.rerun()

    if cluster_choice == "✅ Successful":
        _show_cluster(SUCCESSFUL_DIR, "successful")
    elif cluster_choice == "❌ Unsuccessful":
        _show_cluster(UNSUCCESSFUL_DIR, "unsuccessful")
    else:
        st.write("### ✅ Successful")
        _show_cluster(SUCCESSFUL_DIR, "successful")
        st.write("### ❌ Unsuccessful")
        _show_cluster(UNSUCCESSFUL_DIR, "unsuccessful")
