import base64
import io
import os
import re
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO

st.title("YOLOv9 Product Detection + Vision LM Lookup")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL_KEYWORDS = [
    "gpt-4o",
    "gpt-4-vision",
    "claude-3",
    "claude-3.5",
    "gemini",
    "llava",
    "vision",
    "pixtral",
    "qwen-vl",
    "intern-vl",
    "minicpm-v",
    "phi-3-vision",
]


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


def _compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) * np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
    union = area_a + area_b - inter
    union = np.maximum(union, 1e-9)

    return inter / union


def _classwise_nms(boxes, scores, classes, iou_thr, max_det):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    keep = []
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        cls_idx = np.where(classes == cls)[0]
        cls_scores = scores[cls_idx]
        order = cls_idx[np.argsort(-cls_scores)]

        while len(order) > 0:
            current = order[0]
            keep.append(current)
            if len(order) == 1:
                break

            rest = order[1:]
            ious = _compute_iou(boxes[current], boxes[rest])
            order = rest[ious < iou_thr]

    keep = np.array(keep, dtype=np.int64)
    keep = keep[np.argsort(-scores[keep])]
    return keep[:max_det]


def _draw_detections(image_rgb, boxes, scores, classes, names):
    canvas = image_rgb.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(classes[i])
        score = float(scores[i])

        color = (
            int((37 * (cls_id + 1)) % 255),
            int((17 * (cls_id + 7)) % 255),
            int((29 * (cls_id + 13)) % 255),
        )
        label = f"{names.get(cls_id, str(cls_id))} {score:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(y1 - 8, th + 4)
        cv2.rectangle(canvas, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return canvas


def _predict_full_image(model, img_array, conf, iou, imgsz, max_det, augment):
    result = model.predict(
        source=img_array,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        augment=augment,
        verbose=False,
    )[0]

    if len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    return boxes, scores, classes


def _predict_sliced(model, img_array, conf, iou, imgsz, max_det, augment, tile_size, overlap):
    h, w = img_array.shape[:2]
    stride = max(32, int(tile_size * (1.0 - overlap)))

    x_starts = list(range(0, max(1, w - tile_size + 1), stride))
    y_starts = list(range(0, max(1, h - tile_size + 1), stride))

    if not x_starts or x_starts[-1] != max(0, w - tile_size):
        x_starts.append(max(0, w - tile_size))
    if not y_starts or y_starts[-1] != max(0, h - tile_size):
        y_starts.append(max(0, h - tile_size))

    all_boxes = []
    all_scores = []
    all_classes = []

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            crop = img_array[y0:y1, x0:x1]

            tile_result = model.predict(
                source=crop,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max(200, max_det // 4),
                augment=augment,
                verbose=False,
            )[0]

            if len(tile_result.boxes) == 0:
                continue

            boxes = tile_result.boxes.xyxy.cpu().numpy().astype(np.float32)
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            scores = tile_result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = tile_result.boxes.cls.cpu().numpy().astype(np.int32)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    keep = _classwise_nms(boxes, scores, classes, iou_thr=iou, max_det=max_det)
    return boxes[keep], scores[keep], classes[keep]


def _discover_model_paths():
    candidates = []
    for pattern in ["*.pt", "models/*.pt"]:
        for model_path in Path(".").glob(pattern):
            candidates.append(str(model_path))

    if not candidates:
        return ["yolov9c.pt", "yolov8n.pt"]

    preferred = ["yolov9c.pt", "yolov8n.pt"]
    ordered = [m for m in preferred if m in candidates]
    ordered.extend(sorted([m for m in candidates if m not in ordered]))
    return ordered


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def _fallback_models() -> list[dict]:
    return [
        {"id": "openai/gpt-4o-mini", "label": "GPT-4o Mini", "vision": True, "free": False, "notes": "$0.150/1M · 128,000 ctx"},
        {"id": "openai/gpt-4o", "label": "GPT-4o", "vision": True, "free": False, "notes": "$2.500/1M · 128,000 ctx"},
        {"id": "anthropic/claude-3.5-sonnet", "label": "Claude 3.5 Sonnet", "vision": True, "free": False, "notes": "$3.000/1M · 200,000 ctx"},
        {"id": "google/gemini-flash-1.5", "label": "Gemini 1.5 Flash", "vision": True, "free": False, "notes": "$0.075/1M · 1,000,000 ctx"},
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openrouter_models(api_key: str) -> list[dict]:
    if not api_key:
        return _fallback_models()
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://pear-edtech.app",
                "X-Title": "Pear EdTech Chatbot",
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        models = []
        for m in raw:
            mid = m.get("id", "")
            name = m.get("name", mid)
            pricing = m.get("pricing", {})
            arch = m.get("architecture", {})
            modality = arch.get("modality", "")
            input_mods = arch.get("input_modalities", [])
            has_vision = (
                "image" in modality
                or "image" in input_mods
                or "multimodal" in modality
                or any(kw in mid.lower() for kw in VISION_MODEL_KEYWORDS)
                or any(kw in name.lower() for kw in VISION_MODEL_KEYWORDS)
            )
            if not has_vision:
                continue
            try:
                p_cost = float(pricing.get("prompt", 1))
                c_cost = float(pricing.get("completion", 1))
                is_free = p_cost == 0.0 and c_cost == 0.0
            except (TypeError, ValueError):
                is_free = False
            models.append(
                {
                    "id": mid,
                    "label": name,
                    "vision": has_vision,
                    "free": is_free,
                    "notes": "free" if is_free else "paid",
                }
            )
        models.sort(key=lambda x: (not x["free"], x["label"].lower()))
        return models or _fallback_models()
    except Exception:
        return _fallback_models()


def build_image_content_part(b64: str, mime_type: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def build_user_message_content(text: str, image_parts: list[dict] | None = None) -> str | list:
    if not image_parts:
        return text
    parts = [{"type": "text", "text": text}]
    parts.extend(image_parts)
    return parts


def call_openrouter(messages: list[dict], model: str, api_key: str, temperature: float = 0.1) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pear-edtech.app",
        "X-Title": "Pear EdTech Chatbot",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def pil_to_base64(image: Image.Image, image_format: str = "JPEG") -> tuple[str, str]:
    buf = io.BytesIO()
    mime_type = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    save_format = "PNG" if image_format.upper() == "PNG" else "JPEG"
    image.save(buf, format=save_format)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), mime_type


def extract_product_crop(image: Image.Image, box: np.ndarray, padding_ratio: float = 0.08) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box.tolist()
    pad_x = max(8, int((x2 - x1) * padding_ratio))
    pad_y = max(8, int((y2 - y1) * padding_ratio))
    left = max(0, int(x1) - pad_x)
    top = max(0, int(y1) - pad_y)
    right = min(width, int(x2) + pad_x)
    bottom = min(height, int(y2) + pad_y)
    return image.crop((left, top, right, bottom))


def extract_product_name(raw_reply: str) -> str:
    text = raw_reply.strip()
    match = re.search(r"product_name\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"^[-*\s]+", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    return text.strip(" '\"`")


def identify_product_name(crop: Image.Image, model_id: str, api_key: str, yolo_label: str) -> tuple[str, str]:
    b64, mime_type = pil_to_base64(crop)
    prompt = (
        "You are identifying a packaged food or drink product from a cropped image. "
        f"The YOLO detector label is '{yolo_label}'. "
        "Return only the most likely market-facing product name to help search OpenFoodFacts. "
        "Prefer brand + product name when visible. "
        "If the exact name is unclear, return the shortest useful search phrase. "
        "Output format: product_name: <answer>"
    )
    messages = [
        {
            "role": "user",
            "content": build_user_message_content(prompt, [build_image_content_part(b64, mime_type)]),
        }
    ]
    reply = call_openrouter(messages, model=model_id, api_key=api_key, temperature=0.0)
    return extract_product_name(reply), reply


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def score_openfoodfacts_match(item: dict, query: str) -> int:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", query.lower()) if tok]
    name = normalize_text(item.get("product_name"))
    brand = normalize_text(item.get("brands"))
    categories = normalize_text(item.get("categories"))
    combined = " ".join(part for part in [name, brand, categories] if part)

    if not combined:
        return -1

    score = 0
    if query.lower() == name:
        score += 200
    if query.lower() in name:
        score += 80
    if query.lower() in combined:
        score += 40

    for token in tokens:
        if token in name:
            score += 15
        elif token in combined:
            score += 6

    return score


@st.cache_data(show_spinner=False, ttl=3600)
def search_openfoodfacts(query: str, max_results: int = 5, max_scan: int = 3000) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    dataset = load_dataset("openfoodfacts/product-database", split="food", streaming=True)
    matches = []

    for idx, item in enumerate(dataset):
        if idx >= max_scan:
            break

        score = score_openfoodfacts_match(item, query)
        if score <= 0:
            continue

        matches.append(
            {
                "score": score,
                "product_name": item.get("product_name") or "",
                "brands": item.get("brands") or "",
                "categories": item.get("categories") or "",
                "countries": item.get("countries") or "",
                "nutriscore_grade": item.get("nutriscore_grade") or "",
                "image_small_url": item.get("image_small_url") or "",
                "url": item.get("url") or "",
                "code": item.get("code") or "",
            }
        )

    matches.sort(key=lambda x: (-x["score"], x["product_name"].lower()))
    return matches[:max_results]


available_models = _discover_model_paths()
selected_model_path = st.sidebar.selectbox(
    "YOLO weights",
    options=available_models,
    index=0,
    help="Use custom trained weights for flyer/product detection best results.",
)

try:
    model = load_model(selected_model_path)
except Exception as e:
    st.error(f"Failed to load model '{selected_model_path}': {e}")
    st.stop()

st.sidebar.subheader("Detection settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.01, max_value=0.90, value=0.10, step=0.01)
iou_threshold = st.sidebar.slider("NMS IoU threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.01)
img_size = st.sidebar.select_slider(
    "Inference image size",
    options=[640, 960, 1280, 1536, 1920],
    value=1536,
    help="Larger size helps detect small objects, but is slower.",
)
max_detections = st.sidebar.slider("Max detections", min_value=50, max_value=3000, value=1000, step=50)
use_tta = st.sidebar.checkbox("Test-time augmentation", value=True)
use_sliced_inference = st.sidebar.checkbox(
    "Sliced inference for tiny objects",
    value=True,
    help="Runs detection on overlapping tiles, then merges boxes with NMS.",
)
tile_size = st.sidebar.select_slider("Tile size", options=[512, 640, 768, 896, 1024], value=768, disabled=not use_sliced_inference)
tile_overlap = st.sidebar.slider(
    "Tile overlap",
    min_value=0.10,
    max_value=0.50,
    value=0.30,
    step=0.05,
    disabled=not use_sliced_inference,
)

st.sidebar.subheader("Vision LM + OpenFoodFacts")
api_key_input = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    value=st.session_state.get("openrouter_api_key", ""),
    help="Needed to identify each detected crop with a vision-capable model.",
)
if api_key_input:
    st.session_state["openrouter_api_key"] = api_key_input

effective_key = _get_api_key()
all_openrouter_models = fetch_openrouter_models(effective_key)

# ── Model filter + search: allow Free / Paid / Vision / All like other page
free_models = [m for m in all_openrouter_models if m.get("free")]
paid_models = [m for m in all_openrouter_models if not m.get("free")]
vision_models = [m for m in all_openrouter_models if m.get("vision")]

tier = st.sidebar.radio(
    "Show models:",
    ["🆓 Free Only", "💳 Paid Only", "👁 Vision Only", "🔀 All"],
    index=3,
    horizontal=True,
)
visible_models = (
    free_models if tier == "🆓 Free Only" else
    paid_models if tier == "💳 Paid Only" else
    vision_models if tier == "👁 Vision Only" else
    all_openrouter_models
)

search_q = st.sidebar.text_input("Search models", value="", help="Filter model list by name or id")
if search_q and search_q.strip():
    q = search_q.strip().lower()
    visible_models = [m for m in visible_models if q in m.get("label", "").lower() or q in m.get("id", "").lower()]

vision_model_labels = [f"{m['label']} ({m['id']})" for m in visible_models]
selected_vision_label = st.sidebar.selectbox(
    "Vision model",
    options=vision_model_labels,
    index=0 if vision_model_labels else None,
    help="Choose a vision-capable model to identify products (filtered view).",
)
selected_vision_model = next(
    (m for m, label in zip(visible_models, vision_model_labels) if label == selected_vision_label),
    None,
)
run_product_identification = st.sidebar.checkbox(
    "Identify detected products with Vision LM",
    value=True,
    help="For each detected object, crop it and ask the selected OpenRouter vision model for a product name.",
)
max_products_to_enrich = st.sidebar.slider(
    "Detections to enrich",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Keeps latency and API cost under control. Highest-confidence detections are processed first.",
)
openfoodfacts_scan_limit = st.sidebar.slider(
    "OpenFoodFacts max streamed rows per search",
    min_value=100,
    max_value=10000,
    value=3000,
    step=100,
    help="Higher values improve search recall but are slower.",
)
openfoodfacts_results = st.sidebar.slider(
    "OpenFoodFacts matches per product",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

st.caption(f"Loaded YOLO model: {selected_model_path}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    img_array = np.array(image)

    with st.spinner("Detecting objects..."):
        if use_sliced_inference:
            boxes, scores, classes = _predict_sliced(
                model,
                img_array,
                conf_threshold,
                iou_threshold,
                img_size,
                max_detections,
                use_tta,
                tile_size,
                tile_overlap,
            )
        else:
            boxes, scores, classes = _predict_full_image(
                model,
                img_array,
                conf_threshold,
                iou_threshold,
                img_size,
                max_detections,
                use_tta,
            )

    annotated_img = _draw_detections(img_array, boxes, scores, classes, model.names)
    st.image(annotated_img, caption="Detected objects", use_container_width=True)

    st.subheader("Detected Objects")
    if len(boxes) > 0:
        st.write(f"Total detections: {len(boxes)}")
        ordered_indices = list(np.argsort(-scores))
        for rank, i in enumerate(ordered_indices, start=1):
            class_id = int(classes[i])
            class_name = model.names[class_id]
            confidence = float(scores[i])
            st.write(f"{rank}. {class_name}: {confidence:.2f}")

        if run_product_identification:
            if not effective_key:
                st.warning("OpenRouter API key is missing, so product-name identification is skipped.")
            elif not selected_vision_model:
                st.warning("No vision-capable OpenRouter model is available.")
            else:
                st.subheader("Vision LM Product Identification + OpenFoodFacts Search")
                top_indices = ordered_indices[: min(max_products_to_enrich, len(ordered_indices))]
                progress = st.progress(0.0)
                enrichment_rows = []

                for step, idx in enumerate(top_indices, start=1):
                    crop = extract_product_crop(image, boxes[idx])
                    class_id = int(classes[idx])
                    class_name = model.names[class_id]
                    confidence = float(scores[idx])

                    try:
                        product_name, raw_reply = identify_product_name(
                            crop=crop,
                            model_id=selected_vision_model["id"],
                            api_key=effective_key,
                            yolo_label=class_name,
                        )
                        matches = search_openfoodfacts(
                            query=product_name,
                            max_results=openfoodfacts_results,
                            max_scan=openfoodfacts_scan_limit,
                        )
                        enrichment_rows.append(
                            {
                                "index": int(idx) + 1,
                                "yolo_class": class_name,
                                "confidence": round(confidence, 3),
                                "product_name": product_name,
                                "matches_found": len(matches),
                                "raw_reply": raw_reply,
                                "matches": matches,
                                "crop": crop,
                            }
                        )
                    except Exception as exc:
                        enrichment_rows.append(
                            {
                                "index": int(idx) + 1,
                                "yolo_class": class_name,
                                "confidence": round(confidence, 3),
                                "product_name": "",
                                "matches_found": 0,
                                "raw_reply": f"Error: {exc}",
                                "matches": [],
                                "crop": crop,
                            }
                        )

                    progress.progress(step / len(top_indices))

                st.dataframe(
                    [
                        {
                            "detection": row["index"],
                            "yolo_class": row["yolo_class"],
                            "confidence": row["confidence"],
                            "vision_product_name": row["product_name"],
                            "openfoodfacts_matches": row["matches_found"],
                        }
                        for row in enrichment_rows
                    ],
                    use_container_width=True,
                )

                for row in enrichment_rows:
                    with st.expander(
                        f"Detection {row['index']} | {row['yolo_class']} | {row['product_name'] or 'No product name'}",
                        expanded=False,
                    ):
                        st.image(row["crop"], caption=f"Crop for detection {row['index']}", width=220)
                        st.caption(f"YOLO confidence: {row['confidence']}")
                        st.code(row["raw_reply"], language=None)

                        if row["matches"]:
                            st.write("OpenFoodFacts matches")
                            st.dataframe(row["matches"], use_container_width=True)
                        else:
                            st.write("No OpenFoodFacts match found within the streamed search limit.")

        st.info(
            "This merged flow now does: YOLOv9 detection -> crop each detection -> OpenRouter vision model predicts product name -> "
            "streamed OpenFoodFacts search for that product."
        )
    else:
        st.write("No objects detected.")
