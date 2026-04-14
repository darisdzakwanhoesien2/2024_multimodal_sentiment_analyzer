import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

st.title("Multiple Object Detection with YOLOv9")


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
        verbose=False
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
                verbose=False
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
        for model_path in Path('.').glob(pattern):
            candidates.append(str(model_path))

    if not candidates:
        return ['yolov9c.pt', 'yolov8n.pt']

    preferred = ['yolov9c.pt', 'yolov8n.pt']
    ordered = [m for m in preferred if m in candidates]
    ordered.extend(sorted([m for m in candidates if m not in ordered]))
    return ordered


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


available_models = _discover_model_paths()
selected_model_path = st.sidebar.selectbox(
    "Model weights",
    options=available_models,
    index=0,
    help="Use custom trained weights for flyer/product detection best results."
)

try:
    model = load_model(selected_model_path)
except Exception as e:
    st.error(f"Failed to load model '{selected_model_path}': {e}")
    st.stop()

st.sidebar.subheader("Detection settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.10,
    step=0.01,
    help="Lower values detect more objects but may add false positives."
)
iou_threshold = st.sidebar.slider(
    "NMS IoU threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.01
)
img_size = st.sidebar.select_slider(
    "Inference image size",
    options=[640, 960, 1280, 1536, 1920],
    value=1536,
    help="Larger size helps detect small objects, but is slower."
)
max_detections = st.sidebar.slider(
    "Max detections",
    min_value=50,
    max_value=3000,
    value=1000,
    step=50
)
use_tta = st.sidebar.checkbox(
    "Test-time augmentation (slower, can improve recall)",
    value=True
)
use_sliced_inference = st.sidebar.checkbox(
    "Sliced inference for tiny objects",
    value=True,
    help="Runs detection on overlapping tiles, then merges boxes with NMS."
)

tile_size = st.sidebar.select_slider(
    "Tile size",
    options=[512, 640, 768, 896, 1024],
    value=768,
    disabled=not use_sliced_inference
)
tile_overlap = st.sidebar.slider(
    "Tile overlap",
    min_value=0.10,
    max_value=0.50,
    value=0.30,
    step=0.05,
    disabled=not use_sliced_inference
)

st.caption(f"Loaded model: {selected_model_path}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Perform object detection
    with st.spinner('Detecting objects...'):
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

    # Get the annotated image
    annotated_img = _draw_detections(img_array, boxes, scores, classes, model.names)

    # Display the result
    st.image(annotated_img, caption='Detected Objects', use_column_width=True)

    # Display detected objects
    st.subheader("Detected Objects:")
    if len(boxes) > 0:
        st.write(f"Total detections: {len(boxes)}")
        for i in np.argsort(-scores):
            class_id = int(classes[i])
            class_name = model.names[class_id]
            confidence = float(scores[i])
            st.write(f"{i+1}. {class_name}: {confidence:.2f}")

        st.info(
            "If this still misses many products, the model likely was not trained on those item classes. "
            "Use custom product-trained weights in the model selector."
        )
    else:
        st.write("No objects detected.")