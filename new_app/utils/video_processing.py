from pathlib import Path
import tempfile

import cv2


def _extract_slide_region(frame):
    height, width = frame.shape[:2]

    top = int(height * 0.12)
    bottom = int(height * 0.90)
    left = int(width * 0.06)
    right = int(width * 0.94)

    cropped = frame[top:bottom, left:right]
    if cropped.size == 0:
        return frame

    return cropped


def _prepare_comparison_frame(frame):
    slide_region = _extract_slide_region(frame)
    gray = cv2.cvtColor(slide_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.resize(gray, (480, 270))


def _frame_change_score(previous_gray, current_gray):
    diff = cv2.absdiff(previous_gray, current_gray)
    pixel_change = float(diff.mean() / 255.0)

    previous_edges = cv2.Canny(previous_gray, 80, 160)
    current_edges = cv2.Canny(current_gray, 80, 160)
    edge_diff = cv2.absdiff(previous_edges, current_edges)
    edge_change = float(edge_diff.mean() / 255.0)

    return (0.7 * pixel_change) + (0.3 * edge_change)


def extract_unique_frames(
    video_path,
    output_dir=None,
    threshold=0.08,
    frame_step=10,
    min_time_between_slides=1.0,
):
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="slides_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 1.0

    previous_saved_frame = None
    frame_count = 0
    saved_frames = []
    last_saved_timestamp = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step != 0:
            frame_count += 1
            continue

        timestamp = round(frame_count / fps, 2)
        slide_region = _extract_slide_region(frame)
        prepared_frame = _prepare_comparison_frame(frame)

        if previous_saved_frame is None:
            score = 0.0
            should_save = True
        else:
            score = _frame_change_score(previous_saved_frame, prepared_frame)
            enough_time_passed = (timestamp - last_saved_timestamp) >= min_time_between_slides
            should_save = score >= threshold and enough_time_passed

        if should_save:
            filename = output_dir / f"slide_{frame_count:06d}.jpg"
            cv2.imwrite(str(filename), slide_region)
            saved_frames.append(
                {
                    "frame": str(filename),
                    "timestamp": timestamp,
                    "score": score,
                }
            )
            previous_saved_frame = prepared_frame
            last_saved_timestamp = timestamp

        frame_count += 1

    cap.release()
    return saved_frames
