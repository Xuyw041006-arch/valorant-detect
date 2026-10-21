"""Small-object inference through overlapping source-resolution tiles.

Full-frame and tile predictions share the same model. Boxes are mapped back to
source pixels and class-aware NMS removes duplicates. This is an inference
option, not evidence of an accuracy improvement until validation is run.
"""

from __future__ import annotations
import math
from .specialists import box_iou


def tile_boxes(width, height, size=720, overlap=0.25, max_tiles=36):
    if width < 1 or height < 1 or size < 32 or not 0 <= overlap < 0.8:
        raise ValueError("Invalid tile geometry")

    def starts(length):
        if length <= size:
            return [0]
        stride = max(1, int(size * (1 - overlap)))
        return sorted(set(list(range(0, length - size + 1, stride)) + [length - size]))

    boxes = [
        (x, y, min(width, x + size), min(height, y + size))
        for y in starts(height)
        for x in starts(width)
    ]
    if len(boxes) > max_tiles:
        raise ValueError(
            "Image requires too many tiles; increase tile size or max_tiles explicitly"
        )
    return boxes


def merge_detections(detections, iou=0.5, max_det=100):
    if not math.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError("NMS IoU must be in [0,1]")
    kept = []
    for row in sorted(
        detections, key=lambda d: (-d["confidence"], str(d["label"]), *d["bbox"])
    ):
        if any(
            row["label"] == old["label"] and box_iou(row["bbox"], old["bbox"]) > iou
            for old in kept
        ):
            continue
        kept.append(row)
        if len(kept) >= max_det:
            break
    return kept


def predict_sliced(
    model,
    frame,
    mapping,
    *,
    device,
    imgsz=960,
    confidence=0.1,
    iou=0.5,
    max_det=50,
    tile_size=720,
    overlap=0.25,
    include_full=True,
):
    height, width = frame.shape[:2]
    boxes = tile_boxes(width, height, tile_size, overlap)
    if include_full and (len(boxes) > 1 or boxes[0] != (0, 0, width, height)):
        boxes.insert(0, (0, 0, width, height))
    detections = []
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        result = model.predict(
            frame[y1:y2, x1:x2],
            device=device,
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        for cls, score, bbox in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        ):
            mapped = [
                max(0.0, min(float(width), bbox[0] + x1)),
                max(0.0, min(float(height), bbox[1] + y1)),
                max(0.0, min(float(width), bbox[2] + x1)),
                max(0.0, min(float(height), bbox[3] + y1)),
            ]
            if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
                continue
            detections.append(
                {
                    "label": mapping[int(cls)],
                    "confidence": float(score),
                    "bbox": mapped,
                    "inference_source": (
                        "full_frame"
                        if (x1, y1, x2, y2) == (0, 0, width, height)
                        else "tile"
                    ),
                    "source_tile": [x1, y1, x2, y2],
                }
            )
    return merge_detections(detections, iou, max_det)
