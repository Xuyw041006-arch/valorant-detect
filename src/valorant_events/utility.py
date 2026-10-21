"""Optional world-utility detection and conservative presence events.

Only visible smoke, Fade Haunt eye/Prowler and Sova Owl Drone are supported.
An event is visual presence, never casting, damage, ownership or a kill.
Heavy model dependencies are imported only for an enabled local checkpoint.
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path

from .specialists import box_iou, match_boxes, valid_box

UTILITY_TYPES = ("smoke", "fade_eye", "fade_prowler", "sova_drone")
UTILITY_REGIONS = {
    "smoke": "visible_smoke_volume",
    "fade_eye": "haunt_eye_excluding_tendrils",
    "fade_prowler": "prowler_head_excluding_body_and_trail",
    "sova_drone": "visible_drone_body",
}


def _number(value, key, minimum=0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"utility.{key} must be finite")
    if value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"utility.{key} out of range")
    return float(value)


def utility_config(config):
    cfg = config.get("utility", config)
    if not isinstance(cfg, dict):
        raise ValueError("utility config must be an object")
    cfg = dict(cfg)
    if not isinstance(cfg.get("enabled", False), bool):
        raise ValueError("utility.enabled must be boolean")
    for key, default in (("confidence", 0.4), ("iou", 0.5), ("match_iou", 0.25)):
        cfg[key] = _number(cfg.get(key, default), key, 0, 1)
    if cfg["match_iou"] <= 0:
        raise ValueError("utility.match_iou must be positive")
    for key, default in (
        ("confirmation_seconds", 0.2),
        ("max_gap_seconds", 0.35),
        ("ttl_seconds", 1.0),
    ):
        cfg[key] = _number(cfg.get(key, default), key)
    if cfg["ttl_seconds"] <= 0 or cfg["max_gap_seconds"] <= 0:
        raise ValueError("utility gap and TTL must be positive")
    for key, default, lower in (
        ("min_observations", 3, 1),
        ("imgsz", 640, 32),
        ("max_det", 50, 1),
    ):
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"utility.{key} must be an integer >= {lower}")
        cfg[key] = value
    return cfg


def utility_class_mapping(names, class_map=None):
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Utility model requires named classes")
    if class_map is not None and not isinstance(class_map, dict):
        raise ValueError("utility.class_map must be a mapping")
    mapping = {}
    for key, raw in names.items():
        name = (class_map or {}).get(raw, raw)
        if name not in UTILITY_TYPES:
            raise ValueError(
                f"Unsupported utility class {raw!r}; use audited world-utility labels"
            )
        mapping[int(key)] = name
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate utility classes after mapping")
    return mapping


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class UtilityRunner:
    """predict(frame) returns raw, current-frame observations, with no tracking."""

    def __init__(self, config, model_factory=None, device=None):
        self.cfg = utility_config(config)
        self.enabled = self.cfg.get("enabled", False)
        self.device = device if device is not None else self.cfg.get("device", "cpu")
        self.model, self.mapping, self.metadata = None, {}, {"enabled": self.enabled}
        if not self.enabled:
            return
        raw = self.cfg.get("weights")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Enabled utility requires local weights")
        weights = Path(raw).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Missing local utility checkpoint: {weights}")
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self.model = model_factory(str(weights))
        if getattr(self.model, "task", None) != "detect":
            raise ValueError("Utility checkpoint must be a detection model")
        self.mapping = utility_class_mapping(
            self.model.names, self.cfg.get("class_map")
        )
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.metadata.update(
            weights=str(weights),
            weights_sha256=digest.hexdigest(),
            class_mapping=self.mapping,
            semantics="visible_world_utility_presence_only",
            detection_regions=UTILITY_REGIONS,
            tracking="class_specific_iou_ttl",
        )

    def predict(self, frame):
        if not self.enabled:
            return []
        height, width = frame.shape[:2]
        cfg = self.cfg
        results = self.model.predict(
            frame,
            device=self.device,
            imgsz=cfg["imgsz"],
            conf=cfg["confidence"],
            iou=cfg["iou"],
            max_det=cfg["max_det"],
            verbose=False,
        )
        if len(results) != 1:
            raise ValueError("Expected one utility result per frame")
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        ids, scores, coords = map(_as_list, (boxes.cls, boxes.conf, boxes.xyxy))
        if len(ids) != len(scores) or len(ids) != len(coords):
            raise ValueError("Inconsistent utility result lengths")
        output = []
        for class_id, score, box in zip(ids, scores, coords):
            if (
                not math.isfinite(float(class_id))
                or int(class_id) != class_id
                or int(class_id) not in self.mapping
            ):
                raise ValueError("Unknown utility class ID")
            if (
                not valid_box(box)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not cfg["confidence"] <= score <= 1
            ):
                continue
            box = [
                max(0.0, min(float(width), box[0])),
                max(0.0, min(float(height), box[1])),
                max(0.0, min(float(width), box[2])),
                max(0.0, min(float(height), box[3])),
            ]
            if valid_box(box):
                name = self.mapping[int(class_id)]
                output.append(
                    {
                        "label": name,
                        "utility_type": name,
                        "detection_region": UTILITY_REGIONS[name],
                        "bbox": box,
                        "confidence": float(score),
                    }
                )
        return output


class UtilityTracker:
    """Class-specific IoU tracks; confirm at least3 fresh observations over0.2s.

    Missing observations interrupt unconfirmed evidence. Reset/EOF never create
    action or disappearance events. IDs are local and do not affect ByteTrack.
    """

    def __init__(self, config):
        self.cfg = utility_config(config)
        self.states, self.counter, self.last_time, self.reset_floor = {}, 0, None, None

    def _time(self, timestamp):
        timestamp = _number(timestamp, "timestamp")
        if (
            self.last_time is not None
            and timestamp <= self.last_time
            or self.reset_floor is not None
            and timestamp < self.reset_floor
        ):
            raise ValueError("Utility timestamps must increase monotonically")
        self.last_time = timestamp
        return timestamp

    def reset(self, timestamp):
        timestamp = _number(timestamp, "reset_timestamp")
        if (
            self.last_time is not None
            and timestamp < self.last_time
            or self.reset_floor is not None
            and timestamp < self.reset_floor
        ):
            raise ValueError("Utility reset must not move backwards")
        self.states.clear()
        self.last_time, self.reset_floor = None, timestamp

    def update(self, timestamp, observations, frame_index, segment_id=0):
        timestamp = self._time(timestamp)
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer")
        cfg = self.cfg
        self.states = {
            key: value
            for key, value in self.states.items()
            if timestamp - value["last_seen"] < cfg["ttl_seconds"]
        }
        detections = []
        for item in observations:
            name, score = item.get("utility_type", item.get("label")), item.get(
                "confidence"
            )
            if (
                name not in UTILITY_TYPES
                or not valid_box(item.get("bbox"))
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not cfg["confidence"] <= score <= 1
            ):
                continue
            detections.append({**item, "label": name, "utility_type": name})
        unique = []
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            if not any(
                other["label"] == det["label"]
                and box_iou(other["bbox"], det["bbox"]) > 0.9
                for other in unique
            ):
                unique.append(det)
        assigned = {}
        for label in UTILITY_TYPES:
            ids = [key for key, st in self.states.items() if st["label"] == label]
            indices = [j for j, det in enumerate(unique) if det["label"] == label]
            for i, j, _ in match_boxes(
                [self.states[key]["bbox"] for key in ids],
                [unique[k]["bbox"] for k in indices],
                cfg["match_iou"],
            ):
                assigned[indices[j]] = ids[i]
        observed_ids = set(assigned.values())
        for key, state in self.states.items():
            if key not in observed_ids and not state["confirmed"]:
                state["observations"].clear()
        tracks, events = [], []
        for j, det in enumerate(unique):
            key = assigned.get(j)
            if key is None:
                self.counter += 1
                key = f"utility:{self.counter}"
                self.states[key] = {
                    "label": det["label"],
                    "bbox": det["bbox"],
                    "last_seen": timestamp,
                    "observations": deque(),
                    "confirmed": False,
                }
            state = self.states[key]
            if timestamp - state["last_seen"] > cfg["max_gap_seconds"]:
                state["observations"].clear()
            state["last_seen"], state["bbox"] = timestamp, list(det["bbox"])
            evidence = state["observations"]
            evidence.append((timestamp, det["confidence"], frame_index))
            # Only confirmation needs a history; bound memory for long-lived smoke.
            if state["confirmed"]:
                while len(evidence) > cfg["min_observations"]:
                    evidence.popleft()
            if (
                not state["confirmed"]
                and len(evidence) >= cfg["min_observations"]
                and timestamp - evidence[0][0] + 1e-9 >= cfg["confirmation_seconds"]
            ):
                state["confirmed"] = True
                events.append(
                    {
                        "event": "utility_detected",
                        "timestamp": evidence[0][0],
                        "emitted_at": timestamp,
                        "frame_index": evidence[0][2],
                        "segment_id": segment_id,
                        "track_id": key,
                        "category": "utility",
                        "confidence": sum(v[1] for v in evidence) / len(evidence),
                        "evidence": {
                            "utility_type": det["label"],
                            "detection_region": UTILITY_REGIONS[det["label"]],
                            "bbox": list(det["bbox"]),
                            "observations": len(evidence),
                            "semantics": "visible_world_utility_presence_only",
                            "tracker": "class_specific_iou_ttl",
                            "confidence_method": "mean_model_score_not_calibrated_probability",
                        },
                    }
                )
            tracks.append(
                {
                    **det,
                    "track_id": key,
                    "track_namespace": "utility",
                    "frame_index": frame_index,
                    "segment_id": segment_id,
                    "confirmed": state["confirmed"],
                }
            )
        return tracks, events

    def flush(self, timestamp):
        self.reset(timestamp)
        return []
