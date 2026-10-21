"""Run a fine-tuned four-class YOLO detector + ByteTrack + optional HUD OCR.

Heavy dependencies are imported only in run(), so --help and utility tests work
without a model installation. Events retain source-video timestamps, including
when --start selects a later portion. No game-state or process-memory access is
used. See run_summary.json for actual timing and timestamp fallbacks.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

from .integration import AbilityIntegration, prepare_integrated_config
from .events import EventEngine
from .hud import HudReader
from .stitching import TrackStitcher
from .tactical import TacticalReader
from .utility import UtilityRunner, UtilityTracker, utility_config
from .specialists import (
    GroundWeaponTracker,
    IdentityAssociator,
    SpecialistRunner,
    enrich_person_events,
    validate_specialist_config,
)

CANONICAL_LABELS = {"enemy", "teammate", "dropped_spike", "planted_spike"}
LABEL_ALIASES = {
    "enemies": "enemy",
    "teammates": "teammate",
    "ally": "teammate",
    "allies": "teammate",
    "spike_dropped": "dropped_spike",
    "spike_planted": "planted_spike",
}
COLORS = {
    "enemy": (70, 70, 245),
    "teammate": (80, 220, 80),
    "dropped_spike": (0, 220, 255),
    "planted_spike": (210, 80, 220),
    "dropped_weapon": (255, 190, 65),
    "smoke": (200, 130, 240),
    "fade_eye": (240, 130, 200),
    "fade_prowler": (240, 160, 150),
    "sova_drone": (230, 200, 100),
}
ARTIFACT_NAMES = (
    "events.json",
    "events.csv",
    "tracks.jsonl",
    "annotated.mp4",
    "run_summary.json",
    "hud_roi_preview.jpg",
    "resolved_config.json",
)


def normalize_label(value):
    normalized = re.sub(r"[\s_-]+", "_", str(value).strip().lower())
    canonical = LABEL_ALIASES.get(normalized, normalized)
    if canonical not in CANONICAL_LABELS:
        raise ValueError(
            f"Unsupported model class {value!r}. Expected enemy, teammate, dropped_spike, planted_spike; use your fine-tuned weights."
        )
    return canonical


def class_mapping(names):
    """Resolve model-owned IDs; never assume the order in a downloaded model."""
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Model has no valid class-name mapping")
    mapping = {int(key): normalize_label(value) for key, value in names.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Multiple model classes normalize to the same canonical class")
    if set(mapping.values()) != CANONICAL_LABELS:
        missing = sorted(CANONICAL_LABELS - set(mapping.values()))
        raise ValueError(
            f"This four-class pipeline requires all four classes; missing {missing}"
        )
    return mapping


def reset_boundaries(values):
    """Normalize explicit source-video reset times, preserving a human reason."""
    if not isinstance(values, list):
        raise ValueError(
            "reset_times must be a list of seconds or timestamp/reason objects"
        )
    output = []
    for value in values:
        if isinstance(value, dict):
            timestamp, reason = value.get("timestamp"), str(
                value.get("reason", "baseline_reset")
            )
        else:
            timestamp, reason = value, "baseline_reset"
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("Each reset timestamp must be finite nonnegative seconds")
        output.append({"timestamp": float(timestamp), "reason": reason})
    output.sort(key=lambda item: item["timestamp"])
    if len({item["timestamp"] for item in output}) != len(output):
        raise ValueError("Duplicate reset timestamps are ambiguous")
    return output


def source_timestamp(position_msec, frame_index, fps, previous=None):
    """Prefer decoder timestamps; document frame/fps fallback for broken backends.

    Inconsistent timestamps that cannot be repaired from source frame numbering
    are rejected rather than made monotonically increasing with invented times.
    """
    if not math.isfinite(float(fps)) or fps <= 0 or frame_index < 0:
        raise ValueError(
            "A positive source FPS and nonnegative frame index are required"
        )
    pts = float(position_msec) / 1000.0
    if (
        math.isfinite(pts)
        and pts >= 0
        and (frame_index == 0 or pts > 0)
        and (previous is None or pts > previous)
    ):
        return pts, "decoded_video_pts"
    fallback = frame_index / fps
    if previous is not None and fallback <= previous:
        raise ValueError(
            "Video timestamps are nonmonotonic and frame/FPS fallback is inconsistent"
        )
    return fallback, "frame_index_over_fps_fallback"


def _to_list(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)


def extract_tracks(result, frame_index, mapping, matched_ids, identity_map, segment_id):
    """Only return IDs updated by a detection in the tracker's current frame."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    ids, classes = _to_list(boxes.id), _to_list(boxes.cls)
    scores, coordinates = _to_list(boxes.conf), _to_list(boxes.xyxy)
    if not (len(ids) == len(classes) == len(scores) == len(coordinates)):
        raise ValueError("Inconsistent detector/tracker output lengths")
    tracks = []
    for raw_id, class_id, confidence, bbox in zip(ids, classes, scores, coordinates):
        raw_id, class_id = int(raw_id), int(class_id)
        if raw_id not in matched_ids:
            continue
        if class_id not in mapping:
            raise ValueError(f"Unknown output class ID {class_id}")
        values = [float(value) for value in bbox]
        confidence = float(confidence)
        if (
            len(values) != 4
            or not all(math.isfinite(v) for v in values)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or values[2] <= values[0]
            or values[3] <= values[1]
        ):
            continue
        identity = (segment_id, raw_id)
        if identity not in identity_map:
            identity_map[identity] = len(identity_map) + 1
        tracks.append(
            {
                "track_id": identity_map[identity],
                "tracker_id": raw_id,
                "segment_id": segment_id,
                "label": mapping[class_id],
                "confidence": confidence,
                "bbox": values,
                "frame_index": frame_index,
            }
        )
    return tracks


def write_event_files(output_dir, events):
    output_dir = Path(output_dir)
    (output_dir / "events.json").write_text(
        json.dumps(events, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = [
        "timestamp",
        "event",
        "confidence",
        "emitted_at",
        "track_id",
        "frame_index",
        "before",
        "after",
        "evidence",
    ]
    fields.extend(
        key
        for key in (
            "agent_name",
            "agent_confidence",
            "category",
            "kill_weapon",
            "kill_weapon_confidence",
        )
        if any(key in event for event in events)
    )
    with (output_dir / "events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["evidence"] = json.dumps(
                event.get("evidence", {}), separators=(",", ":"), allow_nan=False
            )
            writer.writerow(row)


def reconcile_spike_events(events, tactical_observation, timestamp, max_age=0.5):
    """Current explicit HUD state vetoes contradictions; raw detections remain."""
    if not tactical_observation:
        return events, []
    observation_time = tactical_observation.get("timestamp")
    values = (timestamp, observation_time, max_age)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        return events, []
    age = timestamp - observation_time
    if not 0 <= age <= max_age:
        return events, []
    if not tactical_observation.get("ui", {}).get("spike_state_confirmed", False):
        return events, []
    state = tactical_observation.get("ui", {}).get("spike_state")
    contradicted = {
        "planted": "dropped_spike_detected",
        "dropped": "planted_spike_detected",
    }.get(state)
    kept = []
    suppressed = []
    for event in events:
        if event["event"] == contradicted:
            suppressed.append(
                {
                    "prediction": event,
                    "reason": "contradicts_current_explicit_HUD_spike_state",
                    "hud_state": state,
                    "hud_timestamp": tactical_observation["timestamp"],
                }
            )
        else:
            kept.append(event)
    return kept, suppressed


TRACK_CSV_FIELDS = [
    "timestamp",
    "frame_index",
    "track_id",
    "track_namespace",
    "segment_id",
    "label",
    "confidence",
    "bbox",
    "agent_name",
    "agent_confidence",
    "agent_evidence",
    "confirmed",
]


def write_track_rows(writer, timestamp, tracks):
    for track in tracks:
        row = {**track, "timestamp": timestamp}
        for key in ("bbox", "agent_evidence"):
            if key in row:
                row[key] = json.dumps(row[key], separators=(",", ":"), allow_nan=False)
        row.setdefault("track_namespace", "baseline")
        writer.writerow(row)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _draw_text(cv2, image, text, origin, color=(240, 240, 240), scale=0.6):
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _draw_rois(cv2, frame, reader):
    image = frame.copy()
    for name, (x1, y1, x2, y2) in reader.pixel_rois(frame.shape).items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        _draw_text(
            cv2, image, f"{name} ROI - calibrate", (x1, max(18, y1 - 8)), (0, 220, 255)
        )
    return image


def _annotate(
    cv2, frame, tracks, timestamp, recent, hud, hud_timestamp, reader, config
):
    image = (
        _draw_rois(cv2, frame, reader)
        if config.get("show_hud_rois", False)
        else frame.copy()
    )
    for track in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in track["bbox"]]
        color = COLORS[track["label"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        identity = (
            f" | Agent {track.get('agent_name') or '?'}"
            if "agent_name" in track
            else ""
        )
        _draw_text(
            cv2,
            image,
            f'{track["label"]} #{track["track_id"]} {track["confidence"]:.2f}{identity}',
            (max(0, x1), max(18, y1 - 7)),
            color,
        )
    _draw_text(
        cv2, image, f"Source time {timestamp:.3f}s | YOLO + temporal tracking", (16, 26)
    )
    if reader.enabled and hud is not None:
        health = "?" if hud.get("health") is None else str(hud["health"])
        ammo = "?" if hud.get("ammo") is None else str(hud["ammo"])
        _draw_text(
            cv2,
            image,
            f"OCR HP {health} | magazine {ammo} | sample age {timestamp - hud_timestamp:.2f}s",
            (16, 52),
        )
    else:
        _draw_text(
            cv2,
            image,
            "HUD OCR disabled" if not reader.enabled else "HUD OCR not sampled",
            (16, 52),
        )
    for index, event in enumerate(recent):
        description = event["event"]
        if event.get("track_id") is not None:
            description += f' #{event["track_id"]}'
        if "before" in event:
            description += f' {event["before"]}->{event["after"]}'
        _draw_text(
            cv2,
            image,
            f'{event["timestamp"]:.2f}s {description}',
            (16, 80 + index * 24),
            (0, 230, 255),
        )
    return image


def _tracker(model):
    trackers = getattr(getattr(model, "predictor", None), "trackers", None)
    if not trackers or not hasattr(trackers[0], "tracked_stracks"):
        raise RuntimeError(
            "Expected Ultralytics ByteTrack/BoT-SORT state is unavailable; check the pinned version"
        )
    return trackers[0]


def _validate_config(config):
    validate_specialist_config(config)
    utility_config(config.get("utility", {}))
    detector, tracker = config.get("detector", {}), config.get("tracker", {})
    tracker_name = tracker.get("name", "bytetrack.yaml")
    if tracker_name not in {"bytetrack.yaml", "botsort.yaml"}:
        raise ValueError("tracker.name must be bytetrack.yaml or botsort.yaml")
    for name, default in (("confidence", 0.15), ("iou", 0.5)):
        value = float(detector.get(name, default))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"detector.{name} must be in [0, 1]")
    if int(detector.get("imgsz", 640)) < 32 or int(detector.get("max_det", 100)) < 1:
        raise ValueError("Detector imgsz must be >= 32 and max_det positive")
    for value in (
        tracker.get("buffer_seconds", 1.5),
        config.get("hud", {}).get("interval_seconds", 0.2),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(
                "Tracker buffer and OCR interval must be finite positive seconds"
            )
    visual = config.get("visualization", {})
    hold_seconds = float(visual.get("event_hold_seconds", 2.5))
    if (
        not math.isfinite(hold_seconds)
        or hold_seconds < 0
        or int(visual.get("max_recent_events", 5)) < 1
    ):
        raise ValueError("Invalid visualization settings")


def run(args):
    video, weights, output = (
        args.video.resolve(),
        args.weights.resolve(),
        args.output.resolve(),
    )
    if not video.is_file() or not weights.is_file():
        raise FileNotFoundError(
            "--video and --weights must be existing local files; weights must be your four-class fine-tuned .pt file"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    config = deepcopy(config)
    if getattr(args, "tactical_config", None):
        profile = json.loads(args.tactical_config.read_text(encoding="utf-8"))
        config["tactical"] = profile.get("tactical", profile)
    config.setdefault("hud", {})
    if args.no_ocr:
        config["hud"]["enabled"] = False
        if "tactical" in config:
            config["tactical"].setdefault("ui", {})["ocr_enabled"] = False
    for task, attribute in (
        ("agent_identity", "agent_weights"),
        ("ground_weapon", "weapon_weights"),
    ):
        path = getattr(args, attribute, None)
        if path is not None:
            config.setdefault("specialists", {}).setdefault(task, {}).update(
                enabled=True, weights=str(path.resolve())
            )
    config = prepare_integrated_config(config, args.device, args.no_ocr)
    _validate_config(config)
    boundaries = reset_boundaries(config.get("reset_times", []))
    boundaries = deque(item for item in boundaries if item["timestamp"] >= args.start)
    engine = EventEngine(config)
    stitcher = (
        TrackStitcher(config.get("tracker", {}).get("stitching", {}))
        if config.get("tracker", {}).get("stitching", {}).get("enabled", False)
        else None
    )
    reader = HudReader(
        config,
        backend=config["hud"].get("backend", "easyocr"),
        gpu=config["hud"].get("gpu", False),
    )
    tactical_reader = TacticalReader(
        config, gpu=config.get("tactical", {}).get("gpu", False)
    )
    tactical_interval = float(
        config.get("tactical", {}).get("sample_interval_seconds", 0.2)
    )
    if not math.isfinite(tactical_interval) or tactical_interval <= 0:
        raise ValueError("tactical.sample_interval_seconds must be positive")
    specialist_configs = validate_specialist_config(config)
    for task, normalized in specialist_configs.items():
        config["specialists"][
            task
        ] = normalized  # Persist the actual thresholds/crop policy, including defaults.
    utility_enabled = bool(config.get("utility", {}).get("enabled", False))
    ability_enabled = bool(config.get("abilities", {}).get("enabled", False))
    artifacts = ARTIFACT_NAMES + (
        ("tracks.csv",) if specialist_configs or utility_enabled else ()
    )
    if ability_enabled:
        artifacts += ("abilities.jsonl",)
    if tactical_reader.enabled:
        artifacts += (
            "tactical.jsonl",
            "suppressed_events.json",
            "tactical_roi_preview.jpg",
        )
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError(
            f"Output artifacts already exist in {output}; use a new --output directory"
        )
    import cv2
    import ultralytics
    from ultralytics import YOLO

    setup_start = time.perf_counter()
    abilities = AbilityIntegration(config, args.device)
    specialists = SpecialistRunner(config, YOLO, args.device)
    utility_runner = UtilityRunner(config.get("utility", {}), YOLO, args.device)
    utility_tracker = UtilityTracker(config["utility"]) if utility_enabled else None
    identity_associator = (
        IdentityAssociator(specialists.configs["agent_identity"])
        if "agent_identity" in specialists.configs
        else None
    )
    weapon_tracker = (
        GroundWeaponTracker(specialists.configs["ground_weapon"])
        if "ground_weapon" in specialists.configs
        else None
    )
    model = YOLO(str(weights))
    mapping = class_mapping(model.names)
    if getattr(model, "task", "detect") != "detect":
        raise ValueError("The pipeline requires an object detection model")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not decode video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(
            "Video has no valid FPS; remux it with reliable timestamps first"
        )
    nominal_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    nominal_frames = (
        int(nominal_frames)
        if math.isfinite(nominal_frames) and nominal_frames >= 0
        else None
    )
    seek_frame = max(0, int(math.floor(args.start * fps)))
    if seek_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, seek_frame):
        seek_frame = 0  # Decode sequentially and use source timestamps to skip.
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "resolved_config.json", config)
    versions = {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
    }
    for package in ("torch", "numpy", "easyocr", "lap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video),
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "class_mapping": mapping,
        "device": args.device,
        "versions": versions,
        "source_fps": fps,
        "source_nominal_frame_count": nominal_frames,
        "requested_start_seconds": args.start,
        "requested_duration_seconds": args.duration,
        "max_frames": args.max_frames,
        "ocr_enabled": reader.enabled,
        "tracker": config.get("tracker", {}).get("name", "bytetrack.yaml"),
        "reset_boundaries_applied": [],
        "event_confidence_note": "Rule-based evidence scores; not calibrated probabilities.",
        "event_time_note": "timestamp is first supporting source-video evidence; emitted_at is confirmation time; disappearance timestamp is last_seen + grace period.",
        "output_video_note": "Silent CFR mp4v at nominal input FPS; authoritative source timestamps are in JSON/JSONL. VFR inputs may have a different rendered duration.",
        "known_limitations": [
            "HUD ROIs require per-video calibration",
            "HUD weapon_key is unavailable; explicit reset_times are required around weapon changes, round resets, and camera cuts",
            "Ammo increases are not classified as reloads",
            "Default ByteTrack lacks camera compensation; optional BoT-SORT uses optical-flow compensation without appearance ReID",
            "Dataset enemy/teammate labels may not generalize across spectator views, HUD themes, and game updates",
        ],
        "artifacts": {name: name for name in artifacts},
        "setup_seconds": time.perf_counter() - setup_start,
    }
    if specialists.configs:
        summary["specialists"] = specialists.metadata
        summary["known_limitations"].extend(
            [
                "Agent identity does not change baseline enemy/teammate labels; unknown is retained when evidence is insufficient",
                "Known silhouette/ineligible attributes are rejected; unmarked silhouettes may still be misclassified",
                "Weapon events describe visible ground-weapon presence only; IoU tracking can fragment during camera motion",
            ]
        )
    summary["utility"] = utility_runner.metadata
    event_output, recent, identity_map = [], deque(), {}
    timestamp_counts, observed_labels = Counter(), Counter()
    segment_id, processed, ocr_samples = 0, 0, 0
    frame_counter, previous_decoded = seek_frame - 1, None
    first_timestamp, last_timestamp, last_hud_time, last_hud = None, None, None, None
    writer, output_shape, failure = None, None, None
    detector_seconds = ocr_seconds = specialist_seconds = utility_seconds = 0.0
    csv_stream = csv_writer = None
    tactical_stream = ability_stream = None
    ability_seconds = 0.0
    last_tactical = None
    tactical_seconds = tactical_samples = 0
    suppressed_events = []
    stop_reason = "end_of_file"
    stream_start = time.perf_counter()
    last_progress = stream_start
    try:
        if ability_enabled:
            ability_stream = (output / "abilities.jsonl").open("w", encoding="utf-8")
        if tactical_reader.enabled:
            tactical_stream = (output / "tactical.jsonl").open("w", encoding="utf-8")
        if specialists.configs or utility_enabled:
            csv_stream = (output / "tracks.csv").open("w", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(
                csv_stream, fieldnames=TRACK_CSV_FIELDS, extrasaction="ignore"
            )
            csv_writer.writeheader()
        with (output / "tracks.jsonl").open("w", encoding="utf-8") as tracks_stream:
            while True:
                if args.max_frames is not None and processed >= args.max_frames:
                    stop_reason = "max_frames"
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                position_frames = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                reported_index = (
                    int(round(position_frames)) - 1
                    if math.isfinite(position_frames) and position_frames >= 1
                    else frame_counter + 1
                )
                if reported_index <= frame_counter:
                    raise ValueError(
                        "Decoder frame indices moved backward; refusing to invent source timestamps"
                    )
                frame_counter = reported_index
                timestamp, time_source = source_timestamp(
                    capture.get(cv2.CAP_PROP_POS_MSEC),
                    frame_counter,
                    fps,
                    previous_decoded,
                )
                previous_decoded = timestamp
                if timestamp + 1e-9 < args.start:
                    continue
                if (
                    args.duration is not None
                    and timestamp >= args.start + args.duration - 1e-9
                ):
                    stop_reason = "duration"
                    break
                while boundaries and boundaries[0]["timestamp"] <= timestamp + 1e-9:
                    boundary = boundaries.popleft()
                    engine.reset(timestamp, boundary["reason"])
                    abilities.reset()
                    tactical_reader.reset(timestamp, boundary["reason"])
                    last_tactical = None
                    if stitcher:
                        stitcher.reset()
                    if identity_associator:
                        identity_associator.reset(timestamp)
                    if weapon_tracker:
                        weapon_tracker.reset(timestamp)
                    if utility_tracker:
                        utility_tracker.reset(timestamp)
                    if getattr(getattr(model, "predictor", None), "trackers", None):
                        _tracker(model).reset()
                    segment_id += 1
                    recent.clear()
                    last_hud, last_hud_time = None, None
                    summary["reset_boundaries_applied"].append(
                        {**boundary, "applied_at": timestamp}
                    )
                if writer is None:
                    height, width = frame.shape[:2]
                    output_shape = (width, height)
                    writer = cv2.VideoWriter(
                        str(output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        output_shape,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            "OpenCV could not create the mp4v annotated video"
                        )
                    preview = _draw_rois(cv2, frame, reader)
                    if not cv2.imwrite(str(output / "hud_roi_preview.jpg"), preview):
                        raise RuntimeError("Could not write HUD ROI preview")
                    if tactical_reader.enabled:
                        tactical_preview = _draw_rois(cv2, frame, tactical_reader)
                        cv2.imwrite(
                            str(output / "tactical_roi_preview.jpg"), tactical_preview
                        )
                if (frame.shape[1], frame.shape[0]) != output_shape:
                    raise ValueError("Video resolution changed during decoding")
                detector = config.get("detector", {})
                inference_start = time.perf_counter()
                results = model.track(
                    frame,
                    persist=True,
                    tracker=config.get("tracker", {}).get("name", "bytetrack.yaml"),
                    device=args.device,
                    imgsz=int(detector.get("imgsz", 640)),
                    conf=float(detector.get("confidence", 0.15)),
                    iou=float(detector.get("iou", 0.5)),
                    max_det=int(detector.get("max_det", 100)),
                    verbose=False,
                )
                tracker = _tracker(model)
                # ByteTrack's default age is measured in processed frames. We
                # process every decoded frame and scale the buffer to source FPS.
                if not hasattr(tracker, "max_frames_lost"):
                    raise RuntimeError(
                        "Expected ByteTrack.max_frames_lost is unavailable; check Ultralytics 8.4.63"
                    )
                tracker.max_frames_lost = max(
                    1,
                    int(
                        math.ceil(
                            float(config.get("tracker", {}).get("buffer_seconds", 1.5))
                            * fps
                        )
                    ),
                )
                matched_ids = {
                    int(item.track_id)
                    for item in tracker.tracked_stracks
                    if item.is_activated and item.frame_id == tracker.frame_id
                }
                if len(results) != 1:
                    raise RuntimeError("Expected one YOLO result for one decoded frame")
                tracks = extract_tracks(
                    results[0],
                    frame_counter,
                    mapping,
                    matched_ids,
                    identity_map,
                    segment_id,
                )
                if stitcher:
                    tracks = stitcher.update(frame, timestamp, tracks)
                detector_seconds += time.perf_counter() - inference_start
                specialist_observations, weapon_tracks, weapon_events = {}, [], []
                if specialists.configs:
                    specialist_start = time.perf_counter()
                    specialist_observations = specialists.predict(frame, tracks)
                    if identity_associator:
                        tracks = identity_associator.update(
                            timestamp,
                            tracks,
                            specialist_observations.get("agent_identity", []),
                        )
                    if weapon_tracker:
                        weapon_tracks, weapon_events = weapon_tracker.update(
                            timestamp,
                            specialist_observations.get("ground_weapon", []),
                            frame_counter,
                            segment_id,
                        )
                    specialist_seconds += time.perf_counter() - specialist_start
                utility_observations, utility_tracks, utility_events = [], [], []
                if utility_enabled:
                    utility_start = time.perf_counter()
                    utility_observations = utility_runner.predict(frame)
                    utility_tracks, utility_events = utility_tracker.update(
                        timestamp, utility_observations, frame_counter, segment_id
                    )
                    utility_seconds += time.perf_counter() - utility_start
                hud_reading = None
                interval = float(config["hud"].get("interval_seconds", 0.2))
                if reader.enabled and (
                    last_hud_time is None
                    or timestamp - last_hud_time + 1e-9 >= interval
                ):
                    ocr_start = time.perf_counter()
                    hud_reading = reader.read(
                        frame
                    )  # Original pixels, before any overlay or resize.
                    ocr_seconds += time.perf_counter() - ocr_start
                    last_hud, last_hud_time = hud_reading, timestamp
                    ocr_samples += 1
                tactical_reading = None
                if tactical_reader.enabled and (
                    last_tactical is None
                    or timestamp - last_tactical["timestamp"] + 1e-9
                    >= tactical_interval
                ):
                    tactical_start = time.perf_counter()
                    if (
                        reader._reader is not None
                        and tactical_reader._reader is None
                        and config["tactical"].get("ui", {}).get("languages", ["en"])
                        == ["en"]
                    ):
                        tactical_reader._reader = reader._reader
                    tactical_reading = tactical_reader.read(frame, timestamp)
                    last_tactical = tactical_reading
                    tactical_seconds += time.perf_counter() - tactical_start
                    tactical_samples += 1
                    tactical_stream.write(
                        json.dumps(tactical_reading, allow_nan=False) + "\n"
                    )
                ability_reading, ability_events = None, []
                if ability_enabled:
                    ability_start = time.perf_counter()
                    # All readers see original source pixels before drawing.
                    if (
                        abilities.reader.ocr is None
                        and reader._reader is not None
                        and config["abilities"]
                        .get("status", {})
                        .get("languages", ["en"])
                        == ["en"]
                    ):
                        abilities.reader.ocr = reader._reader
                    ability_reading, ability_events = abilities.sample(
                        frame, timestamp, frame_counter, segment_id
                    )
                    ability_seconds += time.perf_counter() - ability_start
                    if ability_reading is not None:
                        ability_stream.write(
                            json.dumps(ability_reading, allow_nan=False) + "\n"
                        )
                events = engine.update(timestamp, tracks, hud_reading)
                events, suppressed = reconcile_spike_events(
                    events, last_tactical, timestamp
                )
                suppressed_events.extend(suppressed)
                if identity_associator:
                    events = enrich_person_events(events, tracks)
                events.extend(ability_events)
                events.extend(weapon_events)
                events.extend(utility_events)
                if tactical_reading:
                    events.extend(tactical_reading["events"])
                events.sort(
                    key=lambda event: (
                        event["timestamp"],
                        event["emitted_at"],
                        event["event"],
                    )
                )
                tracks = tracks + weapon_tracks + utility_tracks
                event_output.extend(events)
                recent.extend(events)
                visual = config.get("visualization", {})
                while recent and timestamp - recent[0]["emitted_at"] > float(
                    visual.get("event_hold_seconds", 2.5)
                ):
                    recent.popleft()
                while len(recent) > int(visual.get("max_recent_events", 5)):
                    recent.popleft()
                annotated = _annotate(
                    cv2,
                    frame,
                    tracks,
                    timestamp,
                    recent,
                    last_hud,
                    last_hud_time,
                    reader,
                    visual,
                )
                if last_tactical:
                    tactical_reader.draw(annotated, last_tactical)
                    _draw_text(
                        cv2,
                        annotated,
                        f"Minimap sample age: {timestamp-last_tactical['timestamp']:.2f}s",
                        (16, annotated.shape[0] - 24),
                        (230, 200, 100),
                        0.5,
                    )
                abilities.draw(cv2, annotated, timestamp)
                writer.write(annotated)
                record = {
                    "timestamp": timestamp,
                    "frame_index": frame_counter,
                    "segment_id": segment_id,
                    "tracks": tracks,
                    "hud_sampled": hud_reading is not None,
                    "hud": hud_reading,
                }
                if ability_enabled:
                    record["ability_sampled"] = ability_reading is not None
                if specialists.configs:
                    record["specialist_observations"] = specialist_observations
                if utility_enabled:
                    record["utility_observations"] = utility_observations
                if csv_writer:
                    write_track_rows(csv_writer, timestamp, tracks)
                if tactical_reader.enabled:
                    record["tactical_sampled"] = tactical_reading is not None
                tracks_stream.write(json.dumps(record, allow_nan=False) + "\n")
                timestamp_counts[time_source] += 1
                observed_labels.update(track["label"] for track in tracks)
                first_timestamp = (
                    timestamp if first_timestamp is None else first_timestamp
                )
                last_timestamp = timestamp
                processed += 1
                if time.perf_counter() - last_progress >= 5:
                    elapsed = time.perf_counter() - stream_start
                    print(
                        f"Processed {processed} frames; source {timestamp:.2f}s; {len(event_output)} events; measured {processed / elapsed:.2f} frames/s",
                        flush=True,
                    )
                    last_progress = time.perf_counter()
        if last_timestamp is not None:
            event_output.extend(engine.flush(last_timestamp))
            if weapon_tracker:
                event_output.extend(weapon_tracker.flush(last_timestamp))
        if processed == 0:
            raise ValueError(
                "No frames in the selected interval; check --start, --duration, and video duration"
            )
    except KeyboardInterrupt as exc:
        failure = exc
        stop_reason = "interrupted"
    except Exception as exc:
        failure = exc
        stop_reason = "error"
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_stream is not None:
            csv_stream.close()
        if tactical_stream is not None:
            tactical_stream.close()
        if ability_stream is not None:
            ability_stream.close()
        elapsed = time.perf_counter() - stream_start
        status = (
            "interrupted"
            if isinstance(failure, KeyboardInterrupt)
            else ("failed" if failure else "complete")
        )
        summary.update(
            {
                "status": status,
                "stop_reason": stop_reason,
                "processed_frames": processed,
                "first_source_timestamp": first_timestamp,
                "last_source_timestamp": last_timestamp,
                "timestamp_source_counts": dict(timestamp_counts),
                "source_span_seconds": (
                    last_timestamp - first_timestamp if processed else 0.0
                ),
                "rendered_duration_seconds": processed / fps,
                "processing_wall_seconds": elapsed,
                "measured_processing_fps": processed / elapsed if elapsed > 0 else None,
                "detector_and_tracker_wall_seconds": detector_seconds,
                "ocr_wall_seconds": ocr_seconds,
                "ocr_samples": ocr_samples,
                "unique_track_ids": len(identity_map),
                "observed_box_count_by_label": dict(observed_labels),
                "event_count": len(event_output),
                "event_count_by_type": dict(Counter(e["event"] for e in event_output)),
                "performance_note": "Wall-clock measurements include first inference and OCR initialization/download when enabled; no benchmark or real-time guarantee.",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if specialists.configs:
            summary["specialist_wall_seconds"] = specialist_seconds
            summary["unique_weapon_track_ids"] = (
                weapon_tracker.created_tracks if weapon_tracker else 0
            )
        if ability_enabled:
            summary["abilities"] = {
                "samples": abilities.samples,
                "wall_seconds": ability_seconds,
                "channels_separate": True,
                "config": config["abilities"],
            }
        if utility_enabled:
            summary["utility_wall_seconds"] = utility_seconds
        if stitcher:
            summary["stitching"] = {
                "raw_ids": len(identity_map),
                "canonical_ids": stitcher.created,
                "bridges": stitcher.bridges,
                "note": "Fewer IDs do not prove correct identity association; evaluate annotated trajectories.",
            }
        if tactical_reader.enabled:
            summary["tactical"] = {
                "samples": tactical_samples,
                "wall_seconds": tactical_seconds,
                "profile": config["tactical"].get("profile_name"),
                "minimap_coordinate_space": "normalized_roi",
                "world_positions": False,
                "suppressed_contradictory_spike_events": len(suppressed_events),
            }
            _write_json(output / "suppressed_events.json", suppressed_events)
        if failure:
            summary["error"] = f"{type(failure).__name__}: {failure}"
        if (
            stop_reason == "end_of_file"
            and nominal_frames
            and frame_counter + 1 < nominal_frames - 1
        ):
            summary["decode_warning"] = (
                "Decoder ended before the nominal frame count; check the input for truncated or corrupt video."
            )
        event_output.sort(
            key=lambda event: (event["timestamp"], event["emitted_at"], event["event"])
        )
        write_event_files(output, event_output)
        _write_json(output / "run_summary.json", summary)
    if failure:
        raise failure
    if (
        config.get("narrative", {}).get("enabled", False)
        and summary["status"] == "complete"
    ):
        from .story_replay import enrich
        from .killfeed_agents import KillfeedAgentReader

        portrait_assets = (
            Path(__file__).resolve().parents[2] / "assets" / "agent_portraits"
        )
        narrative_start = time.perf_counter()
        detail = enrich(output, video, video.stem, KillfeedAgentReader(portrait_assets))
        from .normalized_events import normalize_events
        from .compact_events import write_compact

        bundle = write_compact(output / "compact", normalize_events(detail["events"]))
        summary["compact_events"] = {
            "count": len(bundle["events"]),
            "directory": "compact",
        }
        summary["detailed_events"] = {
            "count": len(detail["events"]),
            "mode": "offline_enrichment",
            "postprocess_wall_seconds": time.perf_counter() - narrative_start,
        }
        _write_json(output / "run_summary.json", summary)
    print(
        f"Complete: {processed} frames, {len(event_output)} events. Outputs: {output}"
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path, required=True, help="Local first-person gameplay video"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Fine-tuned four-class YOLO weights (.pt)",
    )
    parser.add_argument(
        "--agent-weights",
        type=Path,
        help="Optional Agent specialist weights; mode/semantics must be configured",
    )
    parser.add_argument(
        "--weapon-weights",
        type=Path,
        help="Optional dropped-weapon detector weights; verified semantics must be configured",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument(
        "--tactical-config",
        type=Path,
        help="Per-video minimap and killfeed calibration profile",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/demo"), help="New output directory"
    )
    parser.add_argument("--device", default="cpu", help="0 for CUDA, cpu, or mps")
    parser.add_argument(
        "--no-ocr", action="store_true", help="Disable all HUD OCR and model downloads"
    )
    parser.add_argument(
        "--max-frames", type=int, help="Maximum number of selected frames to process"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start in source-video seconds; logs keep original time",
    )
    parser.add_argument(
        "--duration", type=float, help="Process this many source-video seconds"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be finite and positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    try:
        return run(args)
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ImportError,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
