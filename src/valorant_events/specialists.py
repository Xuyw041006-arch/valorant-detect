"""Optional visual specialists, independent of baseline team labels and IDs.

Association and event rules have no ML dependencies. Specialists run predict(),
never track(), so they cannot reset Ultralytics' global ByteTrack ID counter.
Identity is a property, not a new team class. Weapon events describe observed
ground-weapon presence, never pickup/drop/reload actions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

TASKS = ("agent_identity", "ground_weapon")
REJECT_IDENTITY = {
    "through_wall_silhouette",
    "unknown_agent",
    "unknown_person",
    "unknown",
}
GENERIC_CLASSES = {
    "person",
    "player",
    "enemy",
    "teammate",
    "agent",
    "weapon",
    "gun",
    "dropped_weapon",
    "held_weapon",
    "hud",
    "unknown",
    "character",
}


def _number(value, name, minimum=0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def valid_box(box):
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in box
        )
        and box[2] > box[0]
        and box[3] > box[1]
    )


def box_iou(left, right):
    if not valid_box(left) or not valid_box(right):
        return 0.0
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def match_boxes(left, right, threshold, ambiguity_margin=0.0):
    """Deterministic one-to-one greedy IoU assignment, with optional ambiguity veto.

    Near-tied alternatives on either side are rejected, not assigned identities.
    This is spatial association, not appearance re-identification.
    """
    scores = [[box_iou(a, b) for b in right] for a in left]
    candidates = []
    for i, row in enumerate(scores):
        for j, score in enumerate(row):
            if score < threshold:
                continue
            if ambiguity_margin > 0:
                alternatives = [s for k, s in enumerate(row) if k != j]
                alternatives += [r[j] for k, r in enumerate(scores) if k != i]
                if alternatives and score - max(alternatives) < ambiguity_margin:
                    continue
            candidates.append((-score, i, j))
    used_left, used_right, matches = set(), set(), []
    for negative, i, j in sorted(candidates):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            matches.append((i, j, -negative))
    return matches


def specialist_class_mapping(names, task, class_map=None):
    if task not in TASKS:
        raise ValueError("Unknown specialist task")
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Specialist model has no class names")
    if class_map is not None and not isinstance(class_map, dict):
        raise ValueError(
            "specialist class_map must be an object mapping model class names"
        )
    mapping = {}
    for class_id, raw_name in names.items():
        name = (class_map or {}).get(str(raw_name), raw_name)
        if not isinstance(name, str):
            raise ValueError("Specialist class names must be strings")
        name = re.sub(r"[\s-]+", "_", name.strip().lower())
        if task == "ground_weapon":
            if name != "dropped_weapon":
                raise ValueError(
                    "Ground-weapon weights must explicitly use dropped_weapon; generic/held/HUD gun labels are unsupported"
                )
        elif name not in REJECT_IDENTITY:
            if name.startswith("agent_"):
                name = name[6:]
            if name in GENERIC_CLASSES or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,39}", name
            ):
                raise ValueError(
                    f"Identity weights need explicit Agent names, not {raw_name!r}"
                )
        mapping[int(class_id)] = name
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Specialist classes normalize to duplicate names")
    if task == "agent_identity" and not set(mapping.values()) - REJECT_IDENTITY:
        raise ValueError("Identity specialist contains no Agent identity classes")
    return mapping


def validate_specialist_config(config):
    root = config.get("specialists", {})
    if not isinstance(root, dict) or any(key not in TASKS for key in root):
        raise ValueError(
            "specialists must contain only agent_identity and/or ground_weapon"
        )
    output = {}
    for task in TASKS:
        supplied = root.get(task, {})
        if not isinstance(supplied, dict) or not isinstance(
            supplied.get("enabled", False), bool
        ):
            raise ValueError(f"specialists.{task}.enabled must be boolean")
        if not supplied.get("enabled", False):
            continue
        cfg = deepcopy(supplied)
        cfg["mode"] = cfg.get("mode", "detector")
        if cfg["mode"] not in {"detector", "classifier"} or (
            task == "ground_weapon" and cfg["mode"] != "detector"
        ):
            raise ValueError(
                "Agent mode must be detector/classifier; ground_weapon supports detector only"
            )
        if not isinstance(cfg.get("weights"), str) or not cfg["weights"].strip():
            raise ValueError(f"Enabled {task} requires a local weights path")
        if (
            task == "ground_weapon"
            and cfg.get("label_semantics") != "ground_dropped_weapon"
        ):
            raise ValueError(
                "ground_weapon requires label_semantics='ground_dropped_weapon' verified from its training data"
            )
        if (
            task == "agent_identity"
            and cfg.get("label_semantics") != "visible_world_body_identity"
        ):
            raise ValueError(
                "agent_identity requires label_semantics='visible_world_body_identity'; HUD portraits/silhouettes are not identity evidence"
            )
        for key, default in (
            ("confidence", 0.75 if task == "agent_identity" else 0.5),
            ("iou", 0.5),
            ("match_iou", 0.4 if task == "agent_identity" else 0.25),
            ("vote_ratio", 0.75),
            ("ambiguity_margin", 0.05 if task == "agent_identity" else 0.0),
        ):
            cfg[key] = _number(cfg.get(key, default), key, 0, 1)
        for key, default in (
            ("confirmation_seconds", 0.15),
            ("max_gap_seconds", 0.35),
            ("ttl_seconds", 1.0),
            ("vote_window_seconds", 1.0),
        ):
            cfg[key] = _number(cfg.get(key, default), key)
        if (
            cfg["ttl_seconds"] <= 0
            or cfg["max_gap_seconds"] <= 0
            or cfg["vote_window_seconds"] < cfg["confirmation_seconds"]
        ):
            raise ValueError(
                "Specialist TTL/gap must be positive and vote window must cover confirmation"
            )
        cfg["min_observations"] = _integer(
            cfg.get("min_observations", 3), "min_observations"
        )
        cfg["imgsz"] = _integer(cfg.get("imgsz", 640), "imgsz", 32)
        if cfg["mode"] == "classifier" and "imgsz" not in supplied:
            cfg["imgsz"] = 224
        cfg["min_box_width"] = _number(cfg.get("min_box_width", 12), "min_box_width", 1)
        cfg["min_box_height"] = _number(
            cfg.get("min_box_height", 40), "min_box_height", 1
        )
        cfg["crop_margin"] = _number(
            cfg.get("crop_margin", 0.05), "crop_margin", 0, 0.25
        )
        cfg["classification_margin"] = _number(
            cfg.get("classification_margin", 0.15), "classification_margin", 0, 1
        )
        cfg["max_det"] = _integer(cfg.get("max_det", 50), "max_det")
        if task == "ground_weapon":
            if not isinstance(cfg.get("sliced", False), bool):
                raise ValueError("ground_weapon.sliced must be boolean")
            cfg["sliced"] = cfg.get("sliced", False)
            cfg["tile_size"] = _integer(cfg.get("tile_size", 720), "tile_size", 32)
            cfg["tile_overlap"] = _number(
                cfg.get("tile_overlap", 0.25), "tile_overlap", 0, 0.79
            )
        output[task] = cfg
    return output


class _Timed:
    def __init__(self):
        self.last_time = None
        self.reset_floor = None

    def check_time(self, timestamp, reset=False):
        timestamp = _number(timestamp, "timestamp")
        if (
            self.last_time is not None
            and (
                timestamp < self.last_time
                or (timestamp == self.last_time and not reset)
            )
        ) or (self.reset_floor is not None and timestamp < self.reset_floor):
            raise ValueError("Specialist timestamps must increase monotonically")
        if reset:
            self.reset_floor = timestamp
            self.last_time = None
        else:
            self.last_time = timestamp
        return timestamp


def _observation_valid(item, threshold):
    score = item.get("confidence")
    return (
        valid_box(item.get("bbox"))
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and threshold <= score <= 1
    )


class IdentityAssociator(_Timed):
    """Append identity evidence to observed person tracks; never alter their team.

    Known silhouettes veto identity immediately. Missing specialist matches,
    conflicts, ambiguous geometry or expired evidence yield agent_name=None.
    Unknown is an abstention; it does not introduce an 'unknown' Agent class.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.states = {}

    def reset(self, timestamp):
        self.check_time(timestamp, reset=True)
        self.states.clear()

    def update(self, timestamp, tracks, detections):
        timestamp = self.check_time(timestamp)
        cfg = self.cfg
        output = deepcopy(tracks)
        people = [
            track
            for track in output
            if track.get("label") in {"enemy", "teammate"}
            and valid_box(track.get("bbox"))
        ]
        if len({p["track_id"] for p in people}) != len(people):
            raise ValueError(
                "Identity association received duplicate baseline track IDs"
            )
        self.states = {
            key: state
            for key, state in self.states.items()
            if timestamp - state["last_seen"] <= cfg.get("ttl_seconds", 1.0)
        }
        valid = [
            item
            for item in detections
            if _observation_valid(item, cfg.get("confidence", 0.5))
        ]
        blockers = [
            item
            for item in valid
            if item.get("label") in REJECT_IDENTITY
            or item.get("rendering") == "through_wall_silhouette"
            or item.get("identity_train_eligible") is False
        ]
        allowed = []
        for track in people:
            track.update(
                agent_name=None,
                agent_confidence=0.0,
                agent_evidence={"status": "no_current_identity_match"},
            )
            small = track["bbox"][2] - track["bbox"][0] < cfg.get(
                "min_box_width", 12
            ) or track["bbox"][3] - track["bbox"][1] < cfg.get("min_box_height", 40)
            blocked = (
                small
                or track.get("rendering") == "through_wall_silhouette"
                or track.get("identity_train_eligible") is False
                or any(
                    box_iou(track["bbox"], b["bbox"]) >= cfg.get("match_iou", 0.4)
                    for b in blockers
                )
            )
            if blocked:
                self.states.pop(track["track_id"], None)
                track["agent_evidence"] = {
                    "status": (
                        "box_too_small"
                        if small
                        else "silhouette_or_ineligible_evidence"
                    )
                }
            else:
                allowed.append(track)
        candidates = [
            item
            for item in valid
            if item not in blockers
            and item.get("label") not in GENERIC_CLASSES
            and isinstance(item.get("label"), str)
            and item["label"]
        ]
        # A classification result belongs to exactly the crop/track that created
        # it, so two overlapping people cannot exchange classifier identities.
        direct = [
            (i, j, 1.0)
            for i, track in enumerate(allowed)
            for j, detection in enumerate(candidates)
            if detection.get("classification_target_id") == track["track_id"]
        ]
        if len({i for i, _, _ in direct}) != len(direct):
            raise ValueError("Duplicate classifier results for one baseline track")
        used_people = {i for i, _, _ in direct}
        remaining_people = [i for i in range(len(allowed)) if i not in used_people]
        remaining_detections = [
            j for j, d in enumerate(candidates) if "classification_target_id" not in d
        ]
        spatial = [
            (remaining_people[i], remaining_detections[j], overlap)
            for i, j, overlap in match_boxes(
                [allowed[i]["bbox"] for i in remaining_people],
                [candidates[j]["bbox"] for j in remaining_detections],
                cfg.get("match_iou", 0.4),
                cfg.get("ambiguity_margin", 0.05),
            )
        ]
        for i, j, overlap in direct + spatial:
            track, detection = allowed[i], candidates[j]
            identity = track["track_id"]
            state = self.states.get(identity)
            if state is None or timestamp - state["last_seen"] > cfg.get(
                "max_gap_seconds", 0.35
            ):
                state = self.states[identity] = {
                    "last_seen": timestamp,
                    "votes": deque(),
                }
            state["last_seen"] = timestamp
            state["votes"].append(
                (timestamp, detection["label"], float(detection["confidence"]))
            )
            while state["votes"] and timestamp - state["votes"][0][0] > cfg.get(
                "vote_window_seconds", 1.0
            ):
                state["votes"].popleft()
            votes = defaultdict(float)
            for _, label, score in state["votes"]:
                votes[label] += score
            winner = max(votes, key=lambda name: (votes[name], name))
            ratio = votes[winner] / sum(votes.values())
            evidence = [row for row in state["votes"] if row[1] == winner]
            confirmed = (
                detection["label"] == winner
                and len(evidence) >= cfg.get("min_observations", 3)
                and timestamp - evidence[0][0] + 1e-9
                >= cfg.get("confirmation_seconds", 0.15)
                and ratio >= cfg.get("vote_ratio", 0.75)
            )
            track["agent_evidence"] = {
                "status": "confirmed" if confirmed else "pending_or_conflicting_votes",
                "observations": len(evidence),
                "vote_ratio": round(ratio, 6),
                "association_iou": round(overlap, 6),
                "last_identity_observed": timestamp,
                "source": (
                    "person_crop_classifier"
                    if "classification_target_id" in detection
                    else "agent_bbox_detector"
                ),
                "team_label_unchanged": True,
            }
            if confirmed:
                track["agent_name"] = winner
                track["agent_confidence"] = round(
                    sum(v[2] for v in evidence) / len(evidence) * ratio, 6
                )
        return output


class GroundWeaponTracker(_Timed):
    """Independent namespace and IoU/TTL association; emits presence once per track."""

    def __init__(self, config):
        super().__init__()
        self.cfg, self.states, self.created_tracks = config, {}, 0

    def reset(self, timestamp):
        self.check_time(timestamp, reset=True)
        self.states.clear()  # Counter intentionally survives resets; no fake disappearance.

    def update(self, timestamp, detections, frame_index, segment_id=0):
        timestamp = self.check_time(timestamp)
        _integer(frame_index, "frame_index", 0)
        cfg = self.cfg
        self.states = {
            key: state
            for key, state in self.states.items()
            if timestamp - state["last_seen"] < cfg.get("ttl_seconds", 1.0)
        }
        detections = [
            dict(item)
            for item in detections
            if item.get("label") == "dropped_weapon"
            and _observation_valid(item, cfg.get("confidence", 0.5))
        ]
        # Duplicate overlapping boxes in one frame cannot manufacture parallel
        # weapon events or contribute multiple confirmation observations.
        unique = []
        for item in sorted(detections, key=lambda d: -d["confidence"]):
            if not any(
                box_iou(item["bbox"], previous["bbox"]) > 0.9 for previous in unique
            ):
                unique.append(item)
        detections = unique
        ids = list(self.states)
        matches = match_boxes(
            [self.states[key]["bbox"] for key in ids],
            [d["bbox"] for d in detections],
            cfg.get("match_iou", 0.25),
        )
        assigned = {j: ids[i] for i, j, _ in matches}
        tracks, events = [], []
        for j, detection in enumerate(detections):
            track_id = assigned.get(j)
            if track_id is None:
                self.created_tracks += 1
                track_id = f"weapon:{self.created_tracks}"
                self.states[track_id] = {
                    "bbox": detection["bbox"],
                    "last_seen": timestamp,
                    "observations": deque(),
                    "confirmed": False,
                }
            state = self.states[track_id]
            if timestamp - state["last_seen"] > cfg.get("max_gap_seconds", 0.35):
                state["observations"].clear()
            state["last_seen"], state["bbox"] = timestamp, detection["bbox"]
            state["observations"].append(
                (timestamp, detection["confidence"], frame_index)
            )
            while state["observations"] and timestamp - state["observations"][0][
                0
            ] > cfg.get("vote_window_seconds", 1.0):
                state["observations"].popleft()
            observations = state["observations"]
            if (
                not state["confirmed"]
                and len(observations) >= cfg.get("min_observations", 3)
                and timestamp - observations[0][0] + 1e-9
                >= cfg.get("confirmation_seconds", 0.15)
            ):
                state["confirmed"] = True
                events.append(
                    {
                        "timestamp": observations[0][0],
                        "emitted_at": timestamp,
                        "event": "weapon_detected",
                        "confidence": round(
                            sum(row[1] for row in observations) / len(observations), 6
                        ),
                        "track_id": track_id,
                        "frame_index": observations[0][2],
                        "category": "dropped_weapon",
                        "evidence": {
                            "label": "dropped_weapon",
                            "bbox": list(detection["bbox"]),
                            "observations": len(observations),
                            "semantics": "visible_ground_weapon_presence_only",
                            "tracker": "independent_iou_ttl",
                            "confidence_method": "mean_detection_score_not_calibrated_probability",
                        },
                    }
                )
            tracks.append(
                {
                    "track_id": track_id,
                    "track_namespace": "ground_weapon",
                    "label": "dropped_weapon",
                    "confidence": float(detection["confidence"]),
                    "bbox": list(detection["bbox"]),
                    "frame_index": frame_index,
                    "segment_id": segment_id,
                    "confirmed": state["confirmed"],
                }
            )
        return tracks, events

    def flush(self, timestamp):
        # EOF has no visual evidence of a disappearance or pickup.
        self.check_time(timestamp, reset=True)
        return []


def enrich_person_events(events, tracks):
    """Copy current identity evidence without retroactively relabeling old events."""
    identities = {track["track_id"]: track for track in tracks if "agent_name" in track}
    output = deepcopy(events)
    for event in output:
        if event["event"] not in {"enemy_detected", "teammate_detected"}:
            continue
        track = identities.get(event.get("track_id"))
        if track:
            event.update(
                agent_name=track["agent_name"],
                agent_confidence=track["agent_confidence"],
            )
            event.setdefault("evidence", {})["agent_identity"] = deepcopy(
                track["agent_evidence"]
            )
    return output


def _list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def crop_geometry(bbox, width, height, margin=0.05):
    """Source crop coordinates and centered square padding (left, top, side)."""
    if not valid_box(bbox):
        raise ValueError("Invalid person crop box")
    margin = _number(margin, "crop_margin", 0, 0.25)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x1, y1 = max(0, math.floor(bbox[0] - margin * bw)), max(
        0, math.floor(bbox[1] - margin * bh)
    )
    x2, y2 = min(width, math.ceil(bbox[2] + margin * bw)), min(
        height, math.ceil(bbox[3] + margin * bh)
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Person crop is outside the frame")
    side = max(x2 - x1, y2 - y1)
    return (x1, y1, x2, y2), ((side - (x2 - x1)) // 2, (side - (y2 - y1)) // 2, side)


def square_person_crop(frame, bbox, margin=0.05, size=224):
    """BGR crop, black centered padding, full square resize; no HUD lookup."""
    import cv2
    import numpy as np

    (x1, y1, x2, y2), (left, top, side) = crop_geometry(
        bbox, frame.shape[1], frame.shape[0], margin
    )
    canvas = np.zeros((side, side, 3), dtype=frame.dtype)
    canvas[top : top + y2 - y1, left : left + x2 - x1] = frame[y1:y2, x1:x2]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


class SpecialistRunner:
    """Load configured local weights; return raw observations for audit/replay."""

    def __init__(self, config, model_factory, device):
        self.configs = validate_specialist_config(config)
        self.device, self.models, self.metadata = device, {}, {}
        self.rejectors = {}
        for task, cfg in self.configs.items():
            weights = Path(cfg["weights"]).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(f"Missing local {task} weights: {weights}")
            model = model_factory(str(weights))
            expected_task = "classify" if cfg["mode"] == "classifier" else "detect"
            if getattr(model, "task", None) != expected_task:
                raise ValueError(
                    f"Specialist mode {cfg['mode']} requires model task={expected_task}"
                )
            mapping = specialist_class_mapping(model.names, task, cfg.get("class_map"))
            if cfg["mode"] == "classifier" and len(mapping) < 2:
                raise ValueError(
                    "An identity classifier needs at least two classes, including an explicit unknown class if only one Agent is known"
                )
            self.models[task] = (model, mapping)
            with weights.open("rb") as stream:
                digest = (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    if hasattr(hashlib, "file_digest")
                    else hashlib.sha256(stream.read()).hexdigest()
                )
            self.metadata[task] = {
                "weights": str(weights),
                "weights_sha256": digest,
                "class_mapping": mapping,
                "mode": cfg["mode"],
                "label_semantics": cfg["label_semantics"],
                "tracking": (
                    "baseline_iou_identity"
                    if task == "agent_identity"
                    else "independent_iou_ttl"
                ),
            }
            if task == "agent_identity" and cfg.get("rejection_adapter"):
                from .rejection import IdentityRejector

                self.rejectors[task] = IdentityRejector(
                    cfg["rejection_adapter"], digest
                )
                self.metadata[task]["rejection_adapter_sha256"] = hashlib.sha256(
                    Path(cfg["rejection_adapter"]).read_bytes()
                ).hexdigest()
                self.metadata[task]["rejection_limit"] = self.rejectors[task].metadata[
                    "limitation"
                ]

    def predict(self, frame, baseline_tracks=()):
        output = {}
        height, width = frame.shape[:2]
        for task, (model, mapping) in self.models.items():
            cfg = self.configs[task]
            if cfg["mode"] == "classifier":
                people = [
                    p
                    for p in baseline_tracks
                    if p.get("label") in {"enemy", "teammate"}
                    and valid_box(p.get("bbox"))
                    and p.get("rendering") != "through_wall_silhouette"
                    and p.get("identity_train_eligible") is not False
                    and p["bbox"][2] - p["bbox"][0] >= cfg["min_box_width"]
                    and p["bbox"][3] - p["bbox"][1] >= cfg["min_box_height"]
                ]
                crops = [
                    square_person_crop(
                        frame, p["bbox"], cfg["crop_margin"], cfg["imgsz"]
                    )
                    for p in people
                ]
                embeddings = None
                if task in self.rejectors and crops:
                    from .rejection import predict_with_embeddings

                    results, embeddings = predict_with_embeddings(
                        model,
                        crops,
                        device=self.device,
                        imgsz=cfg["imgsz"],
                        verbose=False,
                    )
                else:
                    results = (
                        model.predict(
                            crops, device=self.device, imgsz=cfg["imgsz"], verbose=False
                        )
                        if crops
                        else []
                    )
                if len(results) != len(people):
                    raise RuntimeError(
                        "Classifier results must match the submitted person crops"
                    )
                detections = []
                for result_index, (person, result) in enumerate(zip(people, results)):
                    probabilities = getattr(result, "probs", None)
                    if probabilities is None:
                        raise ValueError("Classifier result has no probabilities")
                    scores = _list(probabilities.data)
                    if len(scores) != len(mapping) or set(mapping) != set(
                        range(len(scores))
                    ):
                        raise ValueError(
                            "Classifier probability classes differ from model names"
                        )
                    if not scores or any(
                        not math.isfinite(float(s)) or not 0 <= s <= 1 for s in scores
                    ):
                        raise ValueError("Invalid classifier probabilities")
                    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                    first = ranking[0]
                    second = scores[ranking[1]] if len(ranking) > 1 else 0.0
                    if (
                        scores[first] >= cfg["confidence"]
                        and scores[first] - second >= cfg["classification_margin"]
                    ):
                        accepted, unknown_probability = (
                            self.rejectors[task].accept(embeddings[result_index])
                            if embeddings is not None
                            else (True, None)
                        )
                        detections.append(
                            {
                                "label": mapping[first] if accepted else "unknown",
                                "confidence": float(scores[first]),
                                "bbox": list(person["bbox"]),
                                "classification_target_id": person["track_id"],
                                "class_probability_margin": float(
                                    scores[first] - second
                                ),
                                "rejection_probability": unknown_probability,
                                "raw_predicted_agent": mapping[first],
                            }
                        )
                output[task] = detections
                continue
            if task == "ground_weapon" and cfg.get("sliced", False):
                from .sliced import predict_sliced

                output[task] = predict_sliced(
                    model,
                    frame,
                    mapping,
                    device=self.device,
                    imgsz=cfg["imgsz"],
                    confidence=cfg["confidence"],
                    iou=cfg["iou"],
                    max_det=cfg["max_det"],
                    tile_size=cfg["tile_size"],
                    overlap=cfg["tile_overlap"],
                )
                continue
            results = model.predict(
                frame,
                device=self.device,
                imgsz=cfg["imgsz"],
                conf=cfg["confidence"],
                iou=cfg["iou"],
                max_det=cfg["max_det"],
                verbose=False,
            )
            if len(results) != 1:
                raise RuntimeError("Expected one specialist result for one frame")
            boxes, detections = getattr(results[0], "boxes", None), []
            if boxes is not None:
                classes, scores, coords = (
                    _list(boxes.cls),
                    _list(boxes.conf),
                    _list(boxes.xyxy),
                )
                if not len(classes) == len(scores) == len(coords):
                    raise ValueError("Inconsistent specialist output lengths")
                for class_id, score, bbox in zip(classes, scores, coords):
                    if int(class_id) not in mapping:
                        raise ValueError("Unmapped specialist class ID")
                    if not valid_box(bbox):
                        continue
                    bbox = [
                        max(0.0, min(float(width), bbox[0])),
                        max(0.0, min(float(height), bbox[1])),
                        max(0.0, min(float(width), bbox[2])),
                        max(0.0, min(float(height), bbox[3])),
                    ]
                    item = {
                        "label": mapping[int(class_id)],
                        "confidence": float(score),
                        "bbox": bbox,
                    }
                    if _observation_valid(item, cfg["confidence"]):
                        detections.append(item)
            output[task] = detections
        return output
