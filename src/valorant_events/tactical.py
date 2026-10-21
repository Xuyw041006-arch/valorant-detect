"""Opt-in, calibrated HUD evidence; never a reconstruction of hidden world state.

Minimap color candidates have local image coordinates and short-lived IDs, not
world coordinates/player identities. HUD event times are first observations,
not proven action times. Image/OCR libraries are lazy, and OCR can be disabled.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import re


def _number(value, name, low=0, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return float(value)


def _roi(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be normalized xyxy")
    value = [_number(v, name, 0, 1) for v in value]
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError(f"{name} must have positive area")
    return value


def _pixels(roi, shape):
    h, w = shape[:2]
    return [
        int(roi[0] * w),
        int(roi[1] * h),
        min(w, math.ceil(roi[2] * w)),
        min(h, math.ceil(roi[3] * h)),
    ]


def _crop(frame, box):
    return frame[box[1] : box[3], box[0] : box[2]]


def _mask(hsv, ranges):
    import cv2
    import numpy as np

    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        out |= cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
    return out


RED = [([0, 110, 110], [10, 255, 255]), ([170, 110, 110], [179, 255, 255])]
TEAL = [([65, 80, 100], [95, 255, 255])]
# UI bars are translucent and less saturated than minimap markers.
BAR_RED = [([0, 70, 90], [12, 255, 255]), ([170, 70, 90], [179, 255, 255])]
BAR_TEAL = [([65, 45, 90], [95, 255, 255])]


class EvidenceGate:
    """Confirm fresh observations and suppress a persistent/repositioned row.

    Each key can count once per update. Missing evidence clears an unconfirmed
    candidate. A confirmed key rearms only after absence for dedupe_seconds.
    """

    def __init__(
        self,
        minimum=2,
        seconds=0.15,
        max_gap=0.75,
        dedupe_seconds=10,
        missing_grace=None,
    ):
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("min_observations must be a positive integer")
        self.minimum = minimum
        self.seconds = _number(seconds, "confirmation_seconds")
        self.max_gap = _number(max_gap, "max_gap_seconds", 1e-9)
        self.dedupe = _number(dedupe_seconds, "dedupe_seconds", 1e-9)
        self.missing_grace = {} if missing_grace is None else missing_grace
        if not isinstance(self.missing_grace, dict):
            raise ValueError("missing_grace must map observation keys to seconds")
        for value in self.missing_grace.values():
            _number(value, "missing observation grace", 0, self.max_gap)
        self.states = {}

    def clear(self):
        self.states.clear()

    def update(self, timestamp, observations):
        chosen = {}
        for item in observations:
            key = item["key"]
            if key not in chosen or item["confidence"] > chosen[key]["confidence"]:
                chosen[key] = item
        for key, state in list(self.states.items()):
            grace = self.missing_grace.get(key, 0)
            expired = (
                timestamp - state["last"] > self.dedupe
                if state["fired"]
                else grace == 0 or timestamp - state["last"] > grace
            )
            if key not in chosen and expired:
                del self.states[key]
        output = []
        for key, item in chosen.items():
            state = self.states.get(key)
            if state is not None and timestamp - state["last"] > (
                self.dedupe if state["fired"] else self.max_gap
            ):
                state = None
            if state is None:
                state = self.states[key] = dict(
                    first=timestamp, last=timestamp, count=0, scores=[], fired=False
                )
            state["last"] = timestamp
            state["count"] += 1
            state["scores"].append(item["confidence"])
            if (
                not state["fired"]
                and state["count"] >= self.minimum
                and timestamp - state["first"] + 1e-9 >= self.seconds
            ):
                state["fired"] = True
                output.append(
                    {
                        "timestamp": state["first"],
                        "emitted_at": timestamp,
                        "event": item["event"],
                        "confidence": round(
                            sum(state["scores"]) / len(state["scores"]), 6
                        ),
                        "evidence": {
                            **item["evidence"],
                            "observations": state["count"],
                            "occurrence_time_unknown": True,
                            "confidence_method": "HUD_evidence_score_not_calibrated_probability",
                        },
                    }
                )
                if self.missing_grace.get(key, 0):
                    output[-1]["evidence"]["missing_observation_grace_seconds"] = (
                        self.missing_grace[key]
                    )
            # A persistent UI row must not grow memory without bound.
            if state["fired"]:
                state["scores"] = state["scores"][-self.minimum :]
        return output


MINIMAP_CLASSES = (
    "player_marker",
    "ally_death_marker",
    "enemy_death_marker",
    "spike_marker",
    "smoke_marker",
    "enemy_unknown_marker",
)


def _tensor_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class MinimapDetector:
    """Optional six-class symbol detector; team colors require profile evidence.

    No model import/download occurs until enabled read. Uses predict(), never
    track(). Model labels describe visible UI symbols, not world life state.
    """

    def __init__(self, config, model_factory=None, device=None):
        self.cfg = config or {}
        if not isinstance(self.cfg, dict) or not isinstance(
            self.cfg.get("enabled", False), bool
        ):
            raise ValueError("minimap.detector must have boolean enabled")
        self.enabled = self.cfg.get("enabled", False)
        self.factory, self.device, self.model, self.mapping = (
            model_factory,
            device,
            None,
            None,
        )
        self.metadata = {"enabled": self.enabled}
        if not self.enabled:
            return
        if (
            not isinstance(self.cfg.get("weights"), str)
            or not self.cfg["weights"].strip()
        ):
            raise ValueError("Enabled minimap detector needs local weights")
        self.confidence = _number(
            self.cfg.get("confidence", 0.35), "minimap detector confidence", 0, 1
        )
        self.iou = _number(self.cfg.get("iou", 0.5), "minimap detector iou", 0, 1)
        for key, default in [("imgsz", 768), ("max_det", 80)]:
            value = self.cfg.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Invalid minimap detector " + key)
        if not isinstance(self.cfg.get("team_colors_calibrated", False), bool):
            raise ValueError("team_colors_calibrated must be boolean")
        self.colors = self.cfg.get("team_colors", {})
        if not isinstance(self.colors, dict) or set(self.colors) - {
            "enemy",
            "teammate",
        }:
            raise ValueError(
                "team_colors must map enemy/teammate to calibrated HSV ranges"
            )
        for ranges in self.colors.values():
            if not isinstance(ranges, list) or not ranges:
                raise ValueError("HSV ranges must be nonempty lists")
            for pair in ranges:
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or any(
                        not isinstance(v, (list, tuple)) or len(v) != 3 for v in pair
                    )
                ):
                    raise ValueError("HSV range needs lower/upper triples")
                for i, maximum in enumerate([179, 255, 255]):
                    low = _number(pair[0][i], "HSV lower", 0, maximum)
                    high = _number(pair[1][i], "HSV upper", 0, maximum)
                    if low > high:
                        raise ValueError("HSV lower exceeds upper")
        self.ring_fraction = _number(
            self.cfg.get("edge_ring_fraction", 0.20), "edge_ring_fraction", 0.05, 0.45
        )
        self.min_fraction = _number(
            self.cfg.get("min_edge_color_fraction", 0.15),
            "min_edge_color_fraction",
            0,
            1,
        )
        self.color_margin = _number(
            self.cfg.get("edge_color_margin", 0.08), "edge_color_margin", 0, 1
        )
        self.min_sides = self.cfg.get("min_colored_sides", 2)
        if (
            isinstance(self.min_sides, bool)
            or not isinstance(self.min_sides, int)
            or not 1 <= self.min_sides <= 4
        ):
            raise ValueError("min_colored_sides must be 1..4")

    def _load(self):
        import hashlib

        if self.model is not None:
            return
        path = Path(self.cfg["weights"])
        if not path.is_file():
            raise FileNotFoundError(
                "Minimap weights must be an existing local file: " + str(path)
            )
        factory = self.factory
        if factory is None:
            try:
                factory = importlib.import_module("ultralytics").YOLO
            except ImportError as exc:
                raise RuntimeError("Minimap detector requires Ultralytics") from exc
        self.model = factory(str(path))
        if getattr(self.model, "task", "detect") != "detect":
            raise ValueError("Minimap weights must be a detection model")
        names = self.model.names
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        if not isinstance(names, dict):
            raise ValueError("Minimap model has no class names")
        supplied = self.cfg.get("class_map", {})
        if not isinstance(supplied, dict):
            raise ValueError("Minimap class_map must be an object")
        mapping = {int(k): supplied.get(str(v), str(v)) for k, v in names.items()}
        if set(mapping.values()) != set(MINIMAP_CLASSES) or len(mapping) != len(
            MINIMAP_CLASSES
        ):
            raise ValueError(
                "Minimap model must map one-to-one to the six documented UI classes"
            )
        self.mapping = mapping
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.metadata = {
            "enabled": True,
            "weights_sha256": digest.hexdigest(),
            "classes": list(MINIMAP_CLASSES),
            "imgsz": self.cfg.get("imgsz", 768),
            "confidence": self.confidence,
            "iou": self.iou,
            "team_colors_calibrated": self.cfg.get("team_colors_calibrated", False),
        }

    def _team_from_edge(self, patch):
        import cv2
        import numpy as np

        if not self.cfg.get("team_colors_calibrated", False) or not self.colors:
            return "unknown", {"status": "no_calibrated_color_profile"}
        h, w = patch.shape[:2]
        if min(h, w) < 5:
            return "unknown", {"status": "marker_too_small_for_edge_evidence"}
        dy = max(1, round(h * self.ring_fraction))
        dx = max(1, round(w * self.ring_fraction))
        ring = np.ones((h, w), bool)
        ring[dy : h - dy, dx : w - dx] = False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        fractions = {}
        sides = {}
        for team, ranges in self.colors.items():
            mask = _mask(hsv, ranges) > 0
            fractions[team] = float(
                np.count_nonzero(mask & ring) / max(1, np.count_nonzero(ring))
            )
            bands = [mask[:dy], mask[h - dy :], mask[:, :dx], mask[:, w - dx :]]
            sides[team] = sum(float(b.mean()) >= self.min_fraction for b in bands)
        order = sorted(fractions, key=lambda k: -fractions[k])
        winner = order[0]
        competitor = max((fractions[k] for k in order[1:]), default=0)
        accepted = (
            fractions[winner] >= self.min_fraction
            and fractions[winner] - competitor >= self.color_margin
            and sides[winner] >= self.min_sides
        )
        return (winner if accepted else "unknown"), {
            "status": (
                "calibrated_edge_color" if accepted else "ambiguous_or_weak_edge_color"
            ),
            "fractions": fractions,
            "colored_sides": sides,
            "white_is_not_team_evidence": True,
        }

    def predict(self, crop, roi):
        self._load()
        kwargs = dict(
            imgsz=self.cfg.get("imgsz", 768),
            conf=self.confidence,
            iou=self.iou,
            max_det=self.cfg.get("max_det", 80),
            verbose=False,
        )
        device = self.cfg.get("device", self.device)
        if device is not None:
            kwargs["device"] = device
        results = self.model.predict(crop, **kwargs)
        if len(results) != 1:
            raise RuntimeError("Expected one minimap ROI result")
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        xyxy, conf, classes = map(_tensor_list, [boxes.xyxy, boxes.conf, boxes.cls])
        if not len(xyxy) == len(conf) == len(classes):
            raise ValueError("Inconsistent minimap prediction arrays")
        out = []
        h, w = crop.shape[:2]
        for box, score, class_id in zip(xyxy, conf, classes):
            try:
                vals = [float(v) for v in box]
                score = float(score)
                class_id = float(class_id)
                if (
                    len(vals) != 4
                    or not all(math.isfinite(v) for v in vals + [score, class_id])
                    or not class_id.is_integer()
                    or score < self.confidence
                    or score > 1
                ):
                    continue
                label = self.mapping[int(class_id)]
            except (ValueError, TypeError, KeyError):
                continue
            x1, y1 = max(0, vals[0]), max(0, vals[1])
            x2, y2 = min(w, vals[2]), min(h, vals[3])
            if x2 <= x1 or y2 <= y1:
                continue
            team = "unknown"
            kind = label
            edge = None
            if label == "player_marker":
                patch = crop[int(y1) : math.ceil(y2), int(x1) : math.ceil(x2)]
                team, edge = self._team_from_edge(patch)
            elif label in {"ally_death_marker", "enemy_death_marker"}:
                team = "teammate" if label == "ally_death_marker" else "enemy"
                kind = "death_marker"
            elif label == "enemy_unknown_marker":
                team = "enemy"
                kind = "last_known_enemy_marker"
            position = [(x1 + x2) / (2 * w), (y1 + y2) / (2 * h)]
            out.append(
                {
                    "team_hint": team,
                    "marker_type": kind,
                    "detector_label": label,
                    "position": position,
                    "normalized_roi": position,
                    "bbox": [
                        math.floor(x1) + roi[0],
                        math.floor(y1) + roi[1],
                        math.ceil(x2) + roi[0],
                        math.ceil(y2) + roi[1],
                    ],
                    "confidence": score,
                    "identity": None,
                    "verified_player": False,
                    "verified_marker_semantics": True,
                    "marker_visible": True,
                    "current_visible": label == "player_marker",
                    "count_as_current_player": label == "player_marker",
                    "count_as_alive": False,
                    "alive": False if kind == "death_marker" else None,
                    "world_visibility": "unknown",
                    "spike_state": "unknown",
                    "team_color_evidence": edge,
                    "evidence_method": "minimap_YOLO_symbol_detector_with_optional_calibrated_edge_color",
                    "classification_status": "model_prediction_not_human_verification",
                    "position_semantics": (
                        "last_known_candidate_not_current_location"
                        if kind == "last_known_enemy_marker"
                        else "visible_minimap_symbol_not_world_position"
                    ),
                }
            )
        return out


class TacticalReader:
    """read(BGR frame, strictly increasing timestamp) -> JSON-safe observation.

    Accepts a tactical mapping or a full config containing ``tactical``. Default
    disabled. OCR reader injection permits sharing an existing EasyOCR Reader.
    No cached reading is counted as a fresh observation. reset creates no events.
    See configs/tactical.json for the calibrated English 1920x1080 example.
    Optional minimap.detector runs the six-class symbol model on the configured
    ROI only; it supersedes templates and color-only candidates. All returned
    coordinates stay in the original frame / normalized ROI. Current player-
    marker counts do not establish alive people or world visibility. OCR can
    upscale globally or by field, with original-ROI output coordinates.
    """

    def __init__(
        self, config=None, ocr_reader=None, gpu=False, model_factory=None, device=None
    ):
        from .killfeed_weapon import KillfeedWeaponReader, WeaponConsensus

        cfg = (config or {}).get("tactical", config or {})
        if not isinstance(cfg, dict):
            raise ValueError("tactical must be an object")
        self.enabled = bool(cfg.get("enabled", False))
        self.cfg = cfg
        self.map_cfg = cfg.get("minimap", {})
        self.ui_cfg = cfg.get("ui", {})
        self.kill_weapon_reader = KillfeedWeaponReader(
            self.ui_cfg.get("weapon_recognition", {})
        )
        self.kill_weapon_consensus = WeaponConsensus()
        if not isinstance(self.map_cfg, dict) or not isinstance(self.ui_cfg, dict):
            raise ValueError("minimap/ui must be objects")
        self.map_enabled = self.enabled and bool(self.map_cfg.get("enabled", True))
        self.ui_enabled = self.enabled and bool(self.ui_cfg.get("enabled", True))
        self.ocr_enabled = bool(self.ui_cfg.get("ocr_enabled", True))
        self.map_detector = MinimapDetector(
            self.map_cfg.get("detector", {}), model_factory=model_factory, device=device
        )
        self.model_template_fallback = self.map_cfg.get("model_template_fallback", [])
        if not isinstance(self.model_template_fallback, list) or any(
            value != "death_marker" for value in self.model_template_fallback
        ):
            raise ValueError(
                "model_template_fallback currently supports only death_marker"
            )
        self.rois = {
            "minimap": _roi(
                self.map_cfg.get("roi", [0.008, 0.045, 0.231, 0.435]), "minimap.roi"
            ),
            "spike": _roi(
                self.ui_cfg.get(
                    "spike_roi", [920 / 1920, 17 / 1080, 1000 / 1920, 92 / 1080]
                ),
                "spike_roi",
            ),
            "banner": _roi(
                self.ui_cfg.get("banner_roi", [0.425, 0.22, 0.575, 0.265]), "banner_roi"
            ),
            "spike_text": _roi(
                self.ui_cfg.get("spike_text_roi", [0.40, 0.10, 0.60, 0.28]),
                "spike_text_roi",
            ),
            "planting": _roi(
                self.ui_cfg.get("planting_roi", [0.3, 0.10, 0.7, 0.25]), "planting_roi"
            ),
            "death": _roi(
                self.ui_cfg.get("death_roi", [0.80, 0.28, 0.99, 0.325]), "death_roi"
            ),
            "killfeed": _roi(
                self.ui_cfg.get("killfeed_roi", [0.77, 0.08, 0.995, 0.255]),
                "killfeed_roi",
            ),
        }
        self.min_confidence = _number(
            self.ui_cfg.get("min_ocr_confidence", 0.6), "min_ocr_confidence", 0, 1
        )
        self.ocr_scale = _number(self.ui_cfg.get("ocr_scale", 1), "ocr_scale", 0.5, 4)
        self.ocr_scales = self.ui_cfg.get("ocr_scales", {})
        self.ocr_methods = self.ui_cfg.get("recognition_methods", {})
        self.ocr_allowlists = self.ui_cfg.get("ocr_allowlists", {})
        fields = {"banner", "death", "killfeed", "spike_text", "planting"}
        for mapping in (self.ocr_scales, self.ocr_methods, self.ocr_allowlists):
            if not isinstance(mapping, dict) or set(mapping) - fields:
                raise ValueError("Unknown tactical OCR field configuration")
        for scale in self.ocr_scales.values():
            _number(scale, "field OCR scale", 0.5, 4)
        if any(
            method not in {"detect", "recognize"}
            for method in self.ocr_methods.values()
        ):
            raise ValueError("Tactical OCR method must be detect/recognize")
        if any(
            not isinstance(value, str) or not value
            for value in self.ocr_allowlists.values()
        ):
            raise ValueError("OCR allowlist must be a nonempty string")
        self.template_threshold = _number(
            self.ui_cfg.get("spike_match_threshold", 0.75),
            "spike_match_threshold",
            0,
            1,
        )
        self.spike_match_method = self.ui_cfg.get("spike_match_method", "roi_iou")
        if self.spike_match_method not in {"roi_iou", "local_mask_iou"}:
            raise ValueError("spike_match_method must be roi_iou or local_mask_iou")
        self.spike_reference_height = _number(
            self.ui_cfg.get("spike_template_reference_height", 1080),
            "spike_template_reference_height",
            1,
        )
        scales = self.ui_cfg.get("spike_search_scales", [0.8, 0.9, 1, 1.1, 1.2])
        if not isinstance(scales, list) or not scales or len(scales) > 20:
            raise ValueError("spike_search_scales must contain 1..20 scales")
        self.spike_search_scales = [
            _number(value, "spike search scale", 0.25, 4) for value in scales
        ]
        self.spike_min_pixels = _number(
            self.ui_cfg.get("spike_search_min_red_pixels", 20),
            "spike_search_min_red_pixels",
            1,
        )
        self.spike_min_side = _number(
            self.ui_cfg.get("spike_search_min_side", 12), "spike_search_min_side", 3
        )
        self.distance = _number(
            self.map_cfg.get("association_distance", 0.055),
            "association_distance",
            1e-9,
            1,
        )
        self.ttl = _number(self.map_cfg.get("ttl_seconds", 0.6), "ttl_seconds", 1e-9)
        self.min_area = _number(
            self.map_cfg.get("min_area_at_1080", 5), "min_area_at_1080", 1
        )
        self.max_area = _number(
            self.map_cfg.get("max_area_at_1080", 220), "max_area_at_1080", self.min_area
        )
        self.max_side = _number(
            self.map_cfg.get("max_side_at_1080", 27), "max_side_at_1080", 1
        )
        self.team_mapping = self.ui_cfg.get(
            "victim_color_to_team", {"red": "enemy", "teal": "teammate"}
        )
        if (
            not isinstance(self.team_mapping, dict)
            or set(self.team_mapping) - {"red", "teal"}
            or any(
                v not in {"enemy", "teammate", "unknown"}
                for v in self.team_mapping.values()
            )
        ):
            raise ValueError(
                "victim_color_to_team maps red/teal to enemy/teammate/unknown"
            )
        self.phrases = self.ui_cfg.get("phrases", {})
        if not isinstance(self.phrases, dict) or set(self.phrases) - {
            "defused",
            "spike_down",
            "carrier_killed",
            "death_panel",
            "planting",
        }:
            raise ValueError("Unsupported tactical UI phrase mapping")
        for values in self.phrases.values():
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(v, str) or len(v.strip()) < 3 for v in values)
            ):
                raise ValueError(
                    "Tactical phrases must be nonempty lists of explicit phrases"
                )
        self.whitespace_phrase_keys = self.ui_cfg.get(
            "phrase_ignore_intercharacter_whitespace", []
        )
        if not isinstance(self.whitespace_phrase_keys, list) or any(
            not isinstance(value, str)
            or value
            not in {
                "defused",
                "spike_down",
                "carrier_killed",
                "death_panel",
                "planting",
            }
            for value in self.whitespace_phrase_keys
        ):
            raise ValueError("Unknown phrase_ignore_intercharacter_whitespace key")
        if cfg.get("calibration_aspect_ratio") is not None:
            _number(
                cfg["calibration_aspect_ratio"], "calibration_aspect_ratio", 0.1, 10
            )
        for name, default, high in (
            ("max_rotation_degrees", 2, 180),
            ("max_scale_change", 0.035, 2),
            ("max_translation_roi", 0.025, 1),
            ("max_unresolved_gray_difference", 0.12, 1),
            ("min_gray_surround", 0.25, 1),
            ("template_ambiguity_margin", 0.03, 1),
        ):
            _number(self.map_cfg.get(name, default), name, 0, high)
        for name, default in (
            ("killfeed_row_pitch_at_1080", 39),
            ("killfeed_row_height_at_1080", 34),
        ):
            _number(self.ui_cfg.get(name, default), name, 1, 200)
        _number(
            self.ui_cfg.get("killfeed_row_offset_at_1080", 0),
            "killfeed_row_offset_at_1080",
            0,
            200,
        )
        self.gate = EvidenceGate(
            self.ui_cfg.get("min_observations", 2),
            self.ui_cfg.get("confirmation_seconds", 0.15),
            self.ui_cfg.get("max_gap_seconds", 0.75),
            self.ui_cfg.get("dedupe_seconds", 10),
            missing_grace={
                "banner": self.ui_cfg.get("banner_missing_grace_seconds", 0)
            },
        )
        self._reader = ocr_reader
        self.gpu = gpu
        self._template = None
        self._map_templates = None
        self.mode = self.map_cfg.get("mode", "unknown")
        self.coverage = self.map_cfg.get("coverage", "unknown")
        if self.mode not in {"fixed", "rotating", "unknown"} or self.coverage not in {
            "full",
            "partial",
            "unknown",
        }:
            raise ValueError("minimap mode/coverage must be explicit supported values")
        self._previous_map = None
        self._view_change = {"reset": False, "reason": "first_sample"}
        self._tracks = {}
        self._next_id = 1
        self._last_time = None
        self._floor = None
        self._first_time = None
        self._spike_candidate = None
        self._initial_killfeed_keys = {}

    def reset(self, timestamp=None, reason=None):
        if timestamp is not None:
            timestamp = _number(timestamp, "reset timestamp")
            if self._last_time is not None and timestamp < self._last_time:
                raise ValueError("Reset must not move backward")
        self._floor = timestamp
        self._last_time = None
        self._first_time = None
        self._spike_candidate = None
        self._initial_killfeed_keys.clear()
        self.kill_weapon_consensus.reset()
        self._tracks.clear()
        self._previous_map = None
        self.gate.clear()
        return []

    def _confirm_spike_state(self, timestamp, state):
        """Confirm the current uninterrupted state, independently of dedupe.

        A fired event may remain remembered through a brief UI dropout. That
        history cannot establish current state after a missing, conflicting,
        opposite, or temporally distant observation.
        """
        if state not in {"planted", "dropped", "defused"}:
            self._spike_candidate = None
            return False
        previous = self._spike_candidate
        if (
            previous is None
            or previous["state"] != state
            or timestamp - previous["last"] > self.gate.max_gap
        ):
            previous = self._spike_candidate = {
                "state": state,
                "first": timestamp,
                "last": timestamp,
                "count": 0,
            }
        previous["last"] = timestamp
        previous["count"] += 1
        return (
            previous["count"] >= self.gate.minimum
            and timestamp - previous["first"] + 1e-9 >= self.gate.seconds
        )

    def pixel_rois(self, frame_shape):
        return {key: _pixels(roi, frame_shape) for key, roi in self.rois.items()}

    def _phrase(self, text, key, default):
        for phrase in self.phrases.get(key, [default]):
            pattern = (
                r"(?<!\w)"
                + r"\s+".join(re.escape(w) for w in phrase.upper().split())
                + r"(?!\w)"
            )
            if re.search(pattern, text.upper()):
                return True
            if key in self.whitespace_phrase_keys:
                # Ignore spacing only. Never rewrite OCR symbols/numbers as
                # letters, and retain boundaries against longer words.
                letters = re.sub(r"\s+", "", phrase.upper())
                pattern = (
                    r"(?<!\w)"
                    + r"\s*".join(re.escape(char) for char in letters)
                    + r"(?!\w)"
                )
                if re.search(pattern, text.upper()):
                    return True
        return False

    def _ocr(self, image, field=None):
        if not self.ocr_enabled:
            return []
        if self._reader is None:
            try:
                easyocr = importlib.import_module("easyocr")
            except ImportError as exc:
                raise RuntimeError(
                    "Tactical OCR needs EasyOCR or ui.ocr_enabled=false"
                ) from exc
            kw = dict(
                gpu=self.gpu,
                download_enabled=bool(self.ui_cfg.get("download_enabled", False)),
                verbose=False,
            )
            if self.ui_cfg.get("model_storage_directory"):
                kw["model_storage_directory"] = self.ui_cfg["model_storage_directory"]
            self._reader = easyocr.Reader(self.ui_cfg.get("languages", ["en"]), **kw)
        scale = self.ocr_scales.get(field, self.ocr_scale)
        original_height, original_width = image.shape[:2]
        if scale != 1:
            import cv2

            image = cv2.resize(
                image,
                (
                    max(1, round(original_width * scale)),
                    max(1, round(original_height * scale)),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        # Account for integer output dimensions rather than dividing by a nominal
        # factor, especially on narrow crops. Returned boxes stay in original ROI.
        sx, sy = image.shape[1] / original_width, image.shape[0] / original_height
        kwargs = dict(detail=1, paragraph=False)
        if field in self.ocr_allowlists:
            kwargs["allowlist"] = self.ocr_allowlists[field]
        if self.ocr_methods.get(field, "detect") == "recognize":
            rows = self._reader.recognize(
                image,
                horizontal_list=None,
                free_list=None,
                decoder="greedy",
                batch_size=1,
                workers=0,
                **kwargs,
            )
        else:
            rows = self._reader.readtext(image, **kwargs)
        good = []
        for row in rows or []:
            try:
                box, text, confidence = row[:3]
                confidence = float(confidence)
                coordinates = [[float(x) / sx, float(y) / sy] for x, y in box]
                valid = len(coordinates) == 4 and all(
                    math.isfinite(v) for xy in coordinates for v in xy
                )
                if (
                    valid
                    and math.isfinite(confidence)
                    and self.min_confidence <= confidence <= 1
                    and str(text).strip()
                ):
                    good.append(
                        {
                            "text": str(text).strip(),
                            "confidence": confidence,
                            "bbox": coordinates,
                        }
                    )
            except (TypeError, ValueError, IndexError):
                continue
        return good

    def _check_map_view(self, crop):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        previous = self._previous_map
        self._previous_map = gray
        info = {"reset": False, "reason": "first_sample"}
        if self.mode == "unknown":
            info = {
                "reset": True,
                "reason": "unknown_projection_no_identity_continuity",
            }
        elif previous is not None:
            if previous.shape != gray.shape:
                info = {"reset": True, "reason": "roi_resolution_changed"}
            else:
                difference = float(np.mean(cv2.absdiff(previous, gray)) / 255)
                info = {
                    "reset": False,
                    "reason": "stable_or_unresolved",
                    "gray_difference": difference,
                }
                map_mask = (
                    (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 65) & (hsv[:, :, 2] < 225)
                ).astype(np.uint8) * 255
                pts = cv2.goodFeaturesToTrack(
                    previous,
                    maxCorners=80,
                    qualityLevel=0.02,
                    minDistance=7,
                    mask=map_mask,
                )
                matrix = None
                if pts is not None and len(pts) >= 8:
                    nxt, status, error = cv2.calcOpticalFlowPyrLK(
                        previous, gray, pts, None
                    )
                    if nxt is not None and status is not None:
                        good = status.reshape(-1) > 0
                        if int(good.sum()) >= 8:
                            matrix, inliers = cv2.estimateAffinePartial2D(
                                pts[good],
                                nxt[good],
                                method=cv2.RANSAC,
                                ransacReprojThreshold=2,
                            )
                            if inliers is None or float(inliers.mean()) < 0.65:
                                matrix = None
                if matrix is not None and np.isfinite(matrix).all():
                    a, b = float(matrix[0, 0]), float(matrix[1, 0])
                    scale = math.hypot(a, b)
                    angle = math.degrees(math.atan2(b, a))
                    shift = math.hypot(
                        float(matrix[0, 2]) / gray.shape[1],
                        float(matrix[1, 2]) / gray.shape[0],
                    )
                    changed = (
                        abs(angle) > self.map_cfg.get("max_rotation_degrees", 2)
                        or abs(scale - 1) > self.map_cfg.get("max_scale_change", 0.035)
                        or shift > self.map_cfg.get("max_translation_roi", 0.025)
                    )
                    info.update(
                        reset=bool(changed),
                        reason="map_transform" if changed else "stable_transform",
                        rotation_degrees=angle,
                        scale=scale,
                        translation_roi=shift,
                    )
                elif difference > self.map_cfg.get(
                    "max_unresolved_gray_difference", 0.12
                ):
                    info.update(reset=True, reason="unresolved_visual_change")
        if info["reset"]:
            self._tracks.clear()
        self._view_change = info

    def _minimap(self, frame, timestamp, roi):
        import cv2
        import numpy as np

        crop = _crop(frame, roi)
        self._check_map_view(crop)
        if self.map_detector.enabled:
            detections = self.map_detector.predict(crop, roi)
            if self.model_template_fallback:

                def overlaps_model(candidate):
                    a = candidate["bbox"]
                    for detection in detections:
                        b = detection["bbox"]
                        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                            0, min(a[3], b[3]) - max(a[1], b[1])
                        )
                        smaller = min(
                            (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
                        )
                        if smaller > 0 and intersection / smaller > 0.35:
                            return True
                    return False

                for candidate in self._match_markers(crop, roi):
                    if candidate[
                        "marker_type"
                    ] not in self.model_template_fallback or overlaps_model(candidate):
                        continue
                    candidate.update(
                        marker_visible=True,
                        current_visible=False,
                        count_as_current_player=False,
                        count_as_alive=False,
                        alive=False,
                        world_visibility="unknown",
                        classification_status="profile_template_fallback_not_world_death_confirmation",
                        position_semantics="visible_minimap_death_symbol_not_world_position",
                        evidence_method="calibrated_death_template_without_overlapping_model_detection",
                    )
                    detections.append(candidate)
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            scale = frame.shape[0] / 1080
            detections = []
            for side, ranges in (("unknown", RED), ("teammate", TEAL)):
                mask = _mask(hsv, ranges)
                for excluded in self.map_cfg.get("exclude_rois", []):
                    b = _pixels(_roi(excluded, "exclude_rois"), crop.shape)
                    mask[b[1] : b[3], b[0] : b[2]] = 0
                # Only bridge antialiasing holes; broad dilation merges close people.
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
                )
                n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                for i in range(1, n):
                    x, y, w, h, area = [int(v) for v in stats[i]]
                    if (
                        not self.min_area * scale**2 <= area <= self.max_area * scale**2
                        or max(w, h) > self.max_side * scale
                        or min(w, h) < max(2, 2 * scale)
                    ):
                        continue
                    # Grey map geometry near the symbol excludes scenery leaking
                    # through transparent/outside-map areas. Still not a classifier.
                    if (side == "teammate" and min(w, h) < 7 * scale) or (
                        side == "unknown" and min(w, h) < 3 * scale
                    ):
                        continue
                    pad = max(2, round(8 * scale))
                    surround = hsv[
                        max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad
                    ]
                    grey_fraction = float(
                        (
                            (surround[:, :, 1] < 40)
                            & (surround[:, :, 2] > 65)
                            & (surround[:, :, 2] < 220)
                        ).mean()
                    )
                    if grey_fraction < self.map_cfg.get("min_gray_surround", 0.25):
                        continue
                    # Red may be a danger ping, last-known marker, or enemy icon.
                    # It is never promoted to a known enemy from color alone.
                    center = [
                        float(centroids[i][0]) / crop.shape[1],
                        float(centroids[i][1]) / crop.shape[0],
                    ]
                    detections.append(
                        {
                            "team_hint": side,
                            "position": center,
                            "bbox": [
                                x + roi[0],
                                y + roi[1],
                                x + w + roi[0],
                                y + h + roi[1],
                            ],
                            "normalized_roi": center,
                            "color_pixels": area,
                            "marker_type": (
                                "red_marker_unknown"
                                if side == "unknown"
                                else "teammate_color_candidate"
                            ),
                            "identity": None,
                            "verified_player": False,
                            "gray_surround_fraction": grey_fraction,
                            "confidence": round(
                                min(0.9, 0.5 + 0.4 * area / max(w * h, 1)), 6
                            ),
                        }
                    )
            templates = self._match_markers(crop, roi)
            # Semantic template evidence takes precedence over fragments of its ring.
            detections = [
                d
                for d in detections
                if not any(
                    t["bbox"][0] - 3
                    <= (d["bbox"][0] + d["bbox"][2]) / 2
                    <= t["bbox"][2] + 3
                    and t["bbox"][1] - 3
                    <= (d["bbox"][1] + d["bbox"][3]) / 2
                    <= t["bbox"][3] + 3
                    for t in templates
                )
            ] + templates
        self._tracks = {
            k: v
            for k, v in self._tracks.items()
            if timestamp - v["last_seen"] <= self.ttl
        }
        possible = []
        for i, det in enumerate(detections):
            for key, state in self._tracks.items():
                d = math.dist(det["position"], state["position"])
                if (
                    state["team_hint"] == det["team_hint"]
                    and state.get("marker_type") == det.get("marker_type")
                    and d <= self.distance
                ):
                    possible.append((d, i, key))
        used_d, used_t, assigned = set(), set(), {}
        for d, i, key in sorted(possible):
            if i not in used_d and key not in used_t:
                used_d.add(i)
                used_t.add(key)
                assigned[i] = key
        for i, det in enumerate(detections):
            key = assigned.get(i)
            if key is None:
                key = f"minimap:{self._next_id}"
                self._next_id += 1
            state = self._tracks.get(key, {"first_seen": timestamp, "observations": 0})
            state.update(
                position=det["position"],
                team_hint=det["team_hint"],
                marker_type=det.get("marker_type"),
                last_seen=timestamp,
                observations=state["observations"] + 1,
            )
            self._tracks[key] = state
            det.update(
                track_id=key,
                first_seen=state["first_seen"],
                observations=state["observations"],
            )
        return detections

    def _match_markers(self, crop, roi):
        import cv2
        import numpy as np

        if self._map_templates is None:
            self._map_templates = []
            for entry in self.map_cfg.get("templates", []):
                if entry.get("team", "unknown") not in {"enemy", "teammate", "unknown"}:
                    raise ValueError(
                        "Minimap template team must be enemy/teammate/unknown"
                    )
                if entry.get("marker_type") not in {
                    "player_marker",
                    "death_marker",
                    "spike_marker",
                }:
                    raise ValueError("Minimap template requires explicit marker_type")
                if not entry.get("name") or not entry.get("path"):
                    raise ValueError("Minimap template requires name and path")
                template = cv2.imread(str(entry["path"]))
                if (
                    template is None
                    or min(template.shape[:2]) < 3
                    or float(template.std()) < 1
                ):
                    raise ValueError("Unreadable or constant minimap template")
                threshold = _number(
                    entry.get("threshold", 0.88), "template threshold", 0, 1
                )
                scales = [
                    _number(v, "template scale", 0.25, 4)
                    for v in entry.get("scales", [0.75, 0.9, 1, 1.1, 1.25])
                ]
                reference_width = _number(
                    entry.get("reference_roi_width", 428), "reference_roi_width", 1
                )
                self._map_templates.append(
                    (entry, template, threshold, scales, reference_width)
                )
        candidates = []
        for entry, original, threshold, scales, reference_width in self._map_templates:
            for scale in scales:
                factor = scale * crop.shape[1] / reference_width
                w, h = max(3, round(original.shape[1] * factor)), max(
                    3, round(original.shape[0] * factor)
                )
                if w > crop.shape[1] or h > crop.shape[0]:
                    continue
                template = cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR)
                scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
                # Bounded local maxima avoid thousands of overlapping detections.
                for _ in range(12):
                    _, score, _, (x, y) = cv2.minMaxLoc(scores)
                    if not math.isfinite(score) or score < threshold:
                        break
                    candidates.append(
                        {
                            "team_hint": entry.get("team", "unknown"),
                            "marker_type": entry["marker_type"],
                            "position": [
                                (x + w / 2) / crop.shape[1],
                                (y + h / 2) / crop.shape[0],
                            ],
                            "bbox": [
                                x + roi[0],
                                y + roi[1],
                                x + w + roi[0],
                                y + h + roi[1],
                            ],
                            "confidence": float(score),
                            "template_name": entry["name"],
                            "verified_player": False,
                            "verified_marker_semantics": True,
                            "identity": None,
                            "spike_state": entry.get("spike_state", "unknown"),
                            "evidence_method": "calibrated_profile_color_template",
                        }
                    )
                    scores[
                        max(0, y - h // 2) : min(scores.shape[0], y + h // 2 + 1),
                        max(0, x - w // 2) : min(scores.shape[1], x + w // 2 + 1),
                    ] = -1

        def overlap(a, b):
            a, b = a["bbox"], b["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            return (
                intersection
                / max(
                    1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                )
                > 0.35
            )

        output = []
        for item in sorted(candidates, key=lambda x: -x["confidence"]):
            conflicts = [
                other
                for other in candidates
                if overlap(item, other)
                and (item["team_hint"], item["marker_type"])
                != (other["team_hint"], other["marker_type"])
            ]
            if conflicts and item["confidence"] - max(
                o["confidence"] for o in conflicts
            ) < self.map_cfg.get("template_ambiguity_margin", 0.03):
                continue
            if any(overlap(item, other) for other in output):
                continue
            item["normalized_roi"] = item["position"]
            output.append(item)
        return output

    def _spike(self, frame, roi):
        import cv2
        import numpy as np

        path = self.ui_cfg.get("spike_template_path")
        if not path:
            return {"visible": None, "status": "no_calibrated_template", "score": None}
        if self._template is None:
            self._template = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
            if self._template is None or not np.any(self._template > 127):
                raise ValueError(f"Unreadable/empty spike template: {path}")
        mask = _mask(cv2.cvtColor(_crop(frame, roi), cv2.COLOR_BGR2HSV), RED)
        if self.spike_match_method == "local_mask_iou":
            # Search the complete shape, including disconnected pieces and
            # holes. A connected-component-only crop would discard those parts.
            # Aspect is preserved; coordinates are returned in the source frame.
            mask = (mask > 127).astype(np.float32)
            integral = cv2.integral(mask)
            reference_scale = frame.shape[0] / self.spike_reference_height
            best_score, best_box, best_scale = 0.0, roi, None
            seen_sizes = set()
            for multiplier in self.spike_search_scales:
                scale = reference_scale * multiplier
                width, height = (
                    max(1, round(size * scale))
                    for size in (self._template.shape[1], self._template.shape[0])
                )
                if (width, height) in seen_sizes:
                    continue
                seen_sizes.add((width, height))
                if (
                    min(width, height) < self.spike_min_side
                    or width > mask.shape[1]
                    or height > mask.shape[0]
                ):
                    continue
                template = (
                    cv2.resize(
                        self._template, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                    > 127
                ).astype(np.float32)
                template_pixels = float(template.sum())
                minimum = max(
                    self.spike_min_pixels, self.spike_min_pixels * scale * scale
                )
                if template_pixels < minimum:
                    continue
                intersection = cv2.matchTemplate(mask, template, cv2.TM_CCORR)
                window_pixels = (
                    integral[height:, width:]
                    - integral[:-height, width:]
                    - integral[height:, :-width]
                    + integral[:-height, :-width]
                )
                # Floating-point correlation can drift slightly beyond counts.
                intersection = np.clip(
                    intersection, 0, np.minimum(window_pixels, template_pixels)
                )
                union = window_pixels + template_pixels - intersection
                scores = np.divide(
                    intersection, union, out=np.zeros_like(union), where=union > 0
                )
                scores[intersection < minimum] = 0
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if math.isfinite(score) and score > best_score:
                    best_score, best_scale = float(score), scale
                    best_box = [
                        roi[0] + x,
                        roi[1] + y,
                        roi[0] + x + width,
                        roi[1] + y + height,
                    ]
            return {
                "visible": best_score >= self.template_threshold,
                "score": best_score,
                "method": "local_red_mask_template_iou",
                "bbox": best_box,
                "search_roi": roi,
                "template_scale": best_scale,
                "scale_basis": "frame_height_over_template_reference_height",
            }
        mask = (
            cv2.resize(
                mask,
                (self._template.shape[1], self._template.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 127
        )
        template = self._template > 127
        union = int(np.logical_or(mask, template).sum())
        score = float(np.logical_and(mask, template).sum() / union) if union else 0.0
        return {
            "visible": score >= self.template_threshold,
            "score": score,
            "method": "red_mask_template_iou",
            "bbox": roi,
        }

    def _killfeed(self, frame, roi):
        import cv2
        import numpy as np

        crop = _crop(frame, roi)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red, teal = _mask(hsv, BAR_RED), _mask(hsv, BAR_TEAL)
        scale = frame.shape[0] / 1080
        pitch = max(1, round(self.ui_cfg.get("killfeed_row_pitch_at_1080", 39) * scale))
        height = max(
            1, round(self.ui_cfg.get("killfeed_row_height_at_1080", 34) * scale)
        )
        offset = round(self.ui_cfg.get("killfeed_row_offset_at_1080", 0) * scale)
        ranges = [
            (y, min(y + height, crop.shape[0]))
            for y in range(offset, crop.shape[0] - height + 1, pitch)
        ]
        rows = []
        scale = frame.shape[0] / 1080
        for top, bottom in ranges:
            if not 15 * scale <= bottom - top <= 48 * scale:
                continue
            if (
                min(
                    np.count_nonzero(red[top:bottom]),
                    np.count_nonzero(teal[top:bottom]),
                )
                < (bottom - top) * crop.shape[1] * 0.02
            ):
                continue
            row = crop[max(0, top - 1) : min(crop.shape[0], bottom + 1)]
            # Crop away unused left space based on the bar's occupied columns.
            region = (red | teal)[top:bottom]
            xs = np.where(np.count_nonzero(region, axis=0) > (bottom - top) * 0.15)[0]
            if not len(xs):
                continue
            left, right = int(xs[0]), int(xs[-1]) + 1
            if right - left < crop.shape[1] * 0.45:
                continue
            row = row[:, left:right]
            readings = self._ocr(row, field="killfeed")
            lh = [
                r
                for r in readings
                if sum(p[0] for p in r["bbox"]) / 4 < row.shape[1] * 0.43
            ]
            rh = [
                r
                for r in readings
                if sum(p[0] for p in r["bbox"]) / 4 > row.shape[1] * 0.63
            ]
            killer = " ".join(
                r["text"] for r in sorted(lh, key=lambda r: r["bbox"][0][0])
            ).strip()
            victim = " ".join(
                r["text"] for r in sorted(rh, key=lambda r: r["bbox"][0][0])
            ).strip()
            rhsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
            # Victim color is measured away from the final portrait.
            a, b = int(row.shape[1] * 0.68), int(row.shape[1] * 0.88)
            red_count = int(np.count_nonzero(_mask(rhsv, BAR_RED)[:, a:b]))
            teal_count = int(np.count_nonzero(_mask(rhsv, BAR_TEAL)[:, a:b]))
            larger, smaller = max(red_count, teal_count), min(red_count, teal_count)
            color = (
                ("red" if red_count > teal_count else "teal")
                if larger > max(5 * scale * scale, 2 * smaller)
                else "unknown"
            )
            center = rhsv[:, int(row.shape[1] * 0.43) : int(row.shape[1] * 0.65)]
            white_fraction = float(
                np.count_nonzero((center[:, :, 1] < 65) & (center[:, :, 2] > 195))
                / max(center.shape[0] * center.shape[1], 1)
            )
            names_valid = bool(
                killer
                and victim
                and any(c.isalnum() for c in killer)
                and any(c.isalnum() for c in victim)
            )
            confidence = min([r["confidence"] for r in lh + rh], default=0)
            kill_weapon = self.kill_weapon_reader.read(
                row, (roi[0] + left, roi[1] + max(0, top - 1))
            )
            rows.append(
                {
                    "kill_weapon": kill_weapon,
                    "killer_text": killer or None,
                    "victim_text": victim or None,
                    "victim_color": color,
                    "victim_team_hint": self.team_mapping.get(color, "unknown"),
                    "confidence": confidence,
                    "weapon_icon_white_fraction": white_fraction,
                    "confirmed_visual_structure": names_valid
                    and white_fraction >= 0.04,
                    "bbox": [
                        roi[0] + left,
                        roi[1] + top,
                        roi[0] + right,
                        roi[1] + bottom,
                    ],
                    "readings": readings,
                }
            )
        return rows

    def read(self, frame, timestamp):
        timestamp = _number(timestamp, "timestamp")
        if (self._last_time is not None and timestamp <= self._last_time) or (
            self._floor is not None and timestamp < self._floor
        ):
            raise ValueError("Tactical timestamps must increase monotonically")
        self._last_time = timestamp
        if self._first_time is None:
            self._first_time = timestamp
        # Confirmation candidates are deliberately cleared by unreadable OCR.
        # A row already visible at segment start nevertheless remains censored
        # through a short OCR dropout; forgetting that fact would invent a kill.
        self._initial_killfeed_keys = {
            key: state
            for key, state in self._initial_killfeed_keys.items()
            if timestamp - state["last_seen"] <= self.gate.dedupe
        }
        out = {
            "timestamp": timestamp,
            "enabled": self.enabled,
            "minimap": {
                "markers": [],
                "coordinate_space": "normalized_roi",
                "world_positions": False,
                "mode": self.mode,
                "coverage": self.coverage,
                "semantics": "visible_UI_markers; template semantics limited to calibrated profile; no hidden/world-position or alive guarantee",
            },
            "ui": {},
            "events": [],
        }
        if not self.enabled:
            return out
        if (
            frame is None
            or not hasattr(frame, "shape")
            or len(frame.shape) != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) <= 0
        ):
            raise ValueError("TacticalReader needs a nonempty BGR frame")
        rois = self.pixel_rois(frame.shape)
        out["rois"] = rois
        if self.map_enabled:
            out["minimap"]["markers"] = self._minimap(frame, timestamp, rois["minimap"])
            out["minimap"]["view_change"] = self._view_change
            out["minimap"]["detector"] = dict(self.map_detector.metadata)
            out["minimap"]["current_player_markers_by_team"] = {
                team: sum(
                    m.get("count_as_current_player", False) and m["team_hint"] == team
                    for m in out["minimap"]["markers"]
                )
                for team in ("enemy", "teammate", "unknown")
            }
            out["minimap"][
                "count_semantics"
            ] = "visible player-marker UI observations only; not world alive-player counts"
        expected_aspect = self.cfg.get("calibration_aspect_ratio")
        if (
            expected_aspect is not None
            and abs(frame.shape[1] / frame.shape[0] - expected_aspect) > 0.02
        ):
            out["warnings"] = [
                "Frame aspect differs from calibration; inspect ROI/profile before interpreting markers or UI"
            ]
        candidates = []
        if self.ui_enabled:
            spike = self._spike(frame, rois["spike"])
            out["ui"]["spike_planted"] = spike
            if spike["visible"]:
                candidates.append(
                    {
                        "key": "spike_planted",
                        "event": "spike_planted_ui_observed",
                        "confidence": spike["score"],
                        "evidence": {"source": "top_center_spike_icon", **spike},
                    }
                )
            for field, pattern, event in (
                ("banner", r"\bSPIKE\s+DEFUSED\b", "spike_defused_ui_observed"),
                ("death", r"\bKILLED\s+BY\b", "player_death_ui_observed"),
            ):
                readings = self._ocr(_crop(frame, rois[field]), field=field)
                text = " ".join(r["text"] for r in readings)
                matched = self._phrase(
                    text,
                    "defused" if field == "banner" else "death_panel",
                    "SPIKE DEFUSED" if field == "banner" else "KILLED BY",
                )
                out["ui"][field] = {
                    "text": text,
                    "matched": matched,
                    "readings": readings,
                }
                if matched and (
                    field != "death" or self.ui_cfg.get("death_panel_events", False)
                ):
                    candidates.append(
                        {
                            "key": field,
                            "event": event,
                            "confidence": min(r["confidence"] for r in readings),
                            "evidence": {
                                "source": field + "_ocr",
                                "text": text,
                                "bbox": rois[field],
                                "world_track_id": None,
                                "persistent_panel_not_new_death_each_frame": field
                                == "death",
                            },
                        }
                    )
            planting_readings = self._ocr(
                _crop(frame, rois["planting"]), field="planting"
            )
            planting_text = " ".join(r["text"] for r in planting_readings)
            planting = self._phrase(planting_text, "planting", "ALLY PLANTING")
            out["ui"]["planting"] = {
                "text": planting_text,
                "matched": planting,
                "readings": planting_readings,
                "completed": False,
            }
            if planting:
                candidates.append(
                    {
                        "key": "planting",
                        "event": "planting_ui_observed",
                        "confidence": min(r["confidence"] for r in planting_readings),
                        "evidence": {
                            "source": "explicit_planting_status_OCR",
                            "text": planting_text,
                            "bbox": rois["planting"],
                            "plant_completion_not_inferred": True,
                            "planter_world_track_id": None,
                        },
                    }
                )
            spike_text_readings = self._ocr(
                _crop(frame, rois["spike_text"]), field="spike_text"
            )
            spike_text = " ".join(r["text"] for r in spike_text_readings).upper()
            down = self._phrase(spike_text, "spike_down", "SPIKE DOWN")
            carrier_killed = self._phrase(
                spike_text, "carrier_killed", "SPIKE CARRIER KILLED"
            )
            dropped_markers = [
                m
                for m in out["minimap"]["markers"]
                if m.get("verified_marker_semantics")
                and m.get("marker_type") == "spike_marker"
                and m.get("spike_state") == "dropped"
            ]
            dropped = down or carrier_killed or bool(dropped_markers)
            planted = spike["visible"] is True
            defused = out["ui"]["banner"]["matched"]
            states = [
                name
                for name, visible in (
                    ("planted", planted),
                    ("dropped", dropped),
                    ("defused", defused),
                )
                if visible
            ]
            if states and "defused" not in states:
                # An optional OCR-gap allowance is not permission to combine
                # defuse evidence across an explicitly different Spike state.
                pending = self.gate.states.get("banner")
                if pending is not None and not pending["fired"]:
                    del self.gate.states["banner"]
            out["ui"]["spike_text"] = {
                "text": spike_text,
                "readings": spike_text_readings,
            }
            out["ui"]["spike_state"] = (
                "ambiguous" if len(states) > 1 else states[0] if states else "unknown"
            )
            out["ui"]["spike_state_evidence"] = states
            if len(states) > 1:
                candidates = [c for c in candidates if c["key"] != "spike_planted"]
                # Explicit text remains a UI observation, but contradictory
                # icon/text cannot promote either planted or dropped state.
                for candidate in candidates:
                    if candidate["key"] == "banner":
                        candidate["evidence"]["conflicting_spike_evidence"] = states
            elif dropped:
                confidence = (
                    min(r["confidence"] for r in spike_text_readings)
                    if (down or carrier_killed)
                    else max(m["confidence"] for m in dropped_markers)
                )
                candidates.append(
                    {
                        "key": "spike_dropped",
                        "event": "dropped_spike_state_observed",
                        "confidence": confidence,
                        "evidence": {
                            "source": (
                                "explicit_SPIKE_DOWN_OCR"
                                if down
                                else (
                                    "explicit_SPIKE_CARRIER_KILLED_OCR"
                                    if carrier_killed
                                    else "calibrated_dropped_spike_minimap_template"
                                )
                            ),
                            "text": spike_text,
                            "markers": dropped_markers,
                        },
                    }
                )
            if carrier_killed:
                candidates.append(
                    {
                        "key": "spike_carrier_killed",
                        "event": "spike_carrier_killed_ui_observed",
                        "confidence": min(r["confidence"] for r in spike_text_readings),
                        "evidence": {
                            "source": "explicit_SPIKE_CARRIER_KILLED_OCR",
                            "text": spike_text,
                            "world_track_id": None,
                        },
                    }
                )
            rows = self._killfeed(frame, rois["killfeed"])
            out["ui"]["killfeed"] = rows
            for row in rows:
                if not row["confirmed_visual_structure"]:
                    continue
                team = row["victim_team_hint"]
                event = (
                    f"{team}_death_confirmed"
                    if team in {"enemy", "teammate"}
                    else "killfeed_death_observed"
                )
                key = (
                    "killfeed:"
                    + re.sub(r"\W+", "", row["killer_text"].casefold())
                    + ":"
                    + re.sub(r"\W+", "", row["victim_text"].casefold())
                    + ":"
                    + team
                )
                row["kill_weapon"] = self.kill_weapon_consensus.update(
                    timestamp,
                    key,
                    row.get("kill_weapon", {"name": None, "status": "not_sampled"}),
                )
                if timestamp == self._first_time:
                    self._initial_killfeed_keys[key] = {
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                    }
                initial_row = self._initial_killfeed_keys.get(key)
                if initial_row is not None:
                    initial_row["last_seen"] = timestamp
                candidates.append(
                    {
                        "key": key,
                        "event": event,
                        "confidence": row["confidence"],
                        "evidence": {
                            "source": "killfeed",
                            **row,
                            "world_track_id": None,
                            "visible_at_first_sample": initial_row is not None,
                            "initial_row_observed_at": (
                                initial_row["first_seen"] if initial_row else None
                            ),
                            "team_basis": "configured_victim_HUD_bar_color_relative_to_viewer; not_world_identity",
                            "dedupe_limitation": "same_name_pair_within_window_can_merge_distinct_kills",
                        },
                    }
                )
        out["events"] = self.gate.update(timestamp, candidates)
        out["ui"]["spike_state_confirmed"] = self._confirm_spike_state(
            timestamp, out["ui"].get("spike_state")
        )
        for event in out["events"]:
            weapon = event["evidence"].get("kill_weapon", {})
            if event["evidence"].get("source") == "killfeed":
                event["kill_weapon"] = weapon.get("name")
                event["kill_weapon_confidence"] = weapon.get("confidence")
            initial = event["evidence"].get(
                "visible_at_first_sample", event["timestamp"] == self._first_time
            )
            event["evidence"]["visible_at_first_sample"] = initial
            if initial and event["evidence"].get("source") == "killfeed":
                event["event"] = "killfeed_death_observed"
        return out

    def draw(self, frame, observations):
        """Draw current observed markers only, never projected hidden positions."""
        import cv2

        for row in observations.get("ui", {}).get("killfeed", []):
            weapon = row.get("kill_weapon", {})
            if weapon.get("name"):
                x1, y1, x2, y2 = map(round, row["bbox"])
                cv2.putText(
                    frame,
                    "Kill weapon: " + weapon["name"],
                    (max(0, x1 - 210), y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 230, 255),
                    2,
                    cv2.LINE_AA,
                )
        for marker in observations.get("minimap", {}).get("markers", []):
            x1, y1, x2, y2 = marker["bbox"]
            color = (
                (50, 70, 255)
                if marker["team_hint"] == "enemy"
                else (
                    (220, 255, 70)
                    if marker["team_hint"] == "teammate"
                    else (0, 210, 255)
                )
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        for name, box in observations.get("rois", {}).items():
            if name != "minimap":
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 200, 100), 1)
            cv2.putText(
                frame,
                "Minimap candidates (local UI coordinates)",
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 200, 100),
                1,
            )
        return frame
