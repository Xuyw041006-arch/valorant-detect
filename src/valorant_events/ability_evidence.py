"""Independent visual-ability experiment. Evidence channels never imply ownership.

World boxes, minimap symbols and POV status are separate observations. A temporal
co-occurrence is not a verified association to the same cast or world location.
"""

from pathlib import Path
import re, math
import cv2
import numpy as np

STATUS_PATTERNS = {
    "location_revealed": [
        r"LOCATION\s+REVEALED",
        r"POSITION\s+REVEALED",
        r"位置暴露",
        r"位置已暴露",
    ],
    "suppressed": [r"\bSUPPRESSED\b", r"已被压制", r"压制中"],
    "nearsighted": [r"\bNEARSIGHTED\b", r"视野受限"],
    "detained": [r"\bDETAINED\b", r"已被拘束", r"拘束中"],
    "vulnerable": [r"\bVULNERABLE\b", r"易伤"],
    "decay": [r"\bDECAY\b", r"腐蚀"],
    "haunt_destroyed": [r"\bHAUNT\s+DESTROYED\b"],
    "recon_bolt_destroyed": [r"\bRECON\s+BOLT\s+DESTROYED\b"],
}


def parse_status(readings, min_conf=0.65):
    """Readings must be actual OCR from configured status ROI; no cached observations."""
    out = []
    for r in readings:
        text = re.sub(r"\s+", " ", str(r.get("text", "")).upper()).strip()
        conf = float(r.get("confidence", 0))
        if not math.isfinite(conf) or not min_conf <= conf <= 1:
            continue
        for status, patterns in STATUS_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                # Destruction UI is never taken as evidence that POV is affected.
                is_destroyed = status.endswith("_destroyed")
                out.append(
                    {
                        "channel": "hud",
                        "label": status,
                        "confidence": conf,
                        "bbox": r.get("bbox"),
                        "text": text,
                        "event": (
                            "ability_destroyed_ui_observed"
                            if is_destroyed
                            else "pov_status_ui_observed"
                        ),
                        "affected_subject": (
                            None if is_destroyed else "current_observed_pov"
                        ),
                        "caster_agent": None,
                        "caster_team": "unknown",
                        "semantics": "explicit_UI_text_observation",
                    }
                )
    return out


class FlashEffectReader:
    """Spatial burst-pattern matcher. Limited to the audited radial flash styles.

    This is neither brightness-only nor a claim to detect every blind/nearsight.
    """

    def __init__(self, template_paths, threshold=0.72):
        self.threshold = threshold
        self.templates = []
        self.template_hues = {}
        for p in template_paths:
            f = cv2.imread(str(p))
            if f is None:
                raise FileNotFoundError(p)
            g = self._gray(f)
            self.templates.append((Path(p).name, g[20:112, 58:198]))
            self.templates.append(
                (Path(p).name + "#mirror", cv2.flip(g[20:112, 58:198], 1))
            )
            hsv = cv2.cvtColor(cv2.resize(f, (256, 144)), cv2.COLOR_BGR2HSV)[
                20:112, 58:198
            ]
            valid = (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 40)
            hue = (
                int(np.bincount(hsv[:, :, 0][valid], minlength=180).argmax())
                if valid.any()
                else None
            )
            self.template_hues[Path(p).name] = hue
            self.template_hues[Path(p).name + "#mirror"] = hue

    @staticmethod
    def _gray(frame):
        f = cv2.resize(frame, (256, 144))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (3, 3), 0)

    def read(self, frame):
        g = self._gray(frame)
        region = g[10:134, 25:231]
        best = (0, None, None, 0, 0)
        eligible = []
        hsv = cv2.cvtColor(cv2.resize(frame, (256, 144)), cv2.COLOR_BGR2HSV)
        for name, t in self.templates:
            for scale in [0.65, 0.8, 1.0, 1.2]:
                tt = cv2.resize(t, None, fx=scale, fy=scale)
                if (
                    tt.shape[0] > region.shape[0]
                    or tt.shape[1] > region.shape[1]
                    or tt.std() < 3
                ):
                    continue
                result = cv2.matchTemplate(region, tt, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(result)
                if True:
                    patch = region[
                        loc[1] : loc[1] + tt.shape[0], loc[0] : loc[0] + tt.shape[1]
                    ].astype(np.float32)
                    hp = lambda a: a.astype(np.float32) - cv2.GaussianBlur(
                        a.astype(np.float32), (0, 0), 3
                    )
                    edge = float(
                        cv2.matchTemplate(hp(patch), hp(tt), cv2.TM_CCOEFF_NORMED)[0, 0]
                    )
                    b = [
                        loc[0] + 25,
                        loc[1] + 10,
                        loc[0] + 25 + tt.shape[1],
                        loc[1] + 10 + tt.shape[0],
                    ]
                    color = hsv[b[1] : b[3], b[0] : b[2]]
                    supports = []
                    # Burst geometry and tint may cross template families; either audited tint
                    # may support a shape. This rejects red localized orbs, not all fake effects.
                    for reference_hue in set(self.template_hues.values()):
                        if reference_hue is None:
                            continue
                        delta = np.abs(color[:, :, 0].astype(float) - reference_hue)
                        delta = np.minimum(delta, 180 - delta)
                        supports.append(
                            float(
                                (
                                    (delta <= 12)
                                    & (color[:, :, 1] >= 50)
                                    & (color[:, :, 2] >= 40)
                                ).mean()
                            )
                        )
                    hue_support = max(supports, default=0.0)
                    candidate = (score, name, b, edge, hue_support)
                    if score > best[0]:
                        best = candidate
                    if score >= self.threshold and edge >= 0.2 and hue_support >= 0.35:
                        eligible.append(candidate)
        if eligible:
            best = max(eligible, key=lambda a: a[0])
        score, name, b, edge, hue_support = best
        details = {
            "pattern_score": float(score),
            "template": name,
            "threshold": self.threshold,
            "ray_detail_correlation": edge,
            "ray_detail_threshold": 0.2,
            "template_hue_support": hue_support,
            "minimum_hue_support": 0.35,
        }
        if score < self.threshold or edge < 0.2 or hue_support < 0.35:
            return [], details
        h, w = frame.shape[:2]
        box = [b[i] * (w / 256 if i % 2 == 0 else h / 144) for i in range(4)]
        return [
            {
                "channel": "screen_effect",
                "label": "radial_flash_effect",
                "confidence": float(score),
                "bbox": box,
                "event": "flash_screen_effect_observed",
                "affected_subject": "current_observed_pov",
                "caster_agent": None,
                "caster_team": "unknown",
                "semantics": "audited_radial_flash_visual_pattern_not_all_flash_styles",
                "template": name,
            }
        ], details


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    den = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / den if den > 0 else 0


class EvidenceEpisodes:
    """Confirm independent observations, match by IoU, and never count EOF as an end.

    Static initial observations are left-censored states, not newly cast skills.
    Map layout epochs isolate coordinate changes; world/map tracks never merge.
    """

    def __init__(self, min_observations=2, min_seconds=0.1, max_gap=0.5):
        self.minimum = min_observations
        self.seconds = min_seconds
        self.gap = max_gap
        self.states = {}
        self.last = None
        self.counter = 0
        self.segment_start = None
        self.map_epoch = None

    def reset(self):
        self.states = {}
        self.last = None
        self.segment_start = None
        self.map_epoch = None

    def update(self, timestamp, observations, map_epoch=0):
        if (
            not math.isfinite(timestamp)
            or timestamp < 0
            or self.last is not None
            and timestamp <= self.last
        ):
            raise ValueError("timestamps must increase")
        if self.segment_start is None:
            self.segment_start = timestamp
        if self.map_epoch != map_epoch:
            self.states = {
                k: v
                for k, v in self.states.items()
                if v["item"]["channel"] != "minimap"
            }
            self.map_epoch = map_epoch
        self.states = {
            k: v
            for k, v in self.states.items()
            if timestamp - v["last"]
            <= (1.0 if v["item"]["channel"] == "screen_effect" else self.gap)
        }
        assigned = set()
        events = []
        current = []
        unique = []
        for item in sorted(observations, key=lambda o: -o["confidence"]):
            if any(
                o["channel"] == item["channel"]
                and o["label"] == item["label"]
                and (
                    item["channel"] in ["hud", "screen_effect"]
                    or iou(o["bbox"], item["bbox"]) > 0.85
                )
                for o in unique
            ):
                continue
            unique.append(item)
        for item in unique:
            if (
                not math.isfinite(item["confidence"])
                or not 0 <= item["confidence"] <= 1
            ):
                continue
            candidates = []
            for k, s in self.states.items():
                if (
                    k in assigned
                    or s["item"]["channel"] != item["channel"]
                    or s["item"]["label"] != item["label"]
                ):
                    continue
                # Global status/effect is one episode; spatial channels use their own boxes.
                overlap = (
                    1
                    if item["channel"] in ["hud", "screen_effect"]
                    else iou(item["bbox"], s["item"]["bbox"])
                )
                if overlap >= 0.2:
                    candidates.append((overlap, k))
            if candidates:
                key = max(candidates)[1]
            else:
                self.counter += 1
                key = f"ability:{self.counter}"
                self.states[key] = {
                    "first": timestamp,
                    "last": timestamp,
                    "n": 0,
                    "sum": 0.0,
                    "emitted": False,
                    "item": item,
                    "initial": timestamp == self.segment_start,
                }
            s = self.states[key]
            if s["n"] == 0:
                s["first"] = timestamp
            s.update(last=timestamp, item=item)
            s["n"] += 1
            s["sum"] += item["confidence"]
            assigned.add(key)
            if (
                not s["emitted"]
                and s["n"] >= self.minimum
                and timestamp - s["first"] + 1e-9 >= self.seconds
            ):
                s["emitted"] = True
                events.append(
                    {
                        "timestamp": s["first"],
                        "emitted_at": timestamp,
                        "event": item["event"],
                        "confidence": s["sum"] / s["n"],
                        "track_id": key,
                        "evidence": {
                            **item,
                            "observations": s["n"],
                            "visible_at_first_sample": s["initial"],
                            "occurrence_time_unknown": True,
                            "new_cast_claimed": False,
                            "confidence_method": "evidence_score_not_probability",
                        },
                    }
                )
            current.append({**item, "track_id": key, "confirmed": s["emitted"]})
        # Missing fresh evidence breaks confirmation without creating skill-destruction events.
        for k, s in self.states.items():
            if k not in assigned and not s["emitted"]:
                s["n"] = 0
                s["sum"] = 0.0
                s["first"] = timestamp
                s["initial"] = False
        self.last = timestamp
        return current, events


def context_summary(observations):
    """Preserve all sources; never fuse screen and map coordinates or multiply scores."""
    sources = {
        c: [o["label"] for o in observations if o["channel"] == c]
        for c in ["world", "minimap", "hud", "screen_effect"]
    }
    smoke_world = any(x == "smoke" for x in sources["world"])
    smoke_map = any("smoke" in x for x in sources["minimap"])
    return {
        "evidence_channels": sources,
        "smoke_world_visible": smoke_world,
        "smoke_minimap_visible": smoke_map,
        "cooccurring_smoke_evidence": smoke_world and smoke_map,
        "same_instance_verified": False,
        "caster_team": "unknown",
        "note": "cooccurrence is not a spatial association or an affected-player claim",
    }


class AbilityReader:
    def __init__(self, config, root, device="cpu", ocr_factory=None):
        self.cfg = config
        self.root = Path(root)
        self.device = device
        self.models = {}
        self.ocr = None
        self.ocr_factory = ocr_factory
        self.map_epoch = 0
        self.map_last = None
        self.previous_geometry = None
        for channel in ["world", "smoke", "minimap"]:
            c = config.get(channel, {})
            if c.get("enabled", False):
                from ultralytics import YOLO

                self.models[channel] = YOLO(str(self.root / c["weights"]))
        c = config.get("flash", {})
        self.flash = (
            FlashEffectReader(
                [self.root / p for p in c.get("templates", [])],
                c.get("threshold", 0.72),
            )
            if c.get("enabled")
            else None
        )

    @staticmethod
    def crop(frame, roi):
        if (
            len(roi) != 4
            or not all(
                isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1
                for x in roi
            )
            or roi[2] <= roi[0]
            or roi[3] <= roi[1]
        ):
            raise ValueError("invalid normalized ROI")
        h, w = frame.shape[:2]
        b = [round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(roi)]
        if b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError("empty ROI")
        return frame[b[1] : b[3], b[0] : b[2]], b

    def read(self, frame):
        if self.cfg.get("mirror_input", False):
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        observations = []
        detail = {}
        map_cfg = self.cfg.get("minimap", {})
        for channel, model in self.models.items():
            c = self.cfg[channel]
            crop, offset = (
                (frame, [0, 0, w, h])
                if channel != "minimap"
                else self.crop(frame, c["roi"])
            )
            if channel == "minimap":
                small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32))
                geometry = (
                    tuple(c["roi"]),
                    crop.shape[:2],
                    c.get("rotation_mode", "unknown"),
                    c.get("coverage", "unknown"),
                )
                change = (
                    self.previous_geometry is not None
                    and geometry != self.previous_geometry
                )
                if self.map_last is not None:
                    change = change or float(
                        np.abs(small.astype(float) - self.map_last).mean()
                    ) > c.get("view_change_threshold", 32)
                if change or c.get("rotation_mode", "unknown") != "fixed":
                    self.map_epoch += 1
                self.map_last = small
                self.previous_geometry = geometry
                detail["minimap"] = {
                    "roi": offset,
                    "rotation_mode": c.get("rotation_mode", "unknown"),
                    "coverage": c.get("coverage", "unknown"),
                    "layout_epoch": self.map_epoch,
                    "view_change": change,
                    "world_coordinates": False,
                    "cross_frame_identity_enabled": c.get("rotation_mode", "unknown")
                    == "fixed",
                }
            result = model.predict(
                crop,
                imgsz=c.get("imgsz", 960),
                conf=c.get("confidence", 0.4),
                iou=0.5,
                max_det=50,
                device=self.device,
                verbose=False,
            )[0]
            for cl, conf, box in zip(
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.xyxy.cpu().tolist(),
            ):
                label = model.names[int(cl)]
                raw_label = label
                if channel == "world":
                    label = c.get("label_groups", {}).get(label, label)
                if channel == "world" and "smoke" in self.models and label == "smoke":
                    continue  # one declared smoke specialist owns this category
                b = [box[i] + offset[i % 2] for i in range(4)]
                ch = "minimap" if channel == "minimap" else "world"
                o = {
                    "channel": ch,
                    "label": label,
                    "raw_model_label": raw_label,
                    "confidence": float(conf),
                    "bbox": b,
                    "event": (
                        "minimap_ability_symbol_observed"
                        if ch == "minimap"
                        else "world_ability_observed"
                    ),
                    "affected_subject": None,
                    "caster_agent": None,
                    "caster_team": "unknown",
                    "semantics": (
                        "visible_symbol_not_world_location"
                        if ch == "minimap"
                        else "visible_ability_object_or_effect_not_cast_or_hit"
                    ),
                }
                if ch == "minimap":
                    o.update(
                        position_normalized_roi=[
                            (box[0] + box[2]) / 2 / crop.shape[1],
                            (box[1] + box[3]) / 2 / crop.shape[0],
                        ],
                        label_scope="publisher_symbol_class_hypothesis",
                        layout_epoch=self.map_epoch,
                    )
                observations.append(o)
        if self.flash:
            obs, d = self.flash.read(frame)
            observations += obs
            detail["flash"] = d
        cfg = self.cfg.get("status", {})
        if cfg.get("enabled", False):
            if self.ocr is None:
                if self.ocr_factory:
                    self.ocr = self.ocr_factory()
                else:
                    import easyocr

                    self.ocr = easyocr.Reader(
                        cfg.get("languages", ["en"]),
                        gpu=str(self.device).lower() not in {"cpu", "mps"},
                        download_enabled=False,
                        verbose=False,
                    )
            crop, b = self.crop(frame, cfg.get("roi", [0.3, 0.16, 0.7, 0.36]))
            scale = cfg.get("scale", 2)
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
            raw = self.ocr.readtext(crop, detail=1, paragraph=False)
            readings = []
            for poly, txt, score in raw:
                x = [p[0] / scale + b[0] for p in poly]
                y = [p[1] / scale + b[1] for p in poly]
                readings.append(
                    {
                        "text": str(txt),
                        "confidence": float(score),
                        "bbox": [min(x), min(y), max(x), max(y)],
                    }
                )
            detail["status_ocr"] = readings
            obs = parse_status(readings, cfg.get("confidence", 0.65))
            observations += obs
        detail["context"] = context_summary(observations)
        return frame, observations, detail
