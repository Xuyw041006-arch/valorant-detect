"""Integrated pipeline: source-pixel ability evidence joins the baseline timeline.

One reader sample may inform the overlay until the next sample, but it is never
fed into the temporal gate twice. Source channels and identity namespaces remain
separate. Camera cuts reset every gate without reusing ability IDs.
"""

from copy import deepcopy
import math
from pathlib import Path


def prepare_integrated_config(config, device, no_ocr=False, root=None):
    cfg = deepcopy(config)
    root = Path(root or Path(__file__).resolve().parents[2])

    # Resolve declared local resources against the installed project, not cwd.
    def resolve(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    key in {"weights", "spike_template_path", "path"}
                    and isinstance(value, str)
                    and value.startswith(("weights/", "assets/"))
                ):
                    obj[key] = str(root / value)
                elif isinstance(value, (dict, list)):
                    resolve(value)
        elif isinstance(obj, list):
            for item in obj:
                resolve(item)

    resolve(cfg)
    use_gpu = str(device).lower() not in {"cpu", "mps"}
    cfg.setdefault("hud", {})["gpu"] = use_gpu
    tactical = cfg.setdefault("tactical", {})
    tactical["gpu"] = use_gpu
    tactical.setdefault("minimap", {}).setdefault("detector", {})["device"] = device
    ability = cfg.get("abilities", {})
    if ability.get("enabled"):
        if cfg.get("utility", {}).get("enabled"):
            raise ValueError(
                "Disable legacy utility when integrated abilities are enabled"
            )
        if ability.get("mirror_input"):
            raise ValueError(
                "Integrated modules require the same original pixel coordinates; pre-normalize mirrored video"
            )
        interval = ability.get("sample_interval_seconds", 0.1)
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError(
                "abilities.sample_interval_seconds must be finite and positive"
            )
        if ability.get("minimap", {}).get("enabled"):
            mm = tactical["minimap"]
            if not mm.get("roi") or not ability.get("minimap", {}).get(
                "profile_calibrated", False
            ):
                raise ValueError(
                    "Ability minimap requires a calibrated tactical ROI and profile_calibrated=true"
                )
            ability["minimap"].update(
                roi=mm["roi"],
                rotation_mode=mm.get("mode", "unknown"),
                coverage=mm.get("coverage", "unknown"),
            )
    if no_ocr:
        cfg["hud"]["enabled"] = False
        tactical.setdefault("ui", {})["ocr_enabled"] = False
        ability.setdefault("status", {})["enabled"] = False
    return cfg


class AbilityIntegration:
    def __init__(self, config, device, root=None, reader_factory=None):
        from .ability_evidence import AbilityReader, EvidenceEpisodes

        self.cfg = config.get("abilities", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.reader = (
            (reader_factory or AbilityReader)(
                self.cfg, root or Path(__file__).resolve().parents[2], device
            )
            if self.enabled
            else None
        )
        self.gate = EvidenceEpisodes()
        self.interval = float(self.cfg.get("sample_interval_seconds", 0.1))
        self.last = None
        self.latest = None
        self.samples = 0

    def reset(self):
        self.gate.reset()  # Counter intentionally survives to avoid duplicate IDs.
        self.last = self.latest = None
        if self.reader:
            self.reader.map_last = self.reader.previous_geometry = None
            self.reader.map_epoch += 1

    def sample(self, frame, timestamp, frame_index, segment_id):
        if (
            not self.enabled
            or self.last is not None
            and timestamp - self.last + 1e-9 < self.interval
        ):
            return None, []
        _, observations, details = self.reader.read(frame)
        tracks, events = self.gate.update(
            timestamp, observations, self.reader.map_epoch
        )
        for event in events:
            event.update(frame_index=frame_index, category="ability_evidence")
            event["evidence"]["segment_id"] = segment_id
            event["evidence"]["confirmation_frame_index"] = frame_index
        record = dict(
            timestamp=timestamp,
            frame_index=frame_index,
            segment_id=segment_id,
            observations=tracks,
            evidence=details,
        )
        self.latest, self.last = record, timestamp
        self.samples += 1
        return record, events

    def draw(self, cv2, image, timestamp):
        if not self.latest or timestamp - self.last > self.interval + 0.05:
            return image
        height, width = image.shape[:2]
        for obs in self.latest["observations"]:
            if obs.get("bbox"):
                x1, y1, x2, y2 = map(round, obs["bbox"])
                color = (255, 170, 60) if obs["channel"] == "world" else (210, 100, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{obs['channel']}:{obs['label']} {obs['confidence']:.2f}"
                cv2.putText(
                    image,
                    label,
                    (max(0, min(x1, width - 180)), max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        labels = [
            o["label"]
            for o in self.latest["observations"]
            if o["channel"] in {"hud", "screen_effect"}
        ]
        if labels:
            cv2.putText(
                image,
                "POV: " + ", ".join(labels),
                (12, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )
        return image
