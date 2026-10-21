"""Optional numeric HUD OCR with conservative parsing and configurable ROIs.

No OCR/image dependency is imported until enabled OCR is actually read. Default
ROIs are initial guesses for the standard English PC 16:9 VALORANT HUD; inspect
and calibrate them for the chosen recording before relying on the readings.
"""

import importlib
import math
import re

DEFAULT_ROIS = {
    "health": (0.29, 0.90, 0.35, 0.99),
    "ammo": (0.66, 0.90, 0.715, 0.99),
}


class HudReader:
    """Read health and magazine ammo only; never substitute 0 for OCR failure.

    ``config`` may be the HUD mapping itself or have a top-level ``hud`` mapping.
    Keys: enabled, min_confidence (0.60), health_roi/ammo_roi (normalized xyxy),
    health_max (100), ammo_max (200), model_storage_directory (optional), and
    download_enabled (False). Missing OCR models raise an actionable error;
    downloading is opt-in instead of a hidden run-time network operation.

    recognition_methods optionally maps health/ammo to "detect" (the default
    readtext detection+recognition path) or "recognize" (the complete crop is one
    text region). Direct recognition requires a calibrated ROI containing the
    complete visible number, with no adjacent value or distracting background.
    It can invent digits on an unsuitable crop; confidence/range/time checks
    still apply, and calibration examples are not an independent benchmark.

    Optional ocr_scale/ocr_scales resize crops before OCR (default 1). The
    preprocess mapping supports none, gray, and gray_threshold per field;
    thresholds supplies the fixed 0..255 threshold (default 210). These options
    require profile-specific validation and leave the default image unchanged.
    """

    def __init__(self, config: dict, backend="easyocr", gpu=False):
        config = config or {}
        cfg = config.get("hud", config)
        self.backend = str(backend).lower()
        self.enabled = bool(cfg.get("enabled", True)) and self.backend not in {
            "none",
            "disabled",
            "off",
        }
        if self.enabled and self.backend != "easyocr":
            raise ValueError("Supported HUD backends are 'easyocr' and 'none'")
        self.gpu = gpu
        self.min_confidence = float(cfg.get("min_confidence", 0.60))
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("HUD min_confidence must be in [0, 1]")
        self.maximum = {
            "health": int(cfg.get("health_max", 100)),
            "ammo": int(cfg.get("ammo_max", 200)),
        }
        if any(value < 0 for value in self.maximum.values()):
            raise ValueError("HUD maxima must be nonnegative")
        methods = cfg.get("recognition_methods", {})
        if not isinstance(methods, dict) or set(methods) - set(DEFAULT_ROIS):
            raise ValueError(
                "HUD recognition_methods must map only health/ammo to 'detect' or 'recognize'"
            )
        self.recognition_methods = {
            field: methods.get(field, "detect") for field in DEFAULT_ROIS
        }
        if any(
            method not in ("detect", "recognize")
            for method in self.recognition_methods.values()
        ):
            raise ValueError("HUD recognition method must be 'detect' or 'recognize'")

        def scale(value):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.5 <= value <= 4
            ):
                raise ValueError("HUD OCR scale must be finite and between 0.5 and 4")
            return float(value)

        default_scale = scale(cfg.get("ocr_scale", 1))
        scales, preprocess, thresholds = (
            cfg.get(key, {}) for key in ("ocr_scales", "preprocess", "thresholds")
        )
        for mapping in (scales, preprocess, thresholds):
            if not isinstance(mapping, dict) or set(mapping) - set(DEFAULT_ROIS):
                raise ValueError(
                    "HUD preprocessing mappings may contain only health/ammo"
                )
        self.ocr_scales = {
            field: scale(scales.get(field, default_scale)) for field in DEFAULT_ROIS
        }
        self.preprocess = {
            field: preprocess.get(field, "none") for field in DEFAULT_ROIS
        }
        if any(
            value not in ("none", "gray", "gray_threshold")
            for value in self.preprocess.values()
        ):
            raise ValueError("HUD preprocess must be none, gray, or gray_threshold")
        self.thresholds = {field: thresholds.get(field, 210) for field in DEFAULT_ROIS}
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.thresholds.values()
        ):
            raise ValueError("HUD thresholds must be integers between 0 and 255")
        self.rois = {}
        nested_rois = cfg.get("rois", {})
        for field, default in DEFAULT_ROIS.items():
            roi = cfg.get(f"{field}_roi", nested_rois.get(field, default))
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                raise ValueError(f"{field}_roi must be normalized [x1, y1, x2, y2]")
            roi = tuple(float(v) for v in roi)
            if (
                not all(math.isfinite(v) and 0 <= v <= 1 for v in roi)
                or roi[2] <= roi[0]
                or roi[3] <= roi[1]
            ):
                raise ValueError(f"Invalid normalized {field}_roi")
            self.rois[field] = roi
        self.model_storage_directory = cfg.get("model_storage_directory")
        self.download_enabled = bool(cfg.get("download_enabled", False))
        self._reader = None

    def _load_reader(self):
        if self._reader is None:
            try:
                easyocr = importlib.import_module("easyocr")
            except ImportError as exc:
                raise RuntimeError(
                    "HUD OCR requires easyocr. Install the OCR dependencies or disable hud.enabled."
                ) from exc
            kwargs = {
                "gpu": self.gpu,
                "download_enabled": self.download_enabled,
                "verbose": False,
            }
            if self.model_storage_directory:
                kwargs["model_storage_directory"] = self.model_storage_directory
            try:
                self._reader = easyocr.Reader(["en"], **kwargs)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(
                    "EasyOCR models are unavailable. Prepare its English models, opt in with hud.download_enabled, or disable OCR."
                ) from exc
        return self._reader

    def pixel_rois(self, frame_shape):
        """Return xyxy pixel boxes suitable for drawing/calibration."""
        if len(frame_shape) < 2 or int(frame_shape[0]) <= 0 or int(frame_shape[1]) <= 0:
            raise ValueError("frame must have positive height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        result = {}
        for field, (x1, y1, x2, y2) in self.rois.items():
            left, top = int(x1 * width), int(y1 * height)
            right, bottom = min(width, math.ceil(x2 * width)), min(
                height, math.ceil(y2 * height)
            )
            if right <= left or bottom <= top:
                raise ValueError("HUD ROI is empty at the supplied image resolution")
            result[field] = (left, top, right, bottom)
        return result

    def _parse(self, results, maximum):
        # Keep only complete digit strings. Never repair O->0 or take a substring
        # of '25/75', since that could turn ambiguous OCR into false certainty.
        values = {}
        for result in results or []:
            if not isinstance(result, (tuple, list)) or len(result) < 3:
                continue
            text = str(result[1]).strip()
            try:
                confidence = float(result[2])
            except (ValueError, TypeError):
                continue
            if (
                not math.isfinite(confidence)
                or not self.min_confidence <= confidence <= 1
            ):
                continue
            if not re.fullmatch(r"[0-9]{1,3}", text):
                continue
            value = int(text)
            if value > maximum:
                continue
            values[value] = max(values.get(value, 0.0), confidence)
        if len(values) != 1:
            return None, 0.0
        return next(iter(values.items()))

    def _prepare_crop(self, crop, field):
        scale, mode = self.ocr_scales[field], self.preprocess[field]
        if scale == 1 and mode == "none":
            return crop
        import cv2

        if scale != 1:
            height, width = crop.shape[:2]
            crop = cv2.resize(
                crop,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        if mode != "none":
            if len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if mode == "gray_threshold":
                _, crop = cv2.threshold(
                    crop, self.thresholds[field], 255, cv2.THRESH_BINARY
                )
        return crop

    def read(self, frame) -> dict:
        result = {
            "health": None,
            "ammo": None,
            "health_confidence": 0.0,
            "ammo_confidence": 0.0,
            "weapon_key": None,
        }
        if not self.enabled:
            return result
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("frame must be an image array")
        boxes = self.pixel_rois(frame.shape)
        reader = self._load_reader()
        for field, (x1, y1, x2, y2) in boxes.items():
            crop = self._prepare_crop(frame[y1:y2, x1:x2], field)
            if self.recognition_methods[field] == "recognize":
                # None/None means one whole-image text box in EasyOCR 1.7.2.
                # Empty lists would instead ask it to recognize zero boxes.
                readings = reader.recognize(
                    crop,
                    horizontal_list=None,
                    free_list=None,
                    decoder="greedy",
                    batch_size=1,
                    workers=0,
                    allowlist="0123456789",
                    detail=1,
                    paragraph=False,
                )
            else:
                readings = reader.readtext(
                    crop, allowlist="0123456789", detail=1, paragraph=False
                )
            value, confidence = self._parse(readings, self.maximum[field])
            result[field] = value
            result[f"{field}_confidence"] = round(float(confidence), 6)
        return result
