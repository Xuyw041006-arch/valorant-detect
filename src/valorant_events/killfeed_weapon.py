"""Identify the weapon icon inside one parsed killfeed row using static assets.

No game API or match data is consulted at inference. Similar silhouettes and
unrecognized ability icons must pass an absolute score and a runner-up margin.
The result is a weapon hypothesis, separate from death/OCR confidence.
"""

import json
from pathlib import Path
import math


class WeaponConsensus:
    """Two fresh, consecutive matching icon readings; unknown breaks the vote."""

    def __init__(self, minimum=2, max_gap=0.75):
        self.minimum, self.max_gap = minimum, max_gap
        self.states = {}

    def reset(self):
        self.states.clear()

    def update(self, timestamp, key, observation):
        self.states = {
            k: v
            for k, v in self.states.items()
            if timestamp - v["last"] <= self.max_gap
        }
        result = dict(observation)
        name = observation.get("name")
        state = self.states.get(key)
        if state and timestamp <= state["last"]:
            raise ValueError("Weapon votes require fresh increasing timestamps")
        if not name:
            self.states.pop(key, None)
            return {**result, "observations": 0}
        if state is None or state["name"] != name:
            state = dict(name=name, last=timestamp, n=0, score=1.0)
        state.update(
            last=timestamp,
            n=state["n"] + 1,
            score=min(state["score"], observation["confidence"]),
        )
        self.states[key] = state
        result.update(observations=state["n"])
        if state["n"] < self.minimum:
            result.update(
                name=None, confidence=None, status="awaiting_temporal_confirmation"
            )
        else:
            result.update(confidence=state["score"], status="temporally_confirmed")
        return result


class KillfeedWeaponReader:
    def __init__(self, config=None):
        self.cfg = config or {}
        self.enabled = self.cfg.get("enabled", False)
        self.templates = None
        self.threshold = float(self.cfg.get("threshold", 0.72))
        self.margin = float(self.cfg.get("margin", 0.06))
        if not all(
            math.isfinite(x) and 0 <= x <= 1 for x in (self.threshold, self.margin)
        ):
            raise ValueError("Killfeed weapon thresholds must be in [0,1]")

    def _load(self):
        import cv2
        import numpy as np

        root = Path(
            self.cfg.get(
                "assets",
                Path(__file__).resolve().parents[2] / "assets/killfeed_weapons",
            )
        )
        manifest = json.loads((root / "manifest.json").read_text())
        self.templates = []
        for item in manifest["items"]:
            rgba = cv2.imread(str(root / item["file"]), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                raise ValueError("Weapon reference must be an RGBA icon")
            alpha = rgba[:, :, 3]
            yy, xx = np.where(alpha > 100)
            if not len(xx):
                raise ValueError("Empty weapon icon")
            alpha = alpha[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
            self.templates.append((item["name"], alpha.astype(np.float32) / 255))

    def read(self, row, origin=(0, 0)):
        import cv2
        import numpy as np

        unknown = {
            "name": None,
            "status": "disabled" if not self.enabled else "unrecognized_or_ambiguous",
            "confidence": None,
            "source": "killfeed_weapon_icon",
            "confidence_method": "template_match_score_not_calibrated_probability",
        }
        if not self.enabled:
            return unknown
        if self.templates is None:
            self._load()
        h, w = row.shape[:2]
        if h < 12 or w < 60:
            return {**unknown, "status": "too_small"}
        hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 185)).astype(np.float32)
        left, right = round(w * 0.2), round(w * 0.8)
        search = white[:, left:right]
        ranked = []
        for name, alpha in self.templates:
            best = {"name": name, "score": -1.0}
            used = set()
            for fraction in [0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78]:
                th = max(6, round(h * fraction))
                tw = round(th * alpha.shape[1] / alpha.shape[0])
                if (
                    (tw, th) in used
                    or tw < w * 0.04
                    or tw + 4 > search.shape[1]
                    or th + 4 > h
                ):
                    continue
                used.add((tw, th))
                template = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_AREA)
                for mirror in [False, True]:
                    template_view = cv2.flip(template, 1) if mirror else template
                    template_view = cv2.copyMakeBorder(
                        template_view, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0
                    )
                    scores = cv2.matchTemplate(
                        search, template_view, cv2.TM_CCOEFF_NORMED
                    )
                    _, score, _, loc = cv2.minMaxLoc(scores)
                    if score > best["score"]:
                        best = dict(
                            name=name,
                            score=float(score),
                            mirrored=mirror,
                            bbox=[
                                origin[0] + left + loc[0] + 2,
                                origin[1] + loc[1] + 2,
                                origin[0] + left + loc[0] + 2 + tw,
                                origin[1] + loc[1] + 2 + th,
                            ],
                        )
            ranked.append(best)
        ranked.sort(key=lambda x: -x["score"])
        best, second = ranked[:2]
        gap = best["score"] - second["score"]
        accepted = best["score"] >= self.threshold and gap >= self.margin
        return {
            **unknown,
            "name": best["name"] if accepted else None,
            "status": "recognized" if accepted else unknown["status"],
            "confidence": best["score"] if accepted else None,
            "candidate": best["name"],
            "candidate_score": best["score"],
            "runner_up": second["name"],
            "runner_up_score": second["score"],
            "margin": gap,
            "threshold": self.threshold,
            "required_margin": self.margin,
            "bbox": best.get("bbox"),
            "mirrored_reference": best.get("mirrored"),
            "class_coverage": len(ranked),
            "new_kill_inferred_from_icon": False,
        }
