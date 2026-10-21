"""Static portrait matching on an already located killfeed row.

Not a body classifier; no inference from a player's chosen display name.
Templates include a center crop because the HUD clips head art differently.
"""

from pathlib import Path
import json


class KillfeedAgentReader:
    def __init__(self, directory, threshold=0.60, margin=0.12):
        import cv2

        self.cv2 = cv2
        self.threshold, self.margin = threshold, margin
        self.references = []
        for item in json.loads((Path(directory) / "manifest.json").read_text())[
            "items"
        ]:
            a = cv2.imread(str(Path(directory) / item["file"]), cv2.IMREAD_UNCHANGED)
            self.references.append((item["name"].lower().replace("/", "_"), a))
        self.cache = {}

    def templates(self, height):
        import numpy as np

        if height not in self.cache:
            items = []
            for name, a in self.references:
                for flip in (False, True):
                    b = self.cv2.flip(a, 1) if flip else a
                    for top in (0, 0.12, 0.23, 0.30, 0.4):
                        for bottom in (0.65, 0.75, 0.85, 1.0):
                            if bottom - top < 0.35:
                                continue
                            for side in (0, 0.12, 0.22):
                                q = b[
                                    int(top * b.shape[0]) : int(bottom * b.shape[0]),
                                    int(side * b.shape[1]) : int(
                                        (1 - side) * b.shape[1]
                                    ),
                                ]
                                hh = height - 4
                                ww = round(hh * q.shape[1] / q.shape[0])
                                if hh < 10 or ww < height * 0.55:
                                    continue
                                q = self.cv2.resize(q, (ww, hh))
                                mask = (q[:, :, 3] > 220).astype(np.uint8) * 255
                                if np.count_nonzero(mask) < hh * ww * 0.25:
                                    continue
                                items.append((name, q[:, :, :3], mask, flip))
            self.cache[height] = items
        return self.cache[height]

    def _read(self, patch, row_height):
        import numpy as np

        scores = {}
        for name, q, mask, flip in self.templates(row_height):
            if q.shape[1] > patch.shape[1] or q.shape[0] > patch.shape[0]:
                continue
            v = self.cv2.matchTemplate(patch, q, self.cv2.TM_CCOEFF_NORMED, mask=mask)
            v = np.nan_to_num(v, nan=-1, posinf=-1, neginf=-1)
            _, score, _, pos = self.cv2.minMaxLoc(v)
            if score > scores.get(name, {}).get("score", -1):
                scores[name] = {
                    "score": float(score),
                    "bbox": [pos[0], pos[1], pos[0] + q.shape[1], pos[1] + q.shape[0]],
                    "mirrored": flip,
                }
        ranked = sorted(scores.items(), key=lambda x: -x[1]["score"])
        if len(ranked) < 2:
            return {"name": None, "status": "insufficient_pixels", "confidence": 0}
        (name, first), (second, runner) = ranked[:2]
        gap = first["score"] - runner["score"]
        accepted = first["score"] >= self.threshold and gap >= self.margin
        return {
            "name": name if accepted else None,
            "status": "candidate" if accepted else "unknown",
            "confidence": first["score"] if accepted else 0.0,
            "candidate": name,
            "candidate_score": first["score"],
            "runner_up": second,
            "margin": gap,
            "bbox": first["bbox"],
            "source": "static_killfeed_portrait",
            "confidence_method": "template_score_not_calibrated_probability",
        }

    def read(self, row, readings=None):
        h, w = row.shape[:2]
        end = int(w * 0.4)
        left = [
            r for r in readings or [] if sum(p[0] for p in r["bbox"]) / 4 < w * 0.43
        ]
        if left:
            end = min(
                end,
                max(round(h * 1.2), int(min(p[0] for r in left for p in r["bbox"]))),
            )
        return {
            "killer": self._read(row[:, :end], h),
            "victim": self._read(row[:, int(w * 0.83) :], h),
        }


def consensus(samples):
    result = {}
    for side in ("killer", "victim"):
        observations = [x[side] for x in samples]
        names = {x.get("name") for x in observations}
        if len(observations) >= 2 and len(names) == 1 and None not in names:
            result[side] = {
                **observations[-1],
                "status": "two_frame_confirmed",
                "confidence": min(x["confidence"] for x in observations),
                "observations": len(observations),
            }
        else:
            result[side] = {
                "name": None,
                "confidence": 0,
                "status": "unknown_or_temporally_inconsistent",
                "candidates": [x.get("candidate") for x in observations],
            }
    return result
