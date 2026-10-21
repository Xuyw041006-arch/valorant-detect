"""Optional conservative ID bridging using world-region camera motion and appearance.

Keeps original tracker IDs for audit. It never fabricates missed boxes. An ID
bridge requires a recent inactive track, successful camera-motion estimation,
appearance similarity and unambiguous spatial overlap. Team gating for people
is configurable; raw team observations are retained when it is disabled.
"""

from __future__ import annotations
import math
from .specialists import box_iou


class TrackStitcher:
    def __init__(self, config=None):
        self.cfg = config or {}
        for key, default, low, high in [
            ("max_gap_seconds", 0.5, 1e-9, None),
            ("min_iou", 0.2, 0, 1),
            ("min_similarity", 0.9, 0, 1),
            ("ambiguity_margin", 0.12, 0, 1),
        ]:
            value = self.cfg.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < low
                or (high is not None and value > high)
            ):
                raise ValueError("Invalid stitching " + key)
        if not isinstance(self.cfg.get("direct_reacquire", True), bool):
            raise ValueError("direct_reacquire must be boolean")
        if not isinstance(self.cfg.get("local_reacquire", False), bool):
            raise ValueError("local_reacquire must be boolean")
        if not isinstance(self.cfg.get("soft_team", False), bool):
            raise ValueError("soft_team must be boolean")
        if not isinstance(self.cfg.get("ignore_team_for_person_bridge", False), bool):
            raise ValueError("ignore_team_for_person_bridge must be boolean")
        self.reset()
        self.created = 0
        self.bridges = 0

    def reset(self):
        self.states = {}
        self.raw_map = {}
        self.previous = None
        self.local_gray = None
        self.last_time = None

    def _motion_between(self, previous, gray, frame_shape):
        import cv2, numpy as np

        if previous is None or gray is None:
            return None
        if previous.shape != gray.shape:
            return None
        mask = np.zeros_like(gray)
        mask[100:300, 160:600] = 255
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=250, qualityLevel=0.02, minDistance=8, mask=mask
        )
        if points is None or len(points) < 12:
            return None
        target, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if target is None or status is None:
            return None
        keep = status.ravel().astype(bool)
        old = points[keep]
        new = target[keep]
        if len(old) < 12:
            return None
        affine, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=2.5
        )
        if (
            affine is None
            or inliers is None
            or inliers.mean() < 0.65
            or not np.isfinite(affine).all()
        ):
            return None
        scale = math.hypot(affine[0, 0], affine[0, 1])
        angle = abs(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))
        if not 0.88 < scale < 1.12 or angle > 12:
            return None
        sx, sy = frame_shape[1] / 640, frame_shape[0] / 360
        affine[0, 1] *= sx / sy
        affine[1, 0] *= sy / sx
        affine[0, 2] *= sx
        affine[1, 2] *= sy
        return affine

    def _motion(self, frame):
        import cv2

        gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        previous = self.previous
        self.previous = gray
        return self._motion_between(previous, gray, frame.shape)

    def _direct_motion(self, state, frame):
        # Re-estimate from the actual last-observed image. Never apply a later
        # incremental transform to a box whose intermediate chain failed.
        return self._motion_between(
            state.get("reference_gray"), self.previous, frame.shape
        )

    def _local_motion(self, state, frame):
        """Follow body features, including actor motion and depth-dependent parallax.

        Forward/backward optical-flow checks and RANSAC must both agree. No
        predicted box is emitted; this transform only tests a real new detection.
        """
        import cv2, numpy as np

        previous = state.get("reference_local_gray")
        gray = self.local_gray
        if previous is None or gray is None or previous.shape != gray.shape:
            return None
        sx, sy = gray.shape[1] / frame.shape[1], gray.shape[0] / frame.shape[0]
        x1, y1, x2, y2 = state["observed_box"]
        mx = (x2 - x1) * 0.1
        my = (y2 - y1) * 0.1
        box = [
            max(0, int((x1 + mx) * sx)),
            max(0, int((y1 + my) * sy)),
            min(gray.shape[1], int((x2 - mx) * sx)),
            min(gray.shape[0], int((y2 - my) * sy)),
        ]
        if box[2] - box[0] < 6 or box[3] - box[1] < 12:
            return None
        mask = np.zeros_like(previous)
        mask[box[1] : box[3], box[0] : box[2]] = 255
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=50, qualityLevel=0.02, minDistance=2, mask=mask
        )
        if points is None or len(points) < 6:
            return None
        target, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(21, 21), maxLevel=3
        )
        if target is None or status is None:
            return None
        backward, reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous, target, None, winSize=(21, 21), maxLevel=3
        )
        if backward is None or reverse is None:
            return None
        keep = (
            status.ravel().astype(bool)
            & reverse.ravel().astype(bool)
            & (np.linalg.norm(backward - points, axis=2).ravel() <= 1.0)
        )
        if keep.sum() < 6 or keep.mean() < 0.6:
            return None
        old, new = points[keep], target[keep]
        affine, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=1.5
        )
        if (
            affine is None
            or inliers is None
            or inliers.mean() < 0.75
            or not np.isfinite(affine).all()
        ):
            return None
        scale = math.hypot(affine[0, 0], affine[0, 1])
        angle = abs(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))
        if not 0.8 <= scale <= 1.25 or angle > 15:
            return None
        affine[0, 1] *= sy / sx
        affine[1, 0] *= sx / sy
        affine[0, 2] /= sx
        affine[1, 2] /= sy
        return affine

    @staticmethod
    def _transform(box, affine):
        import numpy as np

        x1, y1, x2, y2 = box
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ affine.T
        return [
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        ]

    def _appearance(self, frame, box):
        import cv2, numpy as np

        x1, y1, x2, y2 = map(int, box)
        w, h = frame.shape[1], frame.shape[0]
        # Exclude outline and background along the body rectangle borders.
        mx, my = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        crop = frame[
            max(0, y1 + my) : min(h, y2 - my), max(0, x1 + mx) : min(w, x2 - mx)
        ]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
        return hist / (np.linalg.norm(hist) + 1e-9)

    def update(self, frame, timestamp, tracks):
        import numpy as np, cv2

        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("Stitcher timestamps must be finite and nonnegative")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Stitcher timestamps must increase")
        if len({t["track_id"] for t in tracks}) != len(tracks):
            raise ValueError("Duplicate raw track IDs in one frame")
        self.last_time = timestamp
        affine = self._motion(frame)
        ttl = float(self.cfg.get("max_gap_seconds", 0.5))
        if self.cfg.get("local_reacquire", False):
            ratio = min(1.0, 960 / frame.shape[1])
            self.local_gray = cv2.cvtColor(
                cv2.resize(
                    frame,
                    (round(frame.shape[1] * ratio), round(frame.shape[0] * ratio)),
                ),
                cv2.COLOR_BGR2GRAY,
            )
        self.states = {
            k: v for k, v in self.states.items() if timestamp - v["last"] <= ttl
        }
        if affine is not None:
            for state in self.states.values():
                state["box"] = self._transform(state["box"], affine)
        else:
            for state in self.states.values():
                state["bridge_valid"] = False
        current_raw = {t["track_id"] for t in tracks}
        active = {
            self.raw_map[x]
            for x in current_raw
            if x in self.raw_map and self.raw_map[x] in self.states
        }
        features = {t["track_id"]: self._appearance(frame, t["bbox"]) for t in tracks}
        pair_scores = {}
        candidate_motion = {}
        for key, state in self.states.items():
            if key in active:
                continue
            local = (
                self._local_motion(state, frame)
                if self.cfg.get("local_reacquire", False)
                else None
            )
            if local is not None:
                candidate_motion[key] = (
                    self._transform(state["observed_box"], local),
                    "body_forward_backward_optical_flow",
                )
                continue
            if affine is not None and state.get("bridge_valid", True):
                candidate_motion[key] = (state["box"], "continuous_camera_motion_chain")
            elif self.cfg.get("direct_reacquire", True):
                direct = self._direct_motion(state, frame)
                if direct is not None:
                    candidate_motion[key] = (
                        self._transform(state["observed_box"], direct),
                        "direct_last_observed_to_current",
                    )
        for track in tracks:
            raw = track["track_id"]
            feature = features[raw]
            if raw in self.raw_map or feature is None:
                continue
            for key, (candidate_box, basis) in candidate_motion.items():
                state = self.states[key]
                if state["feature"] is None:
                    continue
                overlap = box_iou(track["bbox"], candidate_box)
                sim = float(feature @ state["feature"])
                if state["label"] != track["label"]:
                    # Team is optional identity evidence for people. Other object
                    # classes cannot cross. Motion, appearance, spatial overlap
                    # and ambiguity checks apply regardless of team policy.
                    both_people = {state["label"], track["label"]} == {
                        "enemy",
                        "teammate",
                    }
                    ignore_team = (
                        self.cfg.get("ignore_team_for_person_bridge", False)
                        and both_people
                    )
                    soft_match = (
                        self.cfg.get("soft_team", False)
                        and both_people
                        and basis == "body_forward_backward_optical_flow"
                        and overlap >= 0.5
                        and sim >= 0.95
                    )
                    if not (ignore_team or soft_match):
                        continue
                if overlap >= self.cfg.get("min_iou", 0.2) and sim >= self.cfg.get(
                    "min_similarity", 0.9
                ):
                    pair_scores[raw, key] = (0.6 * overlap + 0.4 * sim, overlap, sim)
        output = []
        assigned = set()
        for track in sorted(tracks, key=lambda t: t["track_id"]):
            row = dict(track)
            raw = track["track_id"]
            feature = features[raw]
            canonical = self.raw_map.get(raw)
            evidence = None
            if canonical in assigned:
                # A bridged source ID later reappeared concurrently: split the
                # alias immediately rather than assigning two people one ID.
                self.created += 1
                canonical = self.created
                self.raw_map[raw] = canonical
            if canonical is None:
                candidates = []
                for (candidate_raw, key), (score, overlap, sim) in pair_scores.items():
                    if candidate_raw != raw or key in active:
                        continue
                    competing = [
                        other[0]
                        for (other_raw, other_key), other in pair_scores.items()
                        if other_key == key and other_raw != raw
                    ]
                    if competing and score - max(competing) < self.cfg.get(
                        "ambiguity_margin", 0.12
                    ):
                        continue
                    candidates.append((score, key, overlap, sim))
                candidates.sort(reverse=True)
                if candidates and (
                    len(candidates) == 1
                    or candidates[0][0] - candidates[1][0]
                    >= self.cfg.get("ambiguity_margin", 0.12)
                ):
                    _, canonical, overlap, sim = candidates[0]
                    self.bridges += 1
                    evidence = dict(
                        method="world_optical_flow_and_body_HS_histogram",
                        iou=overlap,
                        appearance_similarity=sim,
                        source_tracker_id=track.get("tracker_id"),
                        motion_basis=candidate_motion[canonical][1],
                        last_observed_timestamp=self.states[canonical]["last"],
                        gap_seconds=timestamp - self.states[canonical]["last"],
                    )
                    evidence.update(
                        historical_label=self.states[canonical]["label"],
                        observed_label=track["label"],
                        team_label_conflict=self.states[canonical]["label"]
                        != track["label"],
                    )
                else:
                    self.created += 1
                    canonical = self.created
                self.raw_map[raw] = canonical
            active.add(canonical)
            assigned.add(canonical)
            history = self.states.get(canonical, {}).get("label_history", [])
            history = [x for x in history if timestamp - x[0] <= 1.5] + [
                (timestamp, track["label"], track["confidence"])
            ]
            votes = {}
            for _, label, confidence in history:
                votes[label] = votes.get(label, 0) + confidence
            label = (
                max(votes, key=votes.get)
                if self.cfg.get("soft_team", False)
                else track["label"]
            )
            self.states[canonical] = dict(
                box=list(track["bbox"]),
                last=timestamp,
                label=label,
                label_history=history,
                feature=feature,
                bridge_valid=True,
                observed_box=list(track["bbox"]),
                reference_gray=self.previous,
                reference_local_gray=self.local_gray,
            )
            row["unstitched_track_id"] = raw
            row["track_id"] = canonical
            if evidence:
                row["stitch_evidence"] = evidence
            output.append(row)
        return output
