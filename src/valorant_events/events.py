"""Time-based, conservative event inference from tracks and optional HUD readings.

This module deliberately has no image, model, or OCR dependencies.  Input times
are video timestamps in seconds.  An event's ``timestamp`` is the first observed
evidence time; ``emitted_at`` is when enough evidence confirmed it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

APPEARANCE_EVENTS = {
    "enemy": "enemy_detected",
    "teammate": "teammate_detected",
    "dropped_spike": "dropped_spike_detected",
    "planted_spike": "planted_spike_detected",
}


def _finite_number(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _confidence(value):
    if not _finite_number(value):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


@dataclass
class _TrackState:
    first_seen: float
    last_seen: float
    frame_index: int
    bbox: list
    observations: deque = field(default_factory=deque)
    confirmed_label: str | None = None
    confirmed_confidence: float = 0.0


@dataclass
class _HudState:
    baseline: int | None = None
    candidate: int | None = None
    candidate_start: float = 0.0
    candidate_count: int = 0
    candidate_confidences: list = field(default_factory=list)
    last_valid_time: float | None = None

    def clear_candidate(self):
        self.candidate = None
        self.candidate_count = 0
        self.candidate_confidences = []


class EventEngine:
    """Convert already-associated tracks into seven documented event types.

    Configuration (either here or in a top-level ``events`` mapping):
      event_confirmation_seconds: 0.25
      event_min_observations: 3
      disappearance_seconds: 0.8
      label_window_seconds: 1.5
      label_min_vote_ratio: 0.70
      min_track_confidence: 0.25
      hud_confirmation_seconds: 0.20
      hud_min_observations: 2
      hud_min_confidence: 0.60
      hud_candidate_max_gap_seconds: 0.75
      health_max: 100
      ammo_max: 200

    Class votes are confidence-weighted before first confirmation. The selected
    class is held for that track lifecycle to avoid contradictory appearance
    events from classification flicker. Trackers should create a new ID for an
    actual object/state change. Disappearance is emitted only for enemies.
    """

    def __init__(self, config: dict):
        config = config or {}
        cfg = config.get("events", config)
        self.confirmation_seconds = float(cfg.get("event_confirmation_seconds", 0.25))
        self.min_observations = int(cfg.get("event_min_observations", 3))
        self.disappearance_seconds = float(cfg.get("disappearance_seconds", 0.8))
        self.label_window_seconds = float(cfg.get("label_window_seconds", 1.5))
        self.label_min_vote_ratio = float(cfg.get("label_min_vote_ratio", 0.70))
        self.min_track_confidence = float(cfg.get("min_track_confidence", 0.25))
        self.hud_confirmation_seconds = float(cfg.get("hud_confirmation_seconds", 0.20))
        self.hud_min_observations = int(cfg.get("hud_min_observations", 2))
        self.hud_min_confidence = float(cfg.get("hud_min_confidence", 0.60))
        self.hud_candidate_max_gap_seconds = float(
            cfg.get("hud_candidate_max_gap_seconds", 0.75)
        )
        self.health_max = int(cfg.get("health_max", 100))
        self.ammo_max = int(cfg.get("ammo_max", 200))
        durations = (
            self.confirmation_seconds,
            self.disappearance_seconds,
            self.label_window_seconds,
            self.hud_confirmation_seconds,
            self.hud_candidate_max_gap_seconds,
        )
        if any(not math.isfinite(v) or v < 0 for v in durations):
            raise ValueError("Event durations must be finite and nonnegative")
        if self.disappearance_seconds <= 0 or self.label_window_seconds <= 0:
            raise ValueError("Disappearance and label windows must be positive")
        if self.label_window_seconds < self.confirmation_seconds:
            raise ValueError(
                "label_window_seconds must cover event_confirmation_seconds"
            )
        if self.min_observations < 1 or self.hud_min_observations < 1:
            raise ValueError("Minimum observation counts must be positive")
        if self.health_max < 0 or self.ammo_max < 0:
            raise ValueError("HUD maxima must be nonnegative")
        if any(
            not math.isfinite(v) or not 0 <= v <= 1
            for v in (
                self.label_min_vote_ratio,
                self.min_track_confidence,
                self.hud_min_confidence,
            )
        ):
            raise ValueError("Confidence and vote thresholds must be in [0, 1]")
        self._tracks = {}
        self._hud = {"health": _HudState(), "ammo": _HudState()}
        self._weapon_key = None
        self._last_timestamp = None
        self._reset_floor = None

    def _check_time(self, timestamp, *, strictly_increasing):
        if not _finite_number(timestamp) or float(timestamp) < 0:
            raise ValueError("timestamp must be a finite nonnegative number")
        timestamp = float(timestamp)
        if self._reset_floor is not None and timestamp < self._reset_floor:
            raise ValueError("timestamp precedes the latest reset boundary")
        if self._last_timestamp is not None:
            invalid = (
                timestamp <= self._last_timestamp
                if strictly_increasing
                else timestamp < self._last_timestamp
            )
            if invalid:
                raise ValueError("Video timestamps must increase monotonically")
        return timestamp

    @staticmethod
    def _event(timestamp, event, confidence, emitted_at, **extra):
        return {
            "timestamp": float(timestamp),
            "event": event,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
            "emitted_at": float(emitted_at),
            **extra,
        }

    def update(
        self, timestamp: float, tracks: list[dict], hud: dict | None = None
    ) -> list[dict]:
        """Consume one sampled frame. Duplicate/backward timestamps are rejected.

        Only actually observed tracks should be passed; predicted boxes for a
        currently missing object must not extend that object's last-seen time.
        Malformed/low-confidence detections and invalid OCR are ignored.
        """
        timestamp = self._check_time(timestamp, strictly_increasing=True)
        self._last_timestamp = timestamp
        events = []
        # Keep one observation per ID per frame; duplicates cannot satisfy the
        # multi-frame confirmation requirement.
        observations = {}
        for track in tracks or []:
            if not isinstance(track, dict):
                continue
            track_id, label = track.get("track_id"), track.get("label")
            confidence = _confidence(track.get("confidence"))
            bbox, frame_index = track.get("bbox"), track.get("frame_index")
            if not isinstance(track_id, Integral) or isinstance(track_id, bool):
                continue
            if (
                label not in APPEARANCE_EVENTS
                or confidence is None
                or confidence < self.min_track_confidence
            ):
                continue
            if (
                not isinstance(frame_index, Integral)
                or isinstance(frame_index, bool)
                or frame_index < 0
            ):
                continue
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not all(_finite_number(v) for v in bbox)
            ):
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            track_id = int(track_id)
            if track_id not in observations or confidence > observations[track_id][1]:
                observations[track_id] = (
                    label,
                    confidence,
                    [float(v) for v in bbox],
                    int(frame_index),
                )

        # Only a frame actually lacking the object can confirm disappearance.
        # Sparse sampling alone must not make an observed ID disappear/reappear.
        for track_id, state in list(self._tracks.items()):
            if (
                track_id not in observations
                and timestamp - state.last_seen + 1e-9 >= self.disappearance_seconds
            ):
                if state.confirmed_label == "enemy":
                    events.append(
                        self._event(
                            state.last_seen + self.disappearance_seconds,
                            "enemy_disappeared",
                            state.confirmed_confidence,
                            timestamp,
                            track_id=track_id,
                            frame_index=state.frame_index,
                            evidence={
                                "last_seen": state.last_seen,
                                "missing_seconds": timestamp - state.last_seen,
                                "label": "enemy",
                                "bbox": state.bbox,
                                "confidence_method": "confirmed_track_score_not_calibrated_probability",
                            },
                        )
                    )
                del self._tracks[track_id]

        for track_id, (label, confidence, bbox, frame_index) in observations.items():
            state = self._tracks.get(track_id)
            if state is None:
                state = self._tracks[track_id] = _TrackState(
                    timestamp, timestamp, frame_index, bbox
                )
            state.last_seen, state.frame_index, state.bbox = (
                timestamp,
                frame_index,
                bbox,
            )
            state.observations.append((timestamp, label, confidence, frame_index))
            while (
                state.observations
                and timestamp - state.observations[0][0] > self.label_window_seconds
            ):
                state.observations.popleft()
            if state.confirmed_label is not None:
                continue
            votes = {}
            for _, observed_label, score, _ in state.observations:
                votes[observed_label] = votes.get(observed_label, 0.0) + score
            winning_label = max(votes, key=votes.get)
            total_weight = sum(votes.values())
            ratio = votes[winning_label] / total_weight if total_weight else 0.0
            supporting = [o for o in state.observations if o[1] == winning_label]
            span = timestamp - supporting[0][0]
            if (
                label != winning_label
                or len(supporting) < self.min_observations
                or span + 1e-9 < self.confirmation_seconds
                or ratio < self.label_min_vote_ratio
            ):
                continue
            state.confirmed_label = winning_label
            state.confirmed_confidence = (
                sum(o[2] for o in supporting) / len(supporting) * ratio
            )
            events.append(
                self._event(
                    supporting[0][0],
                    APPEARANCE_EVENTS[winning_label],
                    state.confirmed_confidence,
                    timestamp,
                    track_id=track_id,
                    frame_index=supporting[0][3],
                    evidence={
                        "label": winning_label,
                        "bbox": bbox,
                        "observations": len(supporting),
                        "vote_ratio": round(ratio, 6),
                        "confirmation_seconds": span,
                        "confidence_method": "mean_detection_score_times_label_vote_ratio",
                    },
                )
            )

        # None means OCR was deliberately not sampled on this frame. An actual
        # failed read is a mapping with None values and breaks confirmation.
        if hud is not None:
            events.extend(self._update_hud(timestamp, hud))
        return sorted(
            events, key=lambda e: (e["timestamp"], e["event"], e.get("track_id", -1))
        )

    def _update_hud(self, timestamp, hud):
        events = []
        weapon_key = hud.get("weapon_key")
        if weapon_key is not None:
            # A known weapon switch changes the ammo baseline, not necessarily
            # ammunition consumption or replenishment.
            if self._weapon_key is not None and weapon_key != self._weapon_key:
                self._hud["ammo"] = _HudState()
            self._weapon_key = weapon_key
        if hud.get("baseline_reset") is True:
            self._hud = {"health": _HudState(), "ammo": _HudState()}
        for field_name, maximum in (
            ("health", self.health_max),
            ("ammo", self.ammo_max),
        ):
            state = self._hud[field_name]
            value, confidence = hud.get(field_name), _confidence(
                hud.get(f"{field_name}_confidence")
            )
            valid = (
                _finite_number(value)
                and float(value).is_integer()
                and 0 <= float(value) <= maximum
                and confidence is not None
                and confidence >= self.hud_min_confidence
            )
            if not valid:
                # An unobserved/ambiguous value is never a zero reading. It also
                # cannot bridge a candidate's required consecutive evidence.
                state.clear_candidate()
                continue
            value = int(value)
            if (
                state.last_valid_time is not None
                and timestamp - state.last_valid_time
                > self.hud_candidate_max_gap_seconds
            ):
                state.clear_candidate()
            state.last_valid_time = timestamp
            if value == state.baseline:
                state.clear_candidate()
                continue
            if value != state.candidate:
                state.candidate = value
                state.candidate_start = timestamp
                state.candidate_count = 1
                state.candidate_confidences = [confidence]
            else:
                state.candidate_count += 1
                state.candidate_confidences.append(confidence)
            if (
                state.candidate_count < self.hud_min_observations
                or timestamp - state.candidate_start + 1e-9
                < self.hud_confirmation_seconds
            ):
                continue
            if state.baseline is not None:
                events.append(
                    self._event(
                        state.candidate_start,
                        f"player_{field_name}_changed",
                        sum(state.candidate_confidences)
                        / len(state.candidate_confidences),
                        timestamp,
                        before=state.baseline,
                        after=value,
                        evidence={
                            "source": "hud_ocr",
                            "observations": state.candidate_count,
                            "weapon_key": (
                                str(self._weapon_key)
                                if self._weapon_key is not None
                                else None
                            ),
                            "confidence_method": "mean_ocr_confidence_not_calibrated_probability",
                        },
                    )
                )
            state.baseline = value
            state.clear_candidate()
        return events

    def reset(self, timestamp, reason="baseline_reset") -> list[dict]:
        """Clear state at a cut, round, or weapon-baseline boundary; emit nothing.

        The next update may use the same boundary timestamp. Timeline rewind is
        rejected; construct a new engine for a different video's time origin.
        """
        timestamp = self._check_time(timestamp, strictly_increasing=False)
        self._tracks.clear()
        self._hud = {"health": _HudState(), "ammo": _HudState()}
        self._weapon_key = None
        self._reset_floor = timestamp
        # Preserve monotonicity relative to earlier processed frames, while a
        # reset immediately before a later frame need not consume that frame.
        return []

    def flush(self, timestamp) -> list[dict]:
        """Finish without inventing disappearance or pending HUD changes at EOF."""
        self._check_time(timestamp, strictly_increasing=False)
        return []
