"""Public event format: four required fields plus optional event-specific details.

Evidence and timing qualifiers stay in an index-aligned sidecar. Uncertainty
that changes the meaning of an event must remain in the public event itself.
"""

from copy import deepcopy
from pathlib import Path
import csv
import json


def compact_event(e):
    s = e["subject"]
    d = e["details"]
    kind = e["event"]
    if kind.startswith("person_"):
        subject = {"team": s["team"], "agent": s.get("agent")}
        if s.get("track_id") is not None:
            subject["track_id"] = s["track_id"]
    else:
        subject = {"type": s.get("class") or s["type"]}
    details = {}
    if kind == "person_disappeared":
        reason = d["reason"]
        if d["certainty"] == "inferred":
            reason = {
                "death": "death_suspected",
                "left_view": "left_view_suspected",
                "tracking_lost": "tracking_loss_suspected",
            }.get(reason, reason)
        details = {"reason": reason}
        if d["reason"] == "death":
            details.update(
                killer=(d.get("killer") or {}).get("agent"), weapon=d.get("weapon")
            )
    elif kind == "object_detected":
        if d.get("source") not in (None, "world"):
            details["source"] = d["source"]
        if d.get("observation") == "reobserved":
            details["reobserved"] = True
        if d.get("review_status"):
            details["review"] = d["review_status"]
    elif kind == "state_changed":
        if d["field"] in ("health", "ammo"):
            details = {"field": d["field"], "from": d["before"], "to": d["after"]}
        elif d["field"] == "status":
            details = {"status": d["value"]}
            if d.get("reason"):
                details["reason"] = d["reason"]
        elif d["field"] == "effect":
            details = {"effect": d["value"]}
        else:
            raise ValueError("Unsupported state field")
    elif kind != "person_appeared":
        raise ValueError("Unsupported event")
    result = {
        "timestamp": e["timestamp"],
        "event": kind,
        "confidence": round(e["confidence"], 3),
        "subject": subject,
    }
    if details:
        result["details"] = details
    return result


def compact_bundle(normalized):
    events = [compact_event(e) for e in normalized["events"]]
    sources = {e["source_clip"] for e in normalized["events"]}
    if len(sources) > 1:
        raise ValueError(
            "Compact format requires one source video; source context belongs in metadata"
        )
    return {
        "events": events,
        "evidence": {
            "schema_version": "9.0",
            "source_clip": next(iter(sources), None),
            "index_base": 0,
            "records": [
                {"event_index": i, "normalized_record": deepcopy(e)}
                for i, e in enumerate(normalized["events"])
            ],
            "source_records": deepcopy(normalized["evidence"]),
            "initial_state": deepcopy(normalized["initial_state"]),
        },
    }


def write_compact(directory, normalized):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle = compact_bundle(normalized)
    for key in ("events", "evidence"):
        (directory / (key + ".json")).write_text(
            json.dumps(bundle[key], ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    with (directory / "events.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "event", "confidence", "subject", "details"]
        )
        writer.writeheader()
        for e in bundle["events"]:
            writer.writerow(
                {
                    **e,
                    "subject": json.dumps(e["subject"], ensure_ascii=False),
                    "details": (
                        json.dumps(e["details"], ensure_ascii=False)
                        if "details" in e
                        else ""
                    ),
                }
            )
    return bundle
