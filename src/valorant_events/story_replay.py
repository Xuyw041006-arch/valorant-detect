"""Replay the detailed event layer on completed detector outputs."""

from pathlib import Path
import json, cv2
from .story_events import build_story, write_story, apply_review_notes
from .killfeed_agents import consensus


def enrich(output, video, clip, reader=None, review_notes=None):
    output, video = Path(output), Path(video)
    events = json.loads((output / "events.json").read_text())
    rows = [json.loads(x) for x in (output / "tracks.jsonl").read_text().splitlines()]
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(3)), int(cap.get(4))
    if not cap.isOpened():
        raise ValueError("Cannot open source video")
    portraits = []
    for i, e in enumerate(events):
        ev = e.get("evidence", {})
        if reader is None or ev.get("source") != "killfeed":
            continue
        # Two genuinely different source frames, sampled at observed/confirmed time.
        times = sorted(set([e["timestamp"], e["emitted_at"]]))
        samples = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = map(int, ev["bbox"])
            patch = frame[max(0, y1 - 2) : min(h, y2 + 2), max(0, x1) : min(w, x2)]
            result = reader.read(patch, ev.get("readings"))
            samples.append(result)
        ev["agent_portraits"] = consensus(samples)
        portraits.append(
            {
                "event_index": i,
                "timestamp": e["timestamp"],
                "sample_times": times,
                "samples": samples,
                "result": ev["agent_portraits"],
            }
        )
    cap.release()
    grace = (
        json.loads((output / "resolved_config.json").read_text())
        .get("events", {})
        .get("disappearance_seconds", 0.8)
    )
    story = build_story(rows, events, clip, (w, h), grace)
    if review_notes:
        apply_review_notes(story, review_notes)
    write_story(output, story)
    (output / "killfeed_portraits.json").write_text(
        json.dumps(portraits, indent=2) + "\n"
    )
    print(
        clip,
        len(story["events"]),
        "detailed records",
        len(story["entities"]),
        "episodes",
        flush=True,
    )
    return story
