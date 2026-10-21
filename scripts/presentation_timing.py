"""One frame schedule shared by offline annotations and browser event seeking.

Event timestamps describe occurrence; emitted_at describes later confirmation.
The saved retrospective presentation uses occurrence time. It never changes the
inference/metric records or invents boxes. Millisecond-rounding tolerance handles
normalized timestamps at exact frame/cut boundaries. Seek into a frame's centre
so a browser cannot decode the preceding frame at a floating-point boundary.
"""

import math

FPS = 30
ROUNDING_SECONDS = 0.0005


def build_event_frames(events, records, manifest):
    if len(events) != len(records):
        raise ValueError("Event and evidence counts differ")
    fps = manifest["fps"]
    segments = {}
    offset = 0
    for segment in manifest["segments"]:
        if abs(segment["start"] - offset / fps) > 1e-5:
            raise ValueError("Segment start does not match output frame offset")
        segments[segment["id"]] = (segment, offset)
        offset += segment["frames"]
    result = []
    for index, (event, record) in enumerate(zip(events, records)):
        segment, offset = segments[record["source_clip"]]
        local_time = record["source_timestamp"]
        if (
            not math.isfinite(local_time)
            or local_time < 0
            or local_time > segment["frames"] / fps
        ):
            raise ValueError("Event is outside its source clip")
        local_frame = min(
            segment["frames"] - 1,
            max(0, math.ceil((local_time - ROUNDING_SECONDS) * fps - 1e-9)),
        )
        frame = offset + local_frame
        result.append(
            dict(
                event_index=index,
                clip=segment["id"],
                local_frame=local_frame,
                frame=frame,
                frame_time=frame / fps,
                seek_time=(frame + 0.5) / fps,
                event_time=event["timestamp"],
                confirmed_at=record["normalized_record"]["emitted_at"],
            )
        )
    return result


def recent_event_indices(plan, clip, local_frame, limit=3):
    available = [
        p for p in plan if p["clip"] == clip and p["local_frame"] <= local_frame
    ]
    available.sort(key=lambda p: (p["local_frame"], p["event_index"]))
    return [p["event_index"] for p in available[-limit:]]
