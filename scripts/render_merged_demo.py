"""Render the final reviewed events in the compact bilingual presentation layout."""

from pathlib import Path
import json, sys, subprocess, argparse, collections, copy
import cv2, numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg
from render_text import write, box_label
from event_text import describe, NAMES, EN_NAMES, LABELS, EN_LABELS
from presentation_timing import build_event_frames, recent_event_indices

R = Path(__file__).resolve().parents[1]
OUT = R / "docs/presentation"
W, H, SIDE, FPS = 1920, 1200, 440, 30
TITLES = {
    "p006": ("交战与 Spike", "Combat & Spike"),
    "weapon": ("武器与拆除", "Weapons & defuse"),
    "spike": ("狙击与掉包", "Sniper & Spike drop"),
    "agents": ("特工识别", "Agent recognition"),
    "baseline": ("血量与弹药", "Health & ammo"),
    "native": ("墙体与闪光", "Wall & flash"),
}


def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def encoder(path, w):
    return subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "3",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    lang = args.lang
    tr = lambda zh, en: zh if lang == "zh" else en
    manifest = json.loads((R / "presentation/edit_manifest.json").read_text())
    sidecar = json.loads((R / "evidence.json").read_text())
    events = json.loads((R / "events.json").read_text())
    records = sidecar["records"]
    mapping = sidecar["track_id_map"]
    plan = build_event_frames(events, records, manifest)
    alignment = []
    out = encoder(OUT / f"valorant_demo_{lang}.mp4", W + SIDE)
    raw = encoder(R / "docs/media/merged_source.mp4", W) if args.clean else None
    frames = 0
    for si, seg in enumerate(manifest["segments"]):
        c = seg["id"]
        off = seg["start"]
        rs = [(i, x) for i, x in enumerate(records) if x["source_clip"] == c]
        allowed = collections.defaultdict(list)
        for idx, x in rs:
            if events[idx]["event"] == "person_appeared":
                allowed[x["original_normalized_record"]["subject"]["track_id"]].append(
                    (
                        plan[idx]["local_frame"],
                        events[idx]["subject"].get("agent"),
                        events[idx]["subject"]["team"],
                    )
                )
        object_tracks = {
            x["original_normalized_record"]["subject"].get("track_id")
            for _, x in rs
            if x["normalized_record"]["event"] == "object_detected"
            and x["original_normalized_record"]["subject"].get("class")
            == "dropped_weapon"
        }
        ability_types = {
            events[i]["subject"]["type"]
            for i, x in rs
            if events[i]["event"] == "object_detected"
        }
        rows = lines(R / "examples/render_inputs" / c / "tracks.jsonl")
        maps = lines(R / "examples/render_inputs" / c / "tactical.jsonl")
        abilities = lines(R / "examples/render_inputs" / c / "abilities.jsonl")
        ri = mi = ai = -1
        last_hp = last_ammo = None
        cap = cv2.VideoCapture(str(R / seg["source_video"]))
        sfps = cap.get(5)
        sw, sh = int(cap.get(3)), int(cap.get(4))
        scale = min(W / sw, H / sh)
        nw, nh = round(sw * scale), round(sh * scale)
        dx, dy = (W - nw) // 2, (H - nh) // 2
        source_index = -1
        source_frame = None
        panel = None
        last_tick = -1
        last_available = None

        def xybox(box):
            x1, y1, x2, y2 = box
            return [
                round(x1 * scale + dx),
                round(y1 * scale + dy),
                round(x2 * scale + dx),
                round(y2 * scale + dy),
            ]

        for n in range(seg["frames"]):
            t = n / FPS
            global_t = off + t
            wanted = int(t * sfps + 1e-8)
            while source_index < wanted:
                ok, source_frame = cap.read()
                assert ok, (c, n, wanted)
                source_index += 1
            frame = np.zeros((H, W, 3), np.uint8)
            frame[dy : dy + nh, dx : dx + nw] = cv2.resize(source_frame, (nw, nh))
            if raw:
                raw.stdin.write(frame.tobytes())
            while ri + 1 < len(rows) and rows[ri + 1]["timestamp"] <= t + 1e-8:
                ri += 1
                hu = rows[ri].get("hud") or {}
                if hu.get("health") is not None:
                    last_hp = (hu["health"], rows[ri]["timestamp"])
                if hu.get("ammo") is not None:
                    last_ammo = (hu["ammo"], rows[ri]["timestamp"])
            while mi + 1 < len(maps) and maps[mi + 1]["timestamp"] <= t + 1e-8:
                mi += 1
            while (
                ai + 1 < len(abilities) and abilities[ai + 1]["timestamp"] <= t + 1e-8
            ):
                ai += 1
            row = rows[ri] if ri >= 0 else {}
            mp = maps[mi] if mi >= 0 else {}
            ar = abilities[ai] if ai >= 0 else {}
            hp = last_hp[0] if last_hp and t - last_hp[1] <= 1 else None
            ammo = last_ammo[0] if last_ammo and t - last_ammo[1] <= 1 else None
            markers = (
                [
                    m
                    for m in mp.get("minimap", {}).get("markers", [])
                    if m.get("verified_marker_semantics")
                    and m.get("confidence", 0) >= 0.6
                ]
                if t - mp.get("timestamp", -99) <= 0.25
                else []
            )
            available = recent_event_indices(plan, c, n)
            if int(t * 10) != last_tick or available != last_available:
                panel = Image.new("RGB", (SIDE, H), (12, 22, 32))
                d = ImageDraw.Draw(panel)
                write(d, (22, 28), "VALORANT", 32, "#74e0c9")
                write(d, (22, 81), tr("游戏事件演示", "Gameplay event review"), 24)
                write(
                    d,
                    (22, 133),
                    f"{int(global_t)//60:02d}:{global_t%60:04.1f} / 01:19.9",
                    28,
                )
                write(
                    d,
                    (22, 188),
                    f"{si+1}/6  " + TITLES[c][0 if lang == "zh" else 1],
                    21,
                    "#a9bdcc",
                )
                if hp is not None or ammo is not None:
                    write(
                        d,
                        (22, 235),
                        f'{tr("血量","HP")} {hp if hp is not None else "—"}   {tr("弹药","Ammo")} {ammo if ammo is not None else "—"}',
                        24,
                    )
                if markers:
                    d.line((22, 284, SIDE - 22, 284), fill="#344b60", width=2)
                    write(
                        d,
                        (22, 300),
                        tr("小地图可见符号", "Visible minimap symbols"),
                        24,
                        "#74e0c9",
                    )
                    count = collections.Counter(m["marker_type"] for m in markers)
                    names = {
                        "player_marker": tr("人物", "player"),
                        "smoke_marker": tr("烟雾", "smoke"),
                        "spike_marker": "Spike",
                        "death_marker": tr("死亡标志", "death marker"),
                        "ally_death_marker": tr("友方死亡", "ally death"),
                        "enemy_death_marker": tr("敌方死亡", "enemy death"),
                    }
                    write(
                        d,
                        (22, 343),
                        " / ".join(
                            f'{names.get(k,k.replace("_"," "))} {v}'
                            for k, v in count.items()
                        ),
                        21,
                    )
                y = 445
                write(d, (22, y), tr("最近事件", "Recent events"), 24, "#74e0c9")
                y += 50
                for idx in reversed(available[-3:]):
                    e = events[idx]
                    write(
                        d,
                        (22, y),
                        f"{e['timestamp']:.1f}s · {(LABELS if lang=='zh' else EN_LABELS)[e['event']]}",
                        18,
                        "#a9bdcc",
                    )
                    y += 28
                    y = write(d, (22, y), describe(e, lang), 21) + 24
                write(
                    d,
                    (22, H - 70),
                    tr("事件回放 · 原速游戏片段", "Event replay · original speed"),
                    18,
                    "#a9bdcc",
                )
                panel = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
                last_tick = int(t * 10)
                last_available = list(available)
            drawn_person_tracks = []
            if 0 <= t - row.get("timestamp", -99) <= 1 / sfps + 0.01:
                for obs in row.get("tracks", []):
                    ident = obs["track_id"]
                    label = obs["label"]
                    is_person = label in ("enemy", "teammate")
                    if is_person:
                        candidates = [a for a in allowed.get(ident, []) if a[0] <= n]
                        if not candidates:
                            continue
                        name = candidates[-1][1]
                        team = candidates[-1][2]
                        # Offline replay attaches the final track identity retrospectively.
                        # Raw per-frame identity votes and confirmation times remain in evidence.
                        drawn_person_tracks.append(ident)
                        text = (
                            (NAMES if lang == "zh" else EN_NAMES).get(team, team)
                            + " "
                            + (
                                name.title()
                                if name
                                else tr("未知特工", "Unknown agent")
                            )
                        )
                        color = (
                            (80, 225, 120)
                            if team == "teammate"
                            else (90, 100, 255) if team == "enemy" else (150, 180, 200)
                        )
                    elif label == "dropped_weapon" and ident in object_tracks:
                        text = tr("掉落武器", "Dropped weapon")
                        color = (0, 220, 255)
                    else:
                        continue
                    x1, y1, x2, y2 = xybox(obs["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    box_label(frame, text, x1, y1 - 3, color)
            for marker in markers:
                x1, y1, x2, y2 = xybox(marker["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 210, 80), 1)
            if 0 <= t - ar.get("timestamp", -99) <= 0.25:
                for o in ar.get("observations", []):
                    if (
                        o.get("confirmed")
                        and o.get("bbox")
                        and o["channel"] == "world"
                        and o["label"] in ability_types
                    ):
                        x1, y1, x2, y2 = xybox(o["bbox"])
                        color = (255, 180, 70)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        box_label(
                            frame,
                            (NAMES if lang == "zh" else EN_NAMES).get(
                                o["label"], o["label"]
                            ),
                            x1,
                            y1 - 3,
                            color,
                        )
            canvas = np.concatenate([frame, panel], axis=1)
            for target in [p for p in plan if p["clip"] == c and p["local_frame"] == n]:
                idx = target["event_index"]
                assert idx in available, ("Event missing from seek frame panel", idx)
                person = events[idx]["event"] == "person_appeared"
                tid = records[idx]["original_normalized_record"]["subject"].get(
                    "track_id"
                )
                assert not person or tid in drawn_person_tracks, (
                    "Person has no actual box on seek frame",
                    idx,
                    tid,
                )
                alignment.append(
                    dict(
                        **target,
                        panel_event_indices=list(available),
                        target_text=describe(events[idx], lang),
                        person_box_drawn=tid in drawn_person_tracks if person else None,
                        source_frame=source_index,
                    )
                )
            out.stdin.write(canvas.tobytes())
            frames += 1
            if (
                n == min(round(2 * FPS), seg["frames"] - 1)
                or c == "native"
                and n == 100
            ):
                cv2.imwrite(str(OUT / f"preview_{lang}_{c}_{n}.jpg"), canvas)
        cap.release()
        print(lang, c, "done", frames, flush=True)
    out.stdin.close()
    assert out.wait() == 0
    if raw:
        raw.stdin.close()
        assert raw.wait() == 0
    assert frames == manifest["frames"]
    assert len(alignment) == len(events)
    (OUT / f"event_alignment_{lang}.json").write_text(
        json.dumps(
            {
                "policy": "Offline retrospective occurrence-time overlays; source boxes only; seek to centre of exact annotated frame. Confirmation times remain unchanged in evidence.",
                "fps": FPS,
                "records": sorted(alignment, key=lambda x: x["event_index"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (OUT / f"manifest_{lang}.json").write_text(
        json.dumps(
            {
                "video": f"presentation/valorant_demo_{lang}.mp4",
                "source": "media/merged_source.mp4",
                "language": lang,
                "width": W + SIDE,
                "height": H,
                "frames": frames,
                "fps": FPS,
                "duration": frames / FPS,
                "still_card_frames": 0,
                "source_segments": 6,
                "retained_events": len(events),
                "overlay_policy": "Offline retrospective occurrence-time event text and final associated identity; retained source person and object boxes; disproven identities shown as unknown; existing map symbol observations; resets at every source boundary. No new model inference.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
