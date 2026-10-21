"""Offline, evidence-qualified person histories. No temporal coincidence is proof of death.

Uses saved, actually observed tracks and HUD events. Agent names can be resolved
later within the SAME visibility episode: identity_resolved_at makes that explicit.
Neither screen-edge motion nor a broken detector track establishes survival/death.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path

TEAMS = {"enemy": "敌方", "teammate": "友方", "unknown": "阵营未确认"}
PERSON_APPEAR = {"enemy_detected", "teammate_detected"}
PERSON_OLD = PERSON_APPEAR | {"enemy_disappeared", "teammate_disappeared"}


def identity(observations):
    samples = {}
    for t, a in observations:
        ev = a.get("agent_evidence") or {}
        name, score = a.get("agent_name"), a.get("agent_confidence", 0)
        fresh = ev.get("last_identity_observed", t)
        if name and ev.get("status") == "confirmed" and score >= 0.75:
            samples[fresh] = (name, float(score), t)
    votes = defaultdict(float)
    for name, score, _ in samples.values():
        votes[name] += score
    result = {
        "agent_name": None,
        "confidence": 0.0,
        "source": "body_crop_classifier",
        "status": "unknown",
        "identity_resolved_at": None,
        "fresh_samples": len(samples),
        "offline_episode_consensus": True,
    }
    if votes:
        name = max(votes, key=votes.get)
        support = [x for x in samples.values() if x[0] == name]
        ratio = votes[name] / sum(votes.values())
        result["candidates"] = dict(votes)
        if len(support) >= 2 and ratio >= 0.8:
            result.update(
                agent_name=name,
                confidence=min(x[1] for x in support) * ratio,
                status="model_prediction",
                identity_resolved_at=max(x[2] for x in support),
            )
        else:
            result["status"] = "insufficient_or_conflicting_identity"
    return result


def episodes(rows, events, clip, grace=0.8):
    active, out, serial = {}, [], Counter()
    for row in rows:
        t, segment = row["timestamp"], row.get("segment_id", 0)
        current = {
            a["track_id"]: a
            for a in row["tracks"]
            if a["label"] in TEAMS and a["label"] != "unknown"
        }
        for key, ep in list(active.items()):
            if key[0] != segment:
                ep.update(end_status="segment_boundary", end_observed_at=t)
                del active[key]
            elif key[1] not in current and t - ep["last_seen"] + 1e-9 >= grace:
                ep.update(end_status="missing_confirmed", end_observed_at=t)
                del active[key]
        for tid, a in current.items():
            key = (segment, tid)
            # Same ID after sparse sampling is not disappearance unless an absent frame confirmed it.
            if key not in active:
                serial[key] += 1
                ep = {
                    "entity_id": f"{clip}:s{segment}:t{tid}:e{serial[key]}",
                    "segment_id": segment,
                    "track_id": tid,
                    "observations": [],
                    "first_seen": t,
                    "last_seen": t,
                    "end_status": "clip_boundary",
                    "end_observed_at": None,
                }
                out.append(ep)
                active[key] = ep
            ep = active[key]
            ep["last_seen"] = t
            ep["observations"].append((t, a))
    confirmed = []
    for ep in out:
        matches = [
            e
            for e in events
            if e["event"] in PERSON_APPEAR
            and e.get("track_id") == ep["track_id"]
            and ep["first_seen"] - 1e-6 <= e["timestamp"] <= ep["last_seen"] + 1e-6
        ]
        if not matches:
            continue
        ep["appearance"] = deepcopy(matches[0])
        ep["team"] = "enemy" if matches[0]["event"] == "enemy_detected" else "teammate"
        ep["identity"] = identity(ep["observations"])
        ep["bbox"] = ep["observations"][-1][1]["bbox"]
        confirmed.append(ep)
    return confirmed


def actor(ep):
    return {
        "entity_id": ep["entity_id"],
        "track_id": ep["track_id"],
        "team": ep["team"],
        "player_name": None,
        "agent_name": ep["identity"]["agent_name"],
        "agent_confidence": ep["identity"]["confidence"],
        "identity": ep["identity"],
    }


def actor_text(a):
    name = a.get("agent_name") or "特工未确认"
    tid = f" #{a['track_id']}" if a.get("track_id") is not None else ""
    player = f"（{a['player_name']}）" if a.get("player_name") else ""
    return f"{TEAMS.get(a.get('team'), '阵营未确认')} {name}{tid}{player}"


def box_iou(a, b):
    cross = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - cross
    return cross / area if area > 0 else 0.0


def exit_evidence(ep, frame_size):
    w, h = frame_size
    obs = ep["observations"]
    first, last = obs[max(0, len(obs) - 6)][1]["bbox"], ep["bbox"]
    dx = (last[0] + last[2] - first[0] - first[2]) / 2 / w
    dy = (last[1] + last[3] - first[1] - first[3]) / 2 / h
    sides = []
    if last[0] / w <= 0.04 and dx < -0.005:
        sides.append("left")
    if last[2] / w >= 0.96 and dx > 0.005:
        sides.append("right")
    if last[1] / h <= 0.04 and dy < -0.005:
        sides.append("top")
    if last[3] / h >= 0.96 and dy > 0.005:
        sides.append("bottom")
    return {
        "edges": sides,
        "normalized_motion": [dx, dy],
        "qualification": "image_motion_only; camera_motion_not_separated",
    }


def death_record(e, clip, index):
    ev = e["evidence"]
    portrait = ev.get("agent_portraits") or {}
    victim, killer = portrait.get("victim") or {}, portrait.get("killer") or {}
    team = ev.get("victim_team_hint", "unknown")
    a = {
        "entity_id": f"{clip}:hud:{index}",
        "track_id": None,
        "team": team,
        "player_name": ev.get("victim_text"),
        "agent_name": victim.get("name"),
        "agent_confidence": victim.get("confidence", 0),
        "identity": victim,
    }
    k = {
        "player_name": ev.get("killer_text"),
        "agent_name": killer.get("name"),
        "agent_confidence": killer.get("confidence", 0),
    }
    # The raw name also represents a new kill whose victim team is unknown.
    # Prefer explicit temporal evidence over that overloaded legacy name.
    old = bool(
        ev.get("visible_at_first_sample", e["event"] == "killfeed_death_observed")
    )
    weapon = e.get("kill_weapon")
    return {
        **deepcopy(e),
        "event": "person_state_changed",
        "state_change": "historical_death_observed" if old else "death_observed",
        "actor": a,
        "killer": k,
        "cause": {
            "code": "killfeed_death",
            "certainty": "hud_confirmed",
            "weapon": weapon,
            "weapon_confidence": e.get("kill_weapon_confidence", 0),
            "world_track_association": "unconfirmed",
        },
        "incident_id": f"{clip}:kill:{index}",
        "raw_event_type": e["event"],
        "description": f"{'开头已有击杀记录：' if old else '击杀栏确认：'}{actor_text(a)} 被 "
        f"{k.get('agent_name') or '特工未确认'}（{k.get('player_name') or '姓名未确认'}）使用 {weapon or '未知武器/技能'} 击杀；画面人物框关联未确认。",
    }


def generic_description(e):
    typ = e["event"]
    ev = e.get("evidence") or {}
    names = {
        "weapon_detected": "发现地上掉落武器（具体枪型未确认，未判断拾取）",
        "dropped_spike_detected": "场景中检测到掉落 Spike 候选",
        "planted_spike_detected": "场景中检测到已安放 Spike",
        "dropped_spike_state_observed": "界面提示 Spike 掉落",
        "spike_carrier_killed_ui_observed": "界面提示携带 Spike 的人物被击杀",
        "planting_ui_observed": "界面显示正在安放 Spike（尚未确认完成）",
        "spike_planted_ui_observed": "界面显示 Spike 已安放",
        "spike_defused_ui_observed": "界面显示 Spike 已拆除",
        "flash_screen_effect_observed": "当前视角出现已支持样式的闪光效果",
        "pov_status_ui_observed": "当前视角出现状态提示："
        + str(ev.get("label") or ev.get("status") or "见证据"),
        "world_ability_observed": "场景中持续出现技能/道具："
        + str(ev.get("label") or e.get("label") or ev.get("ability") or "见证据"),
    }
    if typ in {"player_health_changed", "player_ammo_changed"}:
        return f"{'当前视角血量' if 'health' in typ else '弹匣数量'} {e.get('before')} → {e.get('after')}"
    return names.get(typ, typ)


def build_story(rows, raw_events, clip, frame_size, grace=0.8):
    eps = episodes(rows, raw_events, clip, grace)
    deaths = [
        death_record(e, clip, i)
        for i, e in enumerate(raw_events)
        if e.get("evidence", {}).get("source") == "killfeed"
    ]
    # Candidate pairing requires both agent identity and team; each side must be unique.
    possible = defaultdict(list)
    for di, d in enumerate(deaths):
        if d["state_change"] != "death_observed" or not d["actor"]["agent_name"]:
            continue
        for ep in eps:
            if (
                ep["end_status"] == "missing_confirmed"
                and ep["team"] == d["actor"]["team"]
                and ep["identity"]["agent_name"] == d["actor"]["agent_name"]
                and ep["first_seen"] <= d["timestamp"]
                and -0.15 <= d["timestamp"] - ep["last_seen"] <= 1.25
            ):
                possible[di].append(ep["entity_id"])
    counts = Counter(x for ids in possible.values() for x in ids)
    links = {
        ids[0]: deaths[di]
        for di, ids in possible.items()
        if len(ids) == 1 and counts[ids[0]] == 1
    }
    out = []
    for ep in eps:
        a = actor(ep)
        app = ep["appearance"]
        base = {
            "event": "person_state_changed",
            "actor": a,
            "confidence": app["confidence"],
            "segment_id": ep["segment_id"],
            "track_id": ep["track_id"],
            "agent_name": a["agent_name"],
            "agent_confidence": a["agent_confidence"],
            "timestamp": app["timestamp"],
            "emitted_at": app["emitted_at"],
            "identity_resolved_at": ep["identity"]["identity_resolved_at"],
            "offline_enriched": True,
        }
        out.append(
            {
                **deepcopy(base),
                "state_change": "appeared",
                "cause": {
                    "code": "visible_detection",
                    "certainty": "model_observation",
                },
                "description": f"发现{actor_text(a)}；{'身份来自同次可见期间的模型投票' if a['agent_name'] else '人物太小、身份不足或冲突，保留未知'}。",
                "evidence": {"source_event": app, "bbox": app["evidence"]["bbox"]},
            }
        )
        if ep["end_status"] != "missing_confirmed":
            continue  # End-of-file and cuts are censored, not disappearance.
        cause = {
            "code": "unresolved",
            "certainty": "unknown",
            "weapon": None,
            "alternatives": [
                "left_view",
                "occlusion",
                "detector_miss",
                "track_fragmentation",
                "death",
            ],
        }
        evidence = {
            "last_seen": ep["last_seen"],
            "last_bbox": ep["bbox"],
            "missing_seconds": ep["end_observed_at"] - ep["last_seen"],
        }
        description = (
            f"{actor_text(a)} 从视野中消失；原因未确认，不能仅凭消失认定死亡。"
        )
        linked = links.get(ep["entity_id"])
        incident = None
        if linked:
            cause.update(
                code="possible_killed",
                certainty="inferred_not_confirmed",
                weapon=linked["cause"]["weapon"],
                weapon_confidence=linked["cause"]["weapon_confidence"],
                association_basis="same_predicted_agent_and_team_unique_within_time_window",
            )
            evidence["related_killfeed"] = linked
            incident = linked["incident_id"]
            description = f"{actor_text(a)} 消失；附近击杀栏确认同阵营 {a['agent_name']} 被 {linked['killer'].get('agent_name') or '未知特工'} 使用 {cause['weapon'] or '未知武器/技能'} 击杀。疑似同一人物，框与死者尚未确证。"
        else:
            nearby = [
                x
                for x in eps
                if x["entity_id"] != ep["entity_id"]
                and x["segment_id"] == ep["segment_id"]
                and x["team"] == ep["team"]
                and -0.12 <= x["first_seen"] - ep["last_seen"] <= 0.4
                and box_iou(ep["bbox"], x["observations"][0][1]["bbox"]) >= 0.45
            ]
            edge = exit_evidence(ep, frame_size)
            if nearby:
                cause.update(
                    code="possible_track_fragmentation",
                    certainty="inferred_not_confirmed",
                )
                evidence["replacement_track_candidates"] = [
                    x["entity_id"] for x in nearby
                ]
                description = f"{actor_text(a)} 的原跟踪编号消失；相近位置出现新编号，可能是跟踪断号，不能认定人物死亡或离场。"
            elif edge["edges"]:
                cause.update(
                    code="likely_left_screen_edge", certainty="inferred_not_confirmed"
                )
                evidence["screen_edge_motion"] = edge
                description = f"{actor_text(a)} 消失前向画面边缘移动，倾向离开屏幕；未确认死亡，镜头运动也可能造成这一现象。"
        cause_resolved_at = (
            max(ep["end_observed_at"], linked["emitted_at"])
            if linked
            else ep["end_observed_at"]
        )
        out.append(
            {
                **deepcopy(base),
                "timestamp": ep["last_seen"] + grace,
                "emitted_at": cause_resolved_at,
                "disappearance_confirmed_at": ep["end_observed_at"],
                "cause_resolved_at": cause_resolved_at,
                "state_change": "disappeared",
                "cause": cause,
                "description": description,
                "incident_id": incident,
                "evidence": evidence,
            }
        )
    out.extend(deaths)
    for e in raw_events:
        if (
            e["event"] not in PERSON_OLD
            and e.get("evidence", {}).get("source") != "killfeed"
        ):
            out.append(
                {
                    **deepcopy(e),
                    "state_change": None,
                    "actor": None,
                    "cause": None,
                    "description": generic_description(e),
                }
            )
    out.sort(
        key=lambda e: (
            e["timestamp"],
            e["emitted_at"],
            e.get("state_change") or e["event"],
        )
    )
    for i, e in enumerate(out):
        e["record_id"] = f"{clip}:detail:{i:03d}"
        e["source_clip"] = clip
        e["confidence_note"] = (
            "Evidence score, not a calibrated probability of the stated cause."
        )
    entities = [
        {k: v for k, v in ep.items() if k not in {"observations", "appearance"}}
        for ep in eps
    ]
    return {
        "schema_version": "7.0",
        "mode": "offline_enrichment",
        "events": out,
        "entities": entities,
        "semantics": "Disappearance reason is qualified separately from detection confidence. Killfeed deaths remain HUD-confirmed; timing/agent pairing is only a hypothesis. End-of-clip/camera cuts are not disappearances.",
    }


def write_story(output, story):
    output = Path(output)
    (output / "detailed_events.json").write_text(
        json.dumps(story["events"], ensure_ascii=False, indent=2) + "\n"
    )
    (output / "entity_histories.json").write_text(
        json.dumps(
            {k: v for k, v in story.items() if k != "events"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    fields = [
        "record_id",
        "timestamp",
        "emitted_at",
        "event",
        "state_change",
        "team",
        "agent_name",
        "agent_confidence",
        "identity_resolved_at",
        "track_id",
        "player_name",
        "reason",
        "reason_certainty",
        "killer_agent",
        "killer_player",
        "kill_weapon",
        "weapon_confidence",
        "incident_id",
        "confidence",
        "description",
        "evidence",
    ]
    with (output / "detailed_events.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in story["events"]:
            a = e.get("actor") or {}
            c = e.get("cause") or {}
            k = e.get("killer") or (
                e.get("evidence", {}).get("related_killfeed") or {}
            ).get("killer", {})
            row = {key: e.get(key) for key in fields}
            row.update(
                team=a.get("team"),
                agent_name=a.get("agent_name"),
                agent_confidence=a.get("agent_confidence"),
                track_id=a.get("track_id"),
                player_name=a.get("player_name"),
                reason=c.get("code"),
                reason_certainty=c.get("certainty"),
                killer_agent=k.get("agent_name"),
                killer_player=k.get("player_name"),
                kill_weapon=c.get("weapon"),
                weapon_confidence=c.get("weapon_confidence"),
                evidence=json.dumps(e.get("evidence", {}), ensure_ascii=False),
            )
            w.writerow(row)
    from .normalized_events import normalize_events, write_normalized
    from .compact_events import write_compact

    normalized = normalize_events(story["events"])
    write_normalized(output / "normalized", normalized)
    write_compact(output / "delivery", normalized)


def apply_review_notes(story, reviews):
    """Attach explicit human review without deleting or changing raw predictions."""
    for e in story["events"]:
        for review in reviews:
            if (
                e["source_clip"] == review["clip"]
                and e["event"] == review["event"]
                and abs(e["timestamp"] - review["timestamp"]) < 1e-5
            ):
                e["manual_review"] = deepcopy(review)
                e["description"] = (
                    "【复核提示】" + review["note"] + " 原预测：" + e["description"]
                )
    return story
