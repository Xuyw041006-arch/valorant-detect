"""Compact V8 delivery schema. Death is a reason for person_disappeared.

Death means removal from alive status, not necessarily disappearance of pixels.
An inferred world-track match remains a candidate even when its evidence is
grouped into the same incident. Original records live in a separate evidence map.
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import csv
import json
import math

EVENTS = {"person_appeared", "person_disappeared", "object_detected", "state_changed"}
TEAM = {"enemy": "敌方", "teammate": "友方", "unknown": "阵营未确认"}
OBJECT_NAMES = {
    "dropped_weapon": "掉落武器",
    "dropped_spike": "掉落 Spike",
    "planted_spike": "已安放 Spike",
    "smoke": "烟雾",
    "wall_effect": "墙体效果",
    "cypher_tripwire_anchor": "Cypher 绊线锚点",
}
SPIKE = {
    "dropped_spike_state_observed": "dropped",
    "planting_ui_observed": "planting",
    "spike_planted_ui_observed": "planted",
    "spike_defused_ui_observed": "defused",
    "spike_carrier_killed_ui_observed": "carrier_killed",
}


def person(a):
    a = a or {}
    return {
        "type": "person",
        "team": a.get("team", "unknown"),
        "agent": a.get("agent_name"),
        "player": a.get("player_name"),
        "track_id": a.get("track_id"),
    }


def person_text(a):
    name = (a.get("agent") or "未知特工").title()
    return f"{TEAM.get(a.get('team'),'阵营未确认')} {name}" + (
        f" #{a['track_id']}" if a.get("track_id") is not None else ""
    )


def ready_at(e):
    return max(e.get("emitted_at", e["timestamp"]), e.get("identity_resolved_at") or 0)


def base(e, event, subject, details, description, sources=None):
    sources = sources or [e]
    return {
        "timestamp": round(e["timestamp"], 3),
        "event": event,
        "confidence": round(e["confidence"], 6),
        "emitted_at": round(max(ready_at(s) for s in sources), 3),
        "source_clip": e["source_clip"],
        "subject": subject,
        "details": details,
        "description": description,
        "evidence_ids": [s["record_id"] for s in sources],
    }


def can_group(death, missing):
    if death["source_clip"] != missing["source_clip"] or not death.get("incident_id"):
        return False
    if death["incident_id"] != missing.get("incident_id"):
        return False
    a, b = death.get("actor") or {}, missing.get("actor") or {}
    if (
        not a.get("agent_name")
        or a.get("agent_name") != b.get("agent_name")
        or a.get("team") != b.get("team")
    ):
        return False
    last = missing.get("evidence", {}).get("last_seen")
    return last is not None and -0.15 <= death["timestamp"] - last <= 1.25


def death_event(e, missing=None):
    a = person(e.get("actor"))
    killer = e.get("killer") or {}
    cause = e.get("cause") or {}
    # Subject is the HUD victim. Never copy an inferred world ID into that identity.
    a["track_id"] = None
    match = {"status": "unmatched"}
    if missing:
        candidate = (missing.get("actor") or {}).get("track_id")
        explicit = e.get("evidence", {}).get("world_track_id")
        verified = (
            explicit is not None
            and explicit == candidate
            and cause.get("world_track_association") == "confirmed"
        )
        match = {
            "status": "confirmed" if verified else "inferred",
            "candidate_track_id": candidate,
        }
        if verified:
            a["track_id"] = candidate
    details = {
        "reason": "death",
        "certainty": "confirmed",
        "scope": "alive_status",
        "killer": {
            "agent": killer.get("agent_name"),
            "player": killer.get("player_name"),
        },
        "weapon": cause.get("weapon") or e.get("kill_weapon"),
        "world_match": match,
    }
    if missing:
        details["visibility_lost_at"] = round(missing["timestamp"], 3)
    text = f"{person_text(a)} 被 {(killer.get('agent_name') or '未知特工').title()} 使用 {details['weapon'] or '未知武器/技能'} 击杀"
    if missing and match["status"] == "inferred":
        text += f"（疑似对应画面 #{match['candidate_track_id']}）"
    return base(
        e,
        "person_disappeared",
        a,
        details,
        text + "。",
        [e, missing] if missing else [e],
    )


def normalize_single(e):
    transition = e.get("state_change")
    a = person(e.get("actor"))
    c = e.get("cause") or {}
    ev = e.get("evidence") or {}
    typ = e["event"]
    if transition == "appeared":
        return base(
            e,
            "person_appeared",
            a,
            {"identity": "predicted" if a["agent"] else "unknown"},
            f"发现{person_text(a)}。",
        )
    if transition == "disappeared":
        reasons = {
            "unresolved": ("unknown", "unknown", "原因未知"),
            "likely_left_screen_edge": ("left_view", "inferred", "倾向离开屏幕"),
            "possible_track_fragmentation": (
                "tracking_lost",
                "inferred",
                "可能跟踪断号",
            ),
            "possible_killed": ("death", "inferred", "疑似被击杀"),
        }
        reason, certainty, zh = reasons.get(
            c.get("code"), ("unknown", "unknown", "原因未知")
        )
        details = {
            "reason": reason,
            "certainty": certainty,
            "scope": "screen_visibility",
            "last_seen": (
                round(ev["last_seen"], 3) if ev.get("last_seen") is not None else None
            ),
        }
        if reason == "death":
            details["weapon"] = c.get("weapon")
        return base(
            e, "person_disappeared", a, details, f"{person_text(a)} 消失：{zh}。"
        )
    if transition == "death_observed":
        return death_event(e)
    if typ in {
        "weapon_detected",
        "dropped_spike_detected",
        "planted_spike_detected",
        "world_ability_observed",
        "minimap_ability_symbol_observed",
        "utility_detected",
    }:
        kind = {
            "weapon_detected": "dropped_weapon",
            "dropped_spike_detected": "dropped_spike",
            "planted_spike_detected": "planted_spike",
        }.get(typ, ev.get("label") or ev.get("utility_type") or "unknown")
        subject = {
            "type": (
                "ability"
                if "ability" in typ or typ == "utility_detected"
                else "weapon" if typ == "weapon_detected" else "spike"
            ),
            "class": kind,
            "track_id": e.get("track_id"),
        }
        details = {"source": ev.get("channel") or "world", "observation": "visible"}
        review = e.get("manual_review") or {}
        status = review.get("assessment")
        prefix = "发现"
        if status == "reobserved_same_physical_weapon":
            prefix = "再次发现"
            details["observation"] = "reobserved"
        if status:
            details["review_status"] = status
        text = prefix + OBJECT_NAMES.get(kind, kind) + "。"
        if status == "false_positive":
            text = "【复核误检】" + OBJECT_NAMES.get(kind, kind) + "。"
        elif status == "conflicting_state_unverified":
            text = "【待复核】" + OBJECT_NAMES.get(kind, kind) + "与界面状态冲突。"
        return base(e, "object_detected", subject, details, text)
    if typ in {"player_health_changed", "player_ammo_changed"}:
        field = "health" if "health" in typ else "ammo"
        return base(
            e,
            "state_changed",
            {"type": "pov_player"},
            {"field": field, "before": e.get("before"), "after": e.get("after")},
            f"{'血量' if field=='health' else '弹匣弹药'} {e.get('before')} → {e.get('after')}。",
        )
    if typ in SPIKE:
        state = SPIKE[typ]
        zh = {
            "dropped": "掉落",
            "planting": "正在安放",
            "planted": "已安放",
            "defused": "已拆除",
            "carrier_killed": "携包者被击杀",
        }[state]
        return base(
            e,
            "state_changed",
            {"type": "spike"},
            {"field": "status", "value": state, "source": "hud"},
            "Spike：" + zh + "。",
        )
    if typ in {"flash_screen_effect_observed", "pov_status_ui_observed"}:
        value = (
            "flashed"
            if typ.startswith("flash")
            else ev.get("label") or ev.get("status") or "unknown"
        )
        return base(
            e,
            "state_changed",
            {"type": "pov_player"},
            {"field": "effect", "value": value, "source": ev.get("channel") or "hud"},
            "当前视角：" + ("闪光效果" if value == "flashed" else str(value)) + "。",
        )
    if typ == "ability_destroyed_ui_observed":
        label = ev.get("label") or "unknown_destroyed"
        return base(
            e,
            "state_changed",
            {"type": "ability", "class": label.removesuffix("_destroyed")},
            {"field": "status", "value": "destroyed", "source": "hud"},
            "界面提示：" + label + "。",
        )
    raise ValueError("Unsupported event; add an explicit mapping: " + str(typ))


def normalize_events(records):
    ids = [e["record_id"] for e in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source record IDs")
    evidence = {e["record_id"]: deepcopy(e) for e in records}
    used = set()
    output = []
    initial = []
    by_incident = defaultdict(list)
    for e in records:
        if e.get("incident_id"):
            by_incident[(e["source_clip"], e["incident_id"])].append(e)
    for group in by_incident.values():
        deaths = [e for e in group if e.get("state_change") == "death_observed"]
        missing = [e for e in group if e.get("state_change") == "disappeared"]
        if len(deaths) == len(missing) == 1 and can_group(deaths[0], missing[0]):
            output.append(death_event(deaths[0], missing[0]))
            used.update(e["record_id"] for e in [deaths[0], missing[0]])
    # The same explicit banner supports dropped state + carrier killed; group only identical prompt evidence.
    groups = defaultdict(list)
    for e in records:
        if e["event"] in {
            "dropped_spike_state_observed",
            "spike_carrier_killed_ui_observed",
        }:
            ev = e.get("evidence") or {}
            if ev.get("text") and ev.get("source"):
                groups[
                    (
                        e["source_clip"],
                        round(e["timestamp"], 3),
                        ev["source"],
                        ev["text"],
                    )
                ].append(e)
    for group in groups.values():
        if len(group) == 2 and {e["event"] for e in group} == {
            "dropped_spike_state_observed",
            "spike_carrier_killed_ui_observed",
        }:
            primary = next(
                e for e in group if e["event"] == "dropped_spike_state_observed"
            )
            output.append(
                base(
                    primary,
                    "state_changed",
                    {"type": "spike"},
                    {
                        "field": "status",
                        "value": "dropped",
                        "reason": "carrier_killed",
                        "source": "hud",
                    },
                    "Spike 掉落：携包者被击杀。",
                    group,
                )
            )
            used.update(e["record_id"] for e in group)
    for e in records:
        if e["record_id"] in used:
            continue
        if e.get("state_change") == "historical_death_observed":
            initial.append(
                {
                    "source_clip": e["source_clip"],
                    "kind": "historical_killfeed_row",
                    "observed_at": round(e["timestamp"], 3),
                    "occurrence_time": None,
                    "subject": person(e.get("actor")),
                    "killer": e.get("killer"),
                    "weapon": (e.get("cause") or {}).get("weapon"),
                    "evidence_ids": [e["record_id"]],
                }
            )
        else:
            output.append(normalize_single(e))
    output.sort(
        key=lambda e: (
            e["source_clip"],
            e["timestamp"],
            e["emitted_at"],
            e["event"],
            e["evidence_ids"][0],
        )
    )
    counters = defaultdict(int)
    for e in output:
        counters[e["source_clip"]] += 1
        e["id"] = f"{e['source_clip']}:event:{counters[e['source_clip']]:03d}"
        validate_record(e)
    references = [eid for e in output + initial for eid in e["evidence_ids"]]
    if sorted(references) != sorted(ids):
        raise AssertionError("Source evidence was lost or duplicated")
    return {
        "schema_version": "8.0",
        "events": output,
        "initial_state": initial,
        "evidence": evidence,
        "counts": {
            "source_records": len(records),
            "events": len(output),
            "initial_observations": len(initial),
            "merged_groups": sum(len(e["evidence_ids"]) > 1 for e in output),
        },
    }


def validate_record(e):
    if e["event"] not in EVENTS:
        raise ValueError("Unknown normalized event")
    for name in ("confidence", "timestamp", "emitted_at"):
        if (
            isinstance(e[name], bool)
            or not isinstance(e[name], (int, float))
            or not math.isfinite(e[name])
        ):
            raise ValueError("Invalid numeric field: " + name)
    if not 0 <= e["confidence"] <= 1:
        raise ValueError("Invalid confidence")
    if not 0 <= e["timestamp"] <= e["emitted_at"]:
        raise ValueError("Invalid evidence time")
    if not e["evidence_ids"]:
        raise ValueError("Evidence references required")
    if e["event"] == "person_disappeared" and e["details"]["reason"] not in {
        "death",
        "left_view",
        "tracking_lost",
        "unknown",
    }:
        raise ValueError("Invalid disappearance reason")


def write_normalized(directory, bundle):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("events", "events.json"),
        ("initial_state", "initial_state.json"),
        ("evidence", "evidence.json"),
    ]:
        (directory / filename).write_text(
            json.dumps(bundle[key], ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    (directory / "summary.json").write_text(
        json.dumps(
            {"schema_version": bundle["schema_version"], **bundle["counts"]}, indent=2
        )
        + "\n"
    )
    fields = [
        "id",
        "source_clip",
        "timestamp",
        "event",
        "confidence",
        "subject_type",
        "team",
        "agent",
        "player",
        "track_id",
        "reason",
        "certainty",
        "weapon",
        "killer_agent",
        "killer_player",
        "description",
        "details",
        "evidence_ids",
        "emitted_at",
    ]
    with (directory / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in bundle["events"]:
            s = e["subject"]
            d = e["details"]
            k = d.get("killer") or {}
            row = {name: e.get(name) for name in fields}
            row.update(
                subject_type=s["type"],
                team=s.get("team"),
                agent=s.get("agent"),
                player=s.get("player"),
                track_id=s.get("track_id"),
                reason=d.get("reason"),
                certainty=d.get("certainty"),
                weapon=d.get("weapon"),
                killer_agent=k.get("agent"),
                killer_player=k.get("player"),
                details=json.dumps(d, ensure_ascii=False),
                evidence_ids=json.dumps(e["evidence_ids"]),
            )
            w.writerow(row)
