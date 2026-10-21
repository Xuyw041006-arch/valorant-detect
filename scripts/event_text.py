"""Shared bilingual event labels and descriptions."""

NAMES = {
    "enemy": "敌方",
    "teammate": "友方",
    "unknown": "阵营未知",
    "dropped_weapon": "地上武器",
    "dropped_spike": "掉落 Spike",
    "planted_spike": "已安放 Spike",
    "smoke": "烟雾",
    "wall_effect": "墙体效果",
    "fade_eye": "黑梦侦查眼",
    "cypher_tripwire_anchor": "Cypher 绊线锚点",
    "location_revealed": "位置被侦查",
    "flashed": "闪光影响",
    "haunt_destroyed": "黑梦侦查眼被破坏",
}
REASONS = {
    "death": "被击杀",
    "death_suspected": "疑似被击杀",
    "left_view_suspected": "疑似离开视野",
    "tracking_loss_suspected": "可能跟踪丢失",
    "unknown": "原因未知",
}
LABELS = {
    "person_appeared": "人物出现",
    "person_disappeared": "人物消失",
    "object_detected": "发现物品",
    "state_changed": "状态变化",
}

EN_NAMES = {
    "enemy": "Enemy",
    "teammate": "Ally",
    "unknown": "Team unknown",
    "dropped_weapon": "dropped weapon",
    "dropped_spike": "dropped Spike",
    "planted_spike": "planted Spike",
    "smoke": "smoke",
    "wall_effect": "wall effect",
    "fade_eye": "Fade Haunt",
    "cypher_tripwire_anchor": "Cypher tripwire anchor",
    "location_revealed": "location revealed",
    "flashed": "flash effect",
    "haunt_destroyed": "Haunt destroyed",
}
EN_LABELS = {
    "person_appeared": "Person appeared",
    "person_disappeared": "Person disappeared",
    "object_detected": "Object detected",
    "state_changed": "State changed",
}


def describe_en(e):
    s = e["subject"]
    d = e.get("details", {})
    kind = e["event"]
    if kind.startswith("person_"):
        actor = (
            EN_NAMES.get(s["team"], s["team"])
            + " "
            + (s.get("agent") or "unknown agent").title()
        )
        if kind == "person_appeared":
            return actor + " appeared."
        if s["team"] == "enemy" and d["reason"] != "death":
            return actor + " disappeared (non-death category)."
        if d["reason"] in ("death", "death_suspected"):
            return (
                actor
                + (" killed" if d["reason"] == "death" else " possibly killed")
                + " by "
                + (d.get("killer") or "unknown agent").title()
                + " with "
                + (d.get("weapon") or "unknown weapon")
                + "."
            )
        reason = {
            "unknown": "cause unknown",
            "left_view_suspected": "likely left the view",
            "tracking_loss_suspected": "possible tracking loss",
        }.get(d["reason"], d["reason"])
        return actor + " disappeared: " + reason + "."
    if kind == "object_detected":
        text = (
            ("Reobserved " if d.get("reobserved") else "Detected ")
            + EN_NAMES.get(s["type"], s["type"])
            + "."
        )
        return (
            "Possible dropped Spike; state unconfirmed."
            if d.get("review") == "conflicting_state_unverified"
            else text
        )
    if "field" in d:
        return (
            "Health" if d["field"] == "health" else "Magazine ammo"
        ) + f" {d['from']} → {d['to']}."
    if "effect" in d:
        return "POV: " + EN_NAMES.get(d["effect"], d["effect"]) + "."
    text = (
        ("Spike" if s["type"] == "spike" else EN_NAMES.get(s["type"], s["type"]))
        + ": "
        + str(d.get("status")).replace("_", " ")
    )
    if d.get("reason") == "carrier_killed":
        text += " (carrier killed)"
    return text + "."


def describe(e, lang="zh"):
    if lang == "en":
        return describe_en(e)
    s = e["subject"]
    d = e.get("details", {})
    kind = e["event"]
    if kind.startswith("person_"):
        actor = (
            NAMES.get(s["team"], s["team"])
            + " "
            + (s.get("agent") or "未知特工").title()
        )
        if kind == "person_appeared":
            return "发现" + actor + "。"
        if s["team"] == "enemy" and d["reason"] != "death":
            return actor + "：非死亡消失。"
        if d["reason"] in ("death", "death_suspected"):
            return (
                actor
                + " "
                + REASONS[d["reason"]]
                + "；击杀者 "
                + (d.get("killer") or "未知").title()
                + "，武器 "
                + (d.get("weapon") or "未知")
                + "。"
            )
        return actor + " 消失：" + REASONS.get(d["reason"], d["reason"]) + "。"
    if kind == "object_detected":
        text = (
            ("再次发现" if d.get("reobserved") else "发现")
            + NAMES.get(s["type"], s["type"])
            + "。"
        )
        if d.get("review") == "conflicting_state_unverified":
            text = "疑似掉落 Spike，状态待核实。"
        return text
    if "field" in d:
        return (
            "血量" if d["field"] == "health" else "弹匣弹药"
        ) + f" {d['from']} → {d['to']}。"
    if "effect" in d:
        return "当前视角：" + NAMES.get(d["effect"], d["effect"]) + "。"
    text = (
        ("Spike" if s["type"] == "spike" else NAMES.get(s["type"], s["type"]))
        + "："
        + {
            "dropped": "掉落",
            "planting": "正在安放",
            "planted": "已安放",
            "defused": "已拆除",
            "carrier_killed": "携包者被击杀",
            "destroyed": "被破坏",
        }.get(d.get("status"), str(d.get("status")))
    )
    if d.get("reason") == "carrier_killed":
        text += "（携包者被击杀）"
    return text + "。"
