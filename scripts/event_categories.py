"""Public event categories and subcategories."""

MAINS = {
    "person_appeared": ("人物出现", "Person appeared"),
    "person_disappeared": ("人物消失（含死亡）", "Person disappeared / death"),
    "object_detected": ("物品与技能", "Objects and abilities"),
    "state_changed": ("状态变化", "State changes"),
}
TEAMS = {
    "enemy": ("敌方", "Enemy"),
    "teammate": ("友方", "Ally"),
    "unknown": ("阵营未知", "Unknown team"),
}
OBJECTS = {
    "dropped_weapon": ("掉落武器", "Dropped weapon"),
    "dropped_spike": ("掉落 Spike", "Dropped Spike"),
    "planted_spike": ("已安放 Spike", "Planted Spike"),
    "smoke": ("烟雾", "Smoke"),
    "wall_effect": ("墙体效果", "Wall effect"),
    "fade_eye": ("黑梦侦查眼", "Fade Haunt"),
    "cypher_tripwire_anchor": ("Cypher 绊线锚点", "Cypher tripwire anchor"),
}
STATUS = {
    "dropped": ("Spike 掉落", "Spike dropped"),
    "planting": ("正在安放 Spike", "Spike planting"),
    "planted": ("Spike 已安放", "Spike planted"),
    "defused": ("Spike 已拆除", "Spike defused"),
    "carrier_killed": ("携包者被击杀", "Spike carrier killed"),
    "destroyed": ("道具被破坏", "Ability destroyed"),
}


def category(e):
    k = e["event"]
    s = e["subject"]
    d = e.get("details", {})
    if k == "person_appeared":
        team = s["team"]
        return "appearance_" + team, (
            TEAMS[team][0] + "出现",
            TEAMS[team][1] + " appeared",
        )
    if k == "person_disappeared":
        reason = d["reason"]
        if reason == "death":
            team = s["team"]
            return "death_" + team, (
                TEAMS[team][0] + "被击杀",
                TEAMS[team][1] + " killed",
            )
        if s["team"] == "enemy":
            return "enemy_non_death_disappearance", (
                "非死亡消失",
                "Non-death disappearance",
            )
        names = {
            "death_suspected": ("疑似被击杀", "Possible death"),
            "left_view_suspected": ("疑似离开画面", "Likely left view"),
            "tracking_loss_suspected": ("疑似跟踪丢失", "Possible tracking loss"),
            "unknown": ("消失原因未确认", "Disappearance: unknown cause"),
        }
        return reason, names[reason]
    if k == "object_detected":
        return "object_" + s["type"], OBJECTS.get(
            s["type"], (s["type"], s["type"].replace("_", " ").title())
        )
    if "field" in d:
        up = d["to"] > d["from"]
        field = d["field"]
        return field + ("_increase" if up else "_decrease"), (
            ("血量" if field == "health" else "弹药") + ("增加" if up else "减少"),
            ("Health" if field == "health" else "Ammo")
            + (" increased" if up else " decreased"),
        )
    if "status" in d:
        return "status_" + d["status"], STATUS.get(
            d["status"], (d["status"], d["status"])
        )
    return "effect_" + d["effect"], {
        "flashed": ("闪光影响", "Flash effect"),
        "location_revealed": ("位置被侦查", "Location revealed"),
    }.get(d["effect"], (d["effect"], d["effect"].replace("_", " ").title()))
