from pathlib import Path
import json, argparse, collections
from event_categories import category, MAINS
from event_text import describe
from presentation_timing import build_event_frames

R = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    lang = ap.parse_args().lang
    ix = 0 if lang == "zh" else 1
    es = json.loads((R / "events.json").read_text())
    records = json.loads((R / "evidence.json").read_text())["records"]
    data = []
    subs = {}
    manifest = json.loads((R / "presentation/edit_manifest.json").read_text())
    plan = build_event_frames(es, records, manifest)
    for i, (e, x) in enumerate(zip(es, records)):
        k, names = category(e)
        subs[k] = {"id": k, "main": e["event"], "name": names[ix]}
        data.append(
            {
                "index": i,
                "event": e,
                "text": describe(e, lang),
                "sub": k,
                "seek_time": plan[i]["seek_time"],
                "seek_frame": plan[i]["frame"],
            }
        )
    pairs = {
        "title": ("VALORANT · 游戏事件演示", "VALORANT · Gameplay Events"),
        "subtitle": (
            "01:19.9 · 六段游戏实录 · 中英文演示",
            "01:19.9 · Six gameplay clips · Bilingual demo",
        ),
        "language_link": ("English", "中文"),
        "predicted": ("检测结果", "Detections"),
        "clean": ("原始画面", "Original video"),
        "timeline": ("时间轴", "Timeline"),
        "hint": (
            "选择事件或点击时间轴标记，即可定位并暂停。",
            "Select an event or timeline marker to seek and pause.",
        ),
        "download": ("下载标注视频", "Download video"),
        "jsondownload": ("下载事件 JSON", "Download event JSON"),
        "selection_note": (
            "已排除明确误检；未确认信息会标注为未知或待核实。",
            "Confirmed false detections removed; unresolved information is marked unknown or unconfirmed.",
        ),
        "category": ("事件大类", "Category"),
        "subtype": ("具体事件", "Specific event"),
        "selected": ("事件详情", "Event details"),
        "empty": ("选择一条事件查看。", "Select an event to inspect it."),
        "event_record": ("查看结构化记录", "View structured record"),
        "all": ("全部", "All"),
        "paused": ("已暂停", "Paused"),
        "playing": ("播放中", "Playing"),
        "events": ("条事件", "events"),
    }
    t = {k: v[ix] for k, v in pairs.items()}
    meta = {
        "text": t,
        "duration": 79.9,
        "video": f"presentation/valorant_demo_{lang}.mp4?v=event-frame-v1",
        "other": "DEMO_en.html" if lang == "zh" else "DEMO.html",
        "mains": [{"id": k, "name": v[ix]} for k, v in MAINS.items()],
        "subs": sorted(
            subs.values(), key=lambda s: (list(MAINS).index(s["main"]), s["id"])
        ),
    }
    gallery = json.loads((R / "docs/presentation/skills/gallery.json").read_text())
    gtext = {
        "zh": {
            "gallery_title": "技能与道具识别",
            "gallery_intro": "本项目还支持识别多种英雄的技能与道具。为控制视频长度，未将更多技能片段加入上方视频；此处选取几个带有技能的画面，展示 GT 标注与模型预测。",
            "gallery_hint": "选择技能查看 GT 与预测对比，点击图片可放大。",
            "gt_title": "GT 标注",
            "pred_title": "模型预测",
            "video_report": "视频 Precision / Recall",
            "report_url": "EVALUATION.html",
        },
        "en": {
            "gallery_title": "Ability and utility detection",
            "gallery_intro": "This project also supports detecting abilities and utility from multiple agents. To keep the video concise, additional ability clips were not included above. The frames below show selected ability examples with ground-truth annotations and model predictions.",
            "gallery_hint": "Select an ability to compare GT and prediction. Click an image to enlarge.",
            "gt_title": "Ground truth",
            "pred_title": "Model prediction",
            "video_report": "Video Precision / Recall",
            "report_url": "EVALUATION_en.html",
        },
    }[lang]
    t.update(gtext)
    meta["text"] = t
    meta["lang"] = lang
    html = (R / "scripts/hr_viewer_template.html").read_text()
    for k, v in {
        **t,
        "lang": lang,
        "other": meta["other"],
        "video": meta["video"],
    }.items():
        html = html.replace("{{" + k + "}}", v)
    html = html.replace(
        "__GALLERY__", json.dumps(gallery, ensure_ascii=False).replace("</", "<\\/")
    )
    html = html.replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    ).replace("__META__", json.dumps(meta, ensure_ascii=False))
    (R / "docs" / ("DEMO.html" if lang == "zh" else "DEMO_en.html")).write_text(html)
    (R / "docs/presentation" / f"categories_{lang}.json").write_text(
        json.dumps(
            {
                "main_categories": meta["mains"],
                "subcategories": meta["subs"],
                "counts": dict(collections.Counter(x["sub"] for x in data)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(lang, len(data), "events", len(subs), "subtypes")
    if lang == "en":
        (R / "docs/index.html").write_text(html)


if __name__ == "__main__":
    main()
