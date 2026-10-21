#!/usr/bin/env python3
"""Create a standalone, offline drag/resize HUD/minimap calibration page.

python scripts/calibrate_hud.py --video media/extensions/weapon_demo.mp4 --time 2 \
  --config configs/integrated_weapon.json --output outputs/calibration/weapon.html

The page embeds one decoded frame. It downloads a JSON configuration, never
uploads images or installs a server. Inspect several times before relying on a
profile. Unknown map mode conservatively breaks minimap identity continuity.
"""

from __future__ import annotations
import argparse
import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

DEFAULT_REGIONS = [
    ("Health", ["hud", "health_roi"], [0.29, 0.90, 0.35, 0.99]),
    ("Magazine ammo", ["hud", "ammo_roi"], [0.66, 0.90, 0.715, 0.99]),
    ("Minimap", ["tactical", "minimap", "roi"], [0.008, 0.045, 0.231, 0.435]),
    (
        "Planted icon",
        ["tactical", "ui", "spike_roi"],
        [920 / 1920, 17 / 1080, 1000 / 1920, 92 / 1080],
    ),
    (
        "Round / defuse text",
        ["tactical", "ui", "banner_roi"],
        [0.425, 0.22, 0.575, 0.265],
    ),
    (
        "Death report (observation only)",
        ["tactical", "ui", "death_roi"],
        [0.80, 0.28, 0.99, 0.325],
    ),
    (
        "Killfeed",
        ["tactical", "ui", "killfeed_roi"],
        [0.77, 96 / 1080, 0.995, 260 / 1080],
    ),
    (
        "Spike down / carrier text",
        ["tactical", "ui", "spike_text_roi"],
        [0.3, 0.10, 0.7, 0.35],
    ),
    (
        "Ally planting progress UI",
        ["tactical", "ui", "planting_roi"],
        [0.3, 0.10, 0.7, 0.25],
    ),
]


def region_value(config, path, default):
    value = config
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return list(default)
        value = value[part]
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in value
        )
        or value[0] >= value[2]
        or value[1] >= value[3]
    ):
        raise ValueError("Invalid normalized ROI: " + ".".join(path))
    return list(value)


def build_page(image, config, metadata):
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Unable to encode calibration frame")
    config = deepcopy(config)
    if "tactical" not in config and any(k in config for k in ("minimap", "ui")):
        config = {"tactical": config}
    config.setdefault("tactical", {}).setdefault("minimap", {})
    payload = {
        "config": config,
        "metadata": metadata,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "regions": [
            {"label": name, "path": path, "roi": region_value(config, path, default)}
            for name, path, default in DEFAULT_REGIONS
        ],
    }
    # Escape script termination even if an imported name contains HTML syntax.
    serialized = (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return HTML.replace("__PAYLOAD__", serialized).replace(
        "__IMAGE__", base64.b64encode(encoded).decode("ascii")
    )


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HUD / 小地图校准</title><style>
*{box-sizing:border-box}body{margin:0;background:#111722;color:#e9eef7;font:15px system-ui,sans-serif}header{padding:20px 26px;border-bottom:1px solid #334055}h1{font-size:23px;margin:0 0 6px}p{margin:5px 0;line-height:1.5;color:#afbed2}main{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:18px;padding:20px}section,aside{background:#1a2434;border:1px solid #334055;border-radius:12px;padding:16px}canvas{width:100%;height:auto;display:block;background:#080b11;touch-action:none;border-radius:5px}#preview{max-height:150px;width:auto;max-width:100%;margin:auto;image-rendering:auto}label{display:block;margin:13px 0 5px;color:#c8d7ed}select,input,textarea,button{font:inherit;border-radius:6px;border:1px solid #50637e;background:#101a29;color:#eef4ff;padding:8px}select,input,textarea{width:100%}button{cursor:pointer;margin:4px 4px 4px 0}button.primary{background:#2868c7;border-color:#4b91f5}button:hover{filter:brightness(1.2)}.row{display:flex;gap:8px}.row>*{flex:1}small{display:block;color:#a9bdd6;font-size:12px;line-height:1.5}#status{color:#8de4ba;min-height:25px}.warn{color:#ffd28b}input[type=checkbox]{width:auto}textarea{font:12px ui-monospace,monospace;min-height:140px}details{margin-top:16px}summary{cursor:pointer;color:#bfd5f6}@media(max-width:1000px){main{grid-template-columns:1fr}aside{order:-1}}
</style><header><h1>HUD / 小地图校准</h1><p>选择区域，在图上移动或拖动四角；也可点「重新框选」后拖出新区域。所有坐标保存为 0–1 的 ROI。</p><small id="source"></small></header>
<main><section><canvas id="canvas"></canvas><p id="status"></p><div class="row"><button id="redraw">重新框选</button><button id="restore">恢复此区域</button><button id="toggle">隐藏其他框</button></div><p class="warn">小地图坐标只表示当前可见 UI。旋转、缩放、局部地图不能直接解释为真实世界位置。颜色与模板需逐录像核验。</p><details open><summary>所选区域预览</summary><canvas id="preview"></canvas></details></section>
<aside><label for="region">当前区域</label><select id="region"></select><label for="coords">归一化 x1, y1, x2, y2</label><input id="coords" aria-describedby="coordStatus" spellcheck="false"><button id="applyCoords" type="button">应用坐标</button><small id="coordStatus" aria-live="polite">输入四个 0–1 数字；有效坐标会即时应用。</small><small id="pixels"></small><label for="profile">配置名称</label><input id="profile"><div class="row"><div><label for="mode">地图朝向</label><select id="mode"><option value="unknown">未知（不保持地图 ID）</option><option value="fixed">固定</option><option value="rotating">随视角旋转</option></select></div><div><label for="coverage">覆盖范围</label><select id="coverage"><option value="unknown">未知</option><option value="full">全图</option><option value="partial">局部</option></select></div></div>
<label for="ocrscale">OCR 放大倍数（0.5–4）</label><input id="ocrscale" type="number" min="0.5" max="4" step="0.5"><label for="interval">战术 HUD 采样间隔（秒）</label><input id="interval" type="number" min="0.01" step="0.05"><div class="row"><div><label for="pitch">击杀条行距（1080p 像素）</label><input id="pitch" type="number" min="1"></div><div><label for="rowheight">击杀条高度（1080p 像素）</label><input id="rowheight" type="number" min="1"></div></div><label for="rowoffset">首行偏移（1080p 像素）</label><input id="rowoffset" type="number" min="0"><label><input id="keepTemplates" type="checkbox"> 我已确认原模板与队色适用于当前录像</label><small class="warn">未勾选时导出会移除已有地图/下包图标模板，保留 OCR 与未知颜色观测。头像队色映射和地图大小改变后需要重新校验模板。</small><button id="download" class="primary">下载配置 JSON</button><button id="copy">复制 JSON</button><details><summary>导出内容</summary><textarea id="json" readonly></textarea></details><small>离线文件，不上传图像。下载后使用 --tactical-config 指定配置；health/ammo ROI 需并入主 --config。请在同一视频的多个时刻检查。</small></aside></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
'use strict';const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),image=new Image();
const canvas=$('canvas'),ctx=canvas.getContext('2d'),preview=$('preview'),pctx=preview.getContext('2d');canvas.width=D.width;canvas.height=D.height;
const original=structuredClone(D.regions),regions=D.regions;let index=0,drag=null,drawNew=false,showOthers=true;
$('source').textContent=`${D.metadata.video_name} | ${D.width} × ${D.height} | requested ${D.metadata.requested_seconds}s | decoded ${D.metadata.decoded_seconds ?? 'unknown'}s`;
regions.forEach((r,i)=>{const opt=document.createElement('option');opt.value=i;opt.textContent=r.label;$('region').appendChild(opt)});
const tactical=D.config.tactical||{},map=tactical.minimap||{},ui=tactical.ui||{};
$('profile').value=tactical.profile_name||D.metadata.video_name.replace(/\.[^.]+$/,'')+'_calibrated';$('mode').value=map.mode||'unknown';$('coverage').value=map.coverage||'unknown';$('interval').value=tactical.sample_interval_seconds||.2;$('ocrscale').value=ui.ocr_scale||1;$('pitch').value=ui.killfeed_row_pitch_at_1080||39;$('rowheight').value=ui.killfeed_row_height_at_1080||34;$('rowoffset').value=ui.killfeed_row_offset_at_1080||0;
function setPath(obj,path,value){for(const part of path.slice(0,-1)){if(!obj[part]||typeof obj[part]!=='object')obj[part]={};obj=obj[part]}obj[path.at(-1)]=value}
function exported(){const out=structuredClone(D.config);for(const r of regions)setPath(out,r.path,r.roi.slice());const t=out.tactical||(out.tactical={});t.enabled=true;t.profile_name=$('profile').value.trim()||'calibrated';t.calibration_aspect_ratio=D.width/D.height;t.sample_interval_seconds=+$('interval').value;t.minimap=t.minimap||{};t.minimap.mode=$('mode').value;t.minimap.coverage=$('coverage').value;t.ui=t.ui||{};t.ui.ocr_scale=+$('ocrscale').value;t.ui.killfeed_row_pitch_at_1080=+$('pitch').value;t.ui.killfeed_row_height_at_1080=+$('rowheight').value;t.ui.killfeed_row_offset_at_1080=+$('rowoffset').value;if(!$('keepTemplates').checked){t.minimap.templates=[];delete t.ui.spike_template_path;if(t.minimap.detector)t.minimap.detector.team_colors_calibrated=false}out.calibration={...D.metadata,width:D.width,height:D.height,coordinate_space:'normalized_roi',templates_retained_after_manual_check:$('keepTemplates').checked};return out}
function update(syncInput=true){const r=regions[index],b=r.roi;if(syncInput){$('coords').value=b.map(String).join(', ');$('coords').setAttribute('aria-invalid','false');$('coordStatus').textContent='当前区域坐标已同步；输入有效四值可即时应用。';}$('pixels').textContent=`像素: ${Math.floor(b[0]*D.width)}, ${Math.floor(b[1]*D.height)}, ${Math.ceil(b[2]*D.width)}, ${Math.ceil(b[3]*D.height)}`;$('json').value=JSON.stringify(exported(),null,2);render()}
function render(){if(!image.complete)return;ctx.drawImage(image,0,0);for(let i=0;i<regions.length;i++){if(!showOthers&&i!==index)continue;const b=regions[i].roi,x=b[0]*D.width,y=b[1]*D.height,w=(b[2]-b[0])*D.width,h=(b[3]-b[1])*D.height;ctx.strokeStyle=i===index?'#ffda5e':'#56b5ff';ctx.lineWidth=i===index?3:1;ctx.strokeRect(x,y,w,h);ctx.fillStyle=ctx.strokeStyle;ctx.font=`${Math.max(13,D.width/110)}px sans-serif`;ctx.fillText(regions[i].label,x+3,Math.max(16,y-5));if(i===index){const z=Math.max(8,D.width/180);for(const [cx,cy] of [[x,y],[x+w,y],[x,y+h],[x+w,y+h]])ctx.fillRect(cx-z/2,cy-z/2,z,z)}}const b=regions[index].roi,w=Math.max(1,Math.round((b[2]-b[0])*D.width)),h=Math.max(1,Math.round((b[3]-b[1])*D.height));preview.width=w;preview.height=h;pctx.drawImage(image,b[0]*D.width,b[1]*D.height,w,h,0,0,w,h)}
function point(e){const r=canvas.getBoundingClientRect();return [Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))]}
canvas.addEventListener('pointerdown',e=>{const p=point(e),b=regions[index].roi;canvas.setPointerCapture(e.pointerId);if(drawNew){drag={kind:'new',start:p};return}const tol=12/canvas.getBoundingClientRect().width;const corners=[[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]];let corner=corners.findIndex(c=>Math.hypot(c[0]-p[0],(c[1]-p[1])*D.height/D.width)<tol);if(corner>=0)drag={kind:'corner',corner,base:b.slice()};else if(p[0]>=b[0]&&p[0]<=b[2]&&p[1]>=b[1]&&p[1]<=b[3])drag={kind:'move',start:p,base:b.slice()};else{$('status').textContent='请拖动当前框，或点击「重新框选」。'}});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const p=point(e),epsilon=1/Math.max(D.width,D.height);let b;if(drag.kind==='new')b=[Math.min(p[0],drag.start[0]),Math.min(p[1],drag.start[1]),Math.max(p[0],drag.start[0]),Math.max(p[1],drag.start[1])];else if(drag.kind==='move'){b=drag.base.slice();const dx=Math.max(-b[0],Math.min(1-b[2],p[0]-drag.start[0])),dy=Math.max(-b[1],Math.min(1-b[3],p[1]-drag.start[1]));b=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]}else{b=drag.base.slice();if(drag.corner%2===0)b[0]=Math.min(p[0],b[2]-epsilon);else b[2]=Math.max(p[0],b[0]+epsilon);if(drag.corner<2)b[1]=Math.min(p[1],b[3]-epsilon);else b[3]=Math.max(p[1],b[1]+epsilon)}if(b[2]-b[0]>=epsilon&&b[3]-b[1]>=epsilon)regions[index].roi=b;update()});
canvas.addEventListener('pointerup',()=>{drag=null;drawNew=false;$('status').textContent='区域已更新；下载 JSON 保存配置。'});canvas.addEventListener('pointercancel',()=>{drag=null;drawNew=false});
$('region').onchange=()=>{index=+$('region').value;update()};$('redraw').onclick=()=>{drawNew=true;$('status').textContent='在图上拖出新的左上角与右下角。'};$('restore').onclick=()=>{regions[index].roi=original[index].roi.slice();update()};$('toggle').onclick=()=>{showOthers=!showOthers;render()};function applyCoordinates(explicit=false){const text=$('coords').value.trim().replace(/^\[([\s\S]*)\]$/,'$1'),b=text.split(/[,，;；\s]+/).filter(Boolean).map(Number);if(b.length!==4||b.some(v=>!Number.isFinite(v)||v<0||v>1)||b[0]>=b[2]||b[1]>=b[3]){$('coordStatus').textContent='未应用：请输入四个 0–1 数字，且 x1<x2、y1<y2。';$('coords').setAttribute('aria-invalid','true');if(explicit)$('status').textContent='坐标尚未应用，未导出。请修正后点「应用坐标」。';return false}regions[index].roi=b;$('coords').setAttribute('aria-invalid','false');$('coordStatus').textContent='已应用到图框、像素坐标和导出 JSON。';update(false);if(explicit)$('status').textContent='坐标已应用；下载 JSON 保存配置。';return true}
$('coords').addEventListener('input',()=>applyCoordinates(false));$('coords').addEventListener('change',()=>applyCoordinates(true));$('coords').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();applyCoordinates(true)}});$('applyCoords').onclick=()=>applyCoordinates(true);
for(const id of ['profile','mode','coverage','interval','ocrscale','pitch','rowheight','rowoffset','keepTemplates'])$(id).onchange=()=>update(false);
function valid(){const o=exported(),u=o.tactical.ui;return u.ocr_scale>=.5&&u.ocr_scale<=4&&o.tactical.sample_interval_seconds>0&&u.killfeed_row_pitch_at_1080>0&&u.killfeed_row_height_at_1080>0&&u.killfeed_row_offset_at_1080>=0&&[u.ocr_scale,o.tactical.sample_interval_seconds,u.killfeed_row_pitch_at_1080,u.killfeed_row_height_at_1080,u.killfeed_row_offset_at_1080].every(Number.isFinite)}
$('download').onclick=()=>{if(!applyCoordinates(true))return;if(!valid()){$('status').textContent='采样间隔和行尺寸必须为正数，偏移不得为负。';return}const blob=new Blob([JSON.stringify(exported(),null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=($('profile').value||'hud_profile').replace(/[^a-z0-9_-]/gi,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);$('status').textContent='配置已导出。请在其他时间点复核。'};
$('copy').onclick=async()=>{if(!applyCoordinates(true)||!valid())return;try{await navigator.clipboard.writeText($('json').value);$('status').textContent='JSON 已复制。'}catch(e){$('json').focus();$('json').select();$('status').textContent='浏览器限制剪贴板；已选中 JSON，可手动复制。'}};
image.onload=()=>update();image.src='data:image/jpeg;base64,__IMAGE__';
</script></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--time", type=float, default=0, help="Requested local video time in seconds"
    )
    parser.add_argument(
        "--config", type=Path, help="Optional existing full or tactical config"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/hud_calibration.html")
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.time) or args.time < 0:
        parser.error("--time must be finite and nonnegative")
    if not args.video.is_file():
        parser.error("--video must be a local video file")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        parser.error("Config root must be a JSON object")
    import cv2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = cap.read()
    pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    cap.release()
    if not ok:
        raise RuntimeError(
            "Cannot decode requested frame; choose a time inside the clip"
        )
    digest = hashlib.sha256()
    with args.video.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "video_name": args.video.name,
        "video_sha256": digest.hexdigest(),
        "requested_seconds": args.time,
        "decoded_seconds": pts if math.isfinite(pts) and pts >= 0 else None,
        "decoded_frame_index": (
            int(frame_index)
            if math.isfinite(frame_index) and frame_index >= 0
            else None
        ),
        "time_source": "OpenCV decoder reported timestamp; inspect frame and do not assume exact seeking",
        "calibration_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(frame, config, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "html": str(args.output.resolve()),
                "frame": metadata,
                "resolution": [frame.shape[1], frame.shape[0]],
                "offline": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
