# 游戏视频事件检测 — VALORANT

**[中文在线演示](https://xuyw041006-arch.github.io/valorant-detect/DEMO.html) · [English demo](https://xuyw041006-arch.github.io/valorant-detect/) · [视频评估](https://xuyw041006-arch.github.io/valorant-detect/EVALUATION.html)**

[English](README_en.md) | [简体中文](README_zh.md)

本项目使用 Python，将游戏录像中的视觉信息转换为目标轨迹和结构化事件。系统检测敌方/友方人物、掉落/已安放的 Spike、地面武器，并结合 HUD、击杀栏和部分技能观察。系统不读取游戏内存，也不使用实时对局数据接口。

提交材料包括 [Python 源码](src/valorant_events/)、本中英文 README、[样例 JSON](events.json)/[CSV](events.csv)，以及[中文](docs/presentation/valorant_demo_zh.mp4)/[英文](docs/presentation/valorant_demo_en.mp4)标注视频。[交互 Demo](docs/DEMO.html)包含 128 条复核后的展示事件，提供四个大类、21 个具体事件筛选项。可直接打开在线 Demo，也可用下方本地服务器运行。

## 采用的方法

流水线依次读取视频帧、检测相关目标、进行跨帧关联、读取校准后的 HUD 区域，再通过时间规则生成事件。特工身份、武器、小地图符号、技能和烟雾由独立模型处理，使各任务可以使用合适的标签与阈值，代价是增加推理开销和校准工作。

全部六个演示切片统一启用短间隔人物重关联：根据身体外观、运动校验后的位置和唯一候选接回断裂轨迹，**不再把阵营作为合并条件**。原始阵营观察仍用于事件描述；同时可见的两个人和不同物体类别不能被合并。局部双向光流跟随人物自身运动，补充仅根据镜头运动进行的位置估计。

离线演示的事件文字、检测框和点击暂停使用同一套帧编号，选择事件即可看到对应的标注帧。

输出包括标注视频、逐帧记录及 JSON/CSV 事件。最终 **79.9 秒双语演示**由六段游戏实录合并，切换处重置状态，支持分类筛选、点击事件定位暂停及原片对照。视频下方另有选取的技能截图，对比参考标注（GT）和模型预测。

## 选择的模型与工具

| 模型/工具 | 用途与选择原因 |
|---|---|
| 微调后的 YOLO26s 与专项模型 | 复用预训练视觉特征识别游戏目标；将身份和小物体任务分别处理，无需从零训练模型。 |
| ByteTrack | 关联跨帧检测结果，提供用于事件推导的轨迹编号。 |
| EasyOCR 与离线图标模板 | 读取血量、弹药及游戏提示，利用头像和枪械图标辅助解释击杀栏。 |
| OpenCV、Python、HTML/JavaScript 本地演示 | 完成视频处理、区域裁剪、时间逻辑、结构化导出和交互复核。 |

微调使用 Colab A100。本仓库只保留最终配置使用的 6 个权重、推理与演示代码、必要评估输入；训练脚本、笔记本、原始数据集、旧模型、历史实验报告、测试代码及验证日志不上传。来源与许可见[第三方说明](THIRD_PARTY_NOTICES.md)，模型列表见[权重清单](weights/manifest.json)。

## 如何从检测结果推导事件

系统先通过多帧确认、标签投票、短间隔轨迹关联和去重稳定观察，再输出事件。

| 事件 | 推导逻辑 |
|---|---|
| `person_appeared` | 确认新出现的可见人物轨迹；有足够证据时附加阵营与特工身份。 |
| `person_disappeared` | 轨迹持续缺失 0.8 秒，或新的击杀栏记录确认死亡。死亡使用 `details.reason = "death"`，证据支持时补充击杀者和枪械。 |
| `object_detected` | 观察到有证据支持的武器、Spike 或技能物体。物体消失不能直接推断为拾取。 |
| `state_changed` | 确认 HUD 数值变化，或 Spike/状态/效果证据表明进入新状态。数值变化保存 `field`、`from`、`to`。 |

死亡归入人物消失。敌方没有确认死亡的消失在界面统一显示为“非死亡消失”，但这不代表确认其存活。不确定身份使用 `null` 或 `unknown`；轨迹编号不是跨片段永久有效的玩家身份。

事件包含题目要求的 `timestamp`、`event`、`confidence`，并补充 `subject` 和按需提供的 `details`。时间戳为合并演示视频中的秒数；置信分数不等于经过校准的正确概率。以下为 [events.json](events.json) 中的真实记录：

```json
{
  "timestamp": 1.4,
  "event": "person_disappeared",
  "confidence": 0.992,
  "subject": {"team": "teammate", "agent": "sage"},
  "details": {"reason": "death", "killer": "waylay", "weapon": "Ghost"}
}
```

表格输出和完整格式约束分别见 [events.csv](events.csv)、[event_schema.json](event_schema.json)。

## 已知失败情况

- 遮挡、低分辨率、穿墙轮廓和尸体可能导致误检、漏检及轨迹断裂；断裂可能产生重复的出现/消失事件。
- 小型武器与特工身份可能混淆。击杀栏识别依赖清晰的头像、图标和文字，不确定字段保持未知。
- HUD 和小地图布局需要校准。地图位置仅相对于图像，不代表世界坐标或隐藏敌人位置。小地图技能类型分类器因旋转测试不稳定而关闭。
- 部分技能只有选取的单帧证据，不能据此认定所有技能或烟雾类型均可靠。一条未解决的 Spike 物体/状态冲突仍标记为待确认。
- 当前为离线处理，不保证实时速度。演示由六段片段组成，最后一段分辨率较低，视频不含音轨。

## Assessment：评分标准逐项对应

下表对应原题 **Evaluation** 部分的五项评分标准，说明本项目如何实现，以及评审可以查看什么证据。

| 原题评分标准 | 本项目的对应实现与证据 |
|---|---|
| **Detection/tracking correctness（检测与跟踪正确性）** | 可通过标注视频、原片对照及[逐帧记录](examples/render_inputs/p006/tracks.jsonl)检查目标框、类别和 ID 连续性。ByteTrack 与[视觉重关联](src/valorant_events/stitching.py)维持片段内轨迹。最终视频尚无完整的逐帧目标框及身份 GT，因此未量化最终视频的框级检测准确率和身份跟踪准确率。 |
| **Logic used to derive gameplay events（事件推导逻辑）** | 四类事件采用明确的确认、消失和状态变化规则；死亡需要击杀栏证据，不把物体消失直接认定为拾取。证据见[基础事件逻辑](src/valorant_events/events.py)、[融合事件描述](src/valorant_events/story_events.py)和[格式约束](event_schema.json)。 |
| **Code quality and structure（代码质量与结构）** | 检测、HUD、战术观察、专项模型和事件规范化分为独立模块，提供六套配置；视频、JSON/CSV 与静态演示共享事件定义。验证在仓库外执行，测试代码与日志不上传。 |
| **Evaluation methodology（评估方法）** | 将 [AI 生成的事件参考标注](evaluation/final_video/truth.json)与全部未经删改的预测，按固定[时间/语义协议](evaluation/final_video/protocol.json)比较；提供 TP/FP/FN、分类结果及匹配输入数据。作者初步复核及标注不确定性见下文，整理后的展示事件不参与评分。 |
| **Understanding of limitations and trade-offs（对局限与取舍的理解）** | 已列出具体失败情况；多帧确认减少闪烁但增加延迟，专项模型扩展能力但增加推理成本，逐布局校准改善 HUD 读取但降低通用性。明确区分开发片段评估与独立测试。 |

## Bonus：加分项逐项对应

| 原题加分项 | 实现、证据与适用范围 |
|---|---|
| **OCR from the gameplay UI（游戏界面 OCR）** | [HUD OCR](src/valorant_events/hud.py)读取血量/弹药，[战术模块](src/valorant_events/tactical.py)读取相关提示和击杀栏文字；Demo 展示数值变化与状态事件。新布局或语言需校准。 |
| **Temporal smoothing（时间平滑）** | 多帧确认、标签投票、短间隔重关联、消失等待与去重，统一用于六套配置。[外观与位置关联](src/valorant_events/stitching.py)不要求阵营一致。多帧确认减少闪烁，但增加事件输出延迟。 |
| **Minimap/object localization（小地图/物体定位）** | 检测可见小地图符号并记录地图相对位置；[校准工具](docs/hud_calibration.html)与[配置](configs/integrated_p006.json)支持区域大小、朝向和覆盖范围，布局变化时重置关联。未声称支持世界坐标、隐藏敌人追踪或可靠的旋转技能图标分类。 |
| **Quantitative precision/recall evaluation（定量评估）** | 最终视频事件 Precision **48.46%**、Recall **40.65%**，提供下方 TP/FP/FN 及匹配输入数据。保留全部原始预测参与评分；技能截图仅作展示，不纳入评估。 |

## 视频事件评估

评估使用 **130 条未经删改的原始预测**，与 **155 条由 AI 根据原始视频生成的事件参考标注**比较。项目作者已初步核对，整体未发现明显问题，但尚未经过独立逐条复核，仍可能存在误标、漏标或时间边界偏差，不能保证标注完全准确。这些片段已用于项目开发，因此本次属于开发评估，并非独立留出测试。

每条预测最多匹配一条参考事件，要求位于**同一原始切片，时间差不超过 ±0.5 秒**，且事件类型及对应属性一致：阵营、死亡/非死亡、物品类别、准确的 HUD 数值或 Spike/效果状态。未匹配的预测记为 FP，未匹配的参考事件记为 FN；重复预测不能重复得分。

| 事件 Precision | 事件 Recall | TP | FP | FN |
|---:|---:|---:|---:|---:|
| **48.46%** | **40.65%** | 63 | 67 | 92 |

Precision = 63 ÷ (63 + 67)；Recall = 63 ÷ (63 + 92)。

**评分范围：**匹配不要求是同一个人物或同一个轨迹 ID。特工、击杀者和枪械身份，检测框位置、轨迹连续性及小地图坐标均未纳入评分，因此这些事件指标不能证明检测/跟踪正确性。六组技能 GT/预测截图仅作定性展示，不计入分数。

Demo 移除两条明确的物体误检，并将缺乏支持的身份/阵营字段改为未知，保留 128 条事件；评分使用全部 130 条记录，包含这些错误。主要误差来自人物跟踪断裂产生的重复事件，以及部分片段未输出弹药变化和击杀提示。

可查看[参考标注](evaluation/final_video/truth.json)、[逐条匹配结果](evaluation/final_video/prediction_review.csv)、[漏检记录](evaluation/final_video/missed_events.csv)或[评估页面](docs/EVALUATION.html)。


## 本地运行

观看已生成的 Demo 只需要 Python 3.10+，在仓库根目录执行：

```bash
python scripts/serve_demo.py --port 8794
```

打开 <http://127.0.0.1:8794/DEMO.html>（中文）或 <http://127.0.0.1:8794/>（英文）。GitHub Pages 直接展示同一套 `docs/` 文件；网页播放已有结果，模型推理在 Python 中执行。

### 安装推理依赖

推荐 Python 3.10–3.12 虚拟环境，已验证 3.12。Windows 将激活命令替换为 `.venv\Scripts\activate`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/prepare_ocr.py
```

PyTorch/torchvision 使用对应平台版本；Colab 保留原有兼容的 CUDA 版本。`prepare_ocr.py` 首次下载标准英文 OCR 推理权重，不涉及训练。推理无需账号或 API Key。

### 重新推理

```bash
python -m valorant_events.pipeline \
  --video media/v3/P006_S001_tactical_test_600_620.mp4 \
  --weights weights/best.pt \
  --config configs/integrated_p006.json \
  --device cpu --output runs/p006
```

兼容 CUDA 显卡使用 `--device 0`，Apple GPU 可使用 `--device mps`（OCR 仍使用 CPU）。每次选择新的输出目录。`--no-ocr` 会关闭文字相关能力，仅用于精简运行；更换录像需重新[校准 HUD 与小地图](docs/hud_calibration.html)。

输出包含 `annotated.mp4`、逐帧记录、原始事件及采用四类事件格式的 `compact/events.json` / `compact/events.csv`。时间戳以输入切片为基准。新推理结果与仓库根目录经过复核合并的展示样例分开保存。OpenCV 推理视频采用 `mp4v`，在线演示视频已转成浏览器可播放的 H.264。

使用各自配置运行全部六个原始切片，切片之间重置状态：

```bash
python scripts/run_demo_inference.py --device cpu --output runs/all_clips
```

添加 `--max-frames 40` 可对每套配置先跑 40 帧。这会实际加载模型推理，不读取缓存预测，也不覆盖已发布 Demo 与固定评估。

### 评估与演示生成

```bash
python scripts/build_hr_viewer.py --lang en
python scripts/build_hr_viewer.py --lang zh
```

[评估输入与结果](evaluation/README.md)保留全部 130 条预测和 155 条 AI 参考标注。测试程序及验证日志按要求保留在仓库外。

如需根据保留的原始画面和绘制记录重建最终标注视频：

```bash
python -m pip install -r requirements-presentation.txt
python scripts/render_merged_demo.py --lang en
python scripts/render_merged_demo.py --lang zh
```

渲染需要中文字体：macOS STHeiti、Linux `fonts-noto-cjk`，或设置 `VALORANT_FONT=/path/to/font.ttc`。网页生成和推理不需要中文字体。最终视频属于离线回放，事件文字及最终关联身份绘制在事件发生帧；原始确认时间仍保留于 `evidence.json`，不能用此演示衡量实时延迟。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/valorant_events/` | 最终推理、跟踪、HUD 和事件逻辑 |
| `configs/`、`weights/`、`assets/`、`effects/` | 六套视频配置、最终权重及参考素材 |
| `docs/` | 双语静态 Demo、视频、技能截图及评估页面 |
| `media/`、`examples/render_inputs/`、`presentation/` | 原始切片、绘制记录及时间轴信息 |
| `evaluation/` | AI 参考标注、全部预测、匹配结果 |
| `scripts/` | 最终推理、渲染、网页生成及本地服务命令 |

素材使用范围见[第三方说明](THIRD_PARTY_NOTICES.md)。公开仓库不改变上游素材的非商业使用限制。
