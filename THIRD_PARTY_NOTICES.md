# Third-party assets

This repository is an independent technical demonstration and is not endorsed by Riot Games. Game artwork remains the property of Riot Games. The repository does not grant rights beyond each upstream license.

| Asset | Source | Publisher license / use |
|---|---|---|
| Four-class detector | [Daniels Magonis, valorant-9ufcp v3](https://universe.roboflow.com/daniels-magonis-0pjzx/valorant-9ufcp/dataset/3), [HF mirror](https://huggingface.co/datasets/keremberke/valorant-object-detection) | CC BY 4.0; labels normalized and YOLO fine-tuned |
| Agent classifier | [VALORANT object detection v21](https://universe.roboflow.com/valorant-object-detection/valorant-object-detection-r9qkl/dataset/21) | CC BY 4.0; filtered body crops, 21 identities |
| World utility detector and gallery | [valorant_detector v2](https://universe.roboflow.com/valorantdetector-wvavl/valorant_detector/dataset/2) | CC BY 4.0; reviewed world-object labels and fine-tuning; GT/prediction overlays on selected stills |
| Smoke / flash references | [CSO Vision Valorant v8](https://universe.roboflow.com/cso-esports/cso-vision-valorant/dataset/8) | CC BY 4.0; fine-tuned smoke model and effect templates |
| Minimap detector | [valo-minimap v10](https://universe.roboflow.com/valo-wix1q/valo-minimap/dataset/10) | CC BY 4.0; coarse marker model; skill-specific map model excluded |
| Ground-weapon detector | [BEACON](https://huggingface.co/datasets/beacon-gui/BEACON-Dataset) and the above Roboflow world dataset | Mixed CC BY-NC 4.0 / CC BY 4.0; manually reviewed dropped-weapon boxes; noncommercial research/demo use |
| Gameplay excerpts | [BEACON](https://huggingface.co/datasets/beacon-gui/BEACON-Dataset), [D2E Original](https://huggingface.co/datasets/open-world-agents/D2E-Original), [D2E 480p](https://huggingface.co/datasets/open-world-agents/D2E-480p) | CC BY-NC 4.0; trim, concatenate, letterbox, convert FPS, remove audio, add detection overlays |
| Agent portraits and weapon icons | [VALORANT-API](https://valorant-api.com/) | Unofficial static game artwork; Riot rights retained. Original URLs and SHA256 are in each assets manifest. |
| Model/runtime libraries | [Ultralytics](https://github.com/ultralytics/ultralytics), [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Upstream AGPL-3.0 / Apache-2.0 respectively; third-party license obligations remain applicable |

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Original and mirror datasets are not independent samples; adjacent frames may cross publisher splits. Weights are supplied for this noncommercial technical demonstration, not as a blanket commercial license. Training notebooks and raw datasets are intentionally omitted.

The D2E recording is `recording_20250815_203048_VALORANT_3c71ee23.mkv`: baseline source interval [516, 540), native utility interval [748, 768). BEACON excerpts retain their session IDs in input filenames and source metadata. See [demo attribution](docs/SOURCES.txt) for final video/gallery transformations.
