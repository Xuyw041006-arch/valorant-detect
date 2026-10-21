"""Run fresh model inference on all six original demo clips, with cut resets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="cpu, mps, or CUDA index such as 0")
    p.add_argument("--output", type=Path, default=ROOT / "runs/demo")
    p.add_argument("--max-frames", type=int, help="Optional per-clip smoke-test limit")
    p.add_argument("--no-ocr", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        p.error("Choose a new output directory; existing runs are preserved.")
    manifest = json.loads((ROOT / "presentation/edit_manifest.json").read_text())
    summaries = []
    for segment in manifest["segments"]:
        name = segment["id"]
        command = [
            sys.executable,
            "-m",
            "valorant_events.pipeline",
            "--video",
            str(ROOT / segment["source_video"]),
            "--weights",
            str(ROOT / "weights/best.pt"),
            "--config",
            str(ROOT / f"configs/integrated_{name}.json"),
            "--duration",
            str(segment["duration"]),
            "--device",
            a.device,
            "--output",
            str(a.output / name),
        ]
        if a.max_frames:
            command += ["--max-frames", str(a.max_frames)]
        if a.no_ocr:
            command += ["--no-ocr"]
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads((a.output / name / "run_summary.json").read_text())
        summaries.append(
            {
                "clip": name,
                "status": result["status"],
                "frames": result["processed_frames"],
                "compact_events": result.get("compact_events", {}),
            }
        )
    (a.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
