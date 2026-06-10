from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "competition_artifacts" / "06_html_demo"
DATA_JS = DEMO_DIR / "assets" / "demo-data.js"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def read_text(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"missing file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def parse_demo_payload(text: str, errors: list[str]) -> dict[str, Any]:
    match = re.search(r"window\.XA202608_DEMO_PAYLOAD\s*=\s*(\{.*\});\s*$", text, re.S)
    if not match:
        errors.append("demo-data.js does not assign window.XA202608_DEMO_PAYLOAD")
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        errors.append(f"demo-data.js contains invalid JSON: {exc}")
        return {}


def main() -> None:
    errors: list[str] = []
    index = read_text(DEMO_DIR / "index.html", errors)
    css = read_text(DEMO_DIR / "assets" / "demo.css", errors)
    js = read_text(DEMO_DIR / "assets" / "demo.js", errors)
    data_js = read_text(DATA_JS, errors)
    readme = read_text(DEMO_DIR / "README.md", errors)
    payload = parse_demo_payload(data_js, errors) if data_js else {}

    combined = "\n".join([index, css, js, data_js, readme])
    require("assets/demo-data.js" in index, "index.html must load assets/demo-data.js", errors)
    require("assets/demo.js" in index, "index.html must load assets/demo.js", errors)
    require("assets/demo.css" in index, "index.html must load assets/demo.css", errors)
    require("http://" not in combined and "https://" not in combined, "html demo must not depend on external URLs", errors)
    require("现场训练" not in combined or "不现场训练" in combined, "demo wording must not imply live training", errors)
    for forbidden in ("完全无监督 TC", "真实在轨数据", "AI 图证明实验结果"):
        require(forbidden not in combined, f"forbidden overclaim present: {forbidden}", errors)

    tasks = payload.get("tasks", {})
    require(set(tasks) == {"first_transfer", "second_transfer"}, "payload must contain two transfer tasks", errors)
    for task_name, task in tasks.items():
        metrics = task.get("metrics", {})
        for key in ("rmse", "mae", "nasa_score", "ra", "last_window_rmse"):
            require(key in metrics, f"{task_name}: missing metric {key}", errors)
        points = task.get("representative_curve", {}).get("points", [])
        require(len(points) > 0, f"{task_name}: representative curve is empty", errors)
    require(len(payload.get("tc_ablation", [])) == 8, "payload must contain 8 TC ablation rows", errors)
    boundary = payload.get("reporting_boundary", {})
    require("validation-only" in str(boundary.get("tc", "")), "TC boundary must state validation-only", errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("html demo check ok")


if __name__ == "__main__":
    main()
