from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "competition_artifacts"
SOURCE_DIR = PROJECT_ROOT / "html_demo"
OUTPUT_DIR = ARTIFACTS / "06_html_demo"
PAYLOAD_PATH = ARTIFACTS / "05_report_assets" / "demo_payload.json"


def load_payload() -> dict[str, Any]:
    if not PAYLOAD_PATH.exists():
        raise FileNotFoundError(f"missing demo payload: {PAYLOAD_PATH}")
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tasks = payload.get("tasks", {})
    if set(tasks) != {"first_transfer", "second_transfer"}:
        raise ValueError("demo payload must contain first_transfer and second_transfer")
    if len(payload.get("tc_ablation", [])) != 8:
        raise ValueError("demo payload must contain 8 TC ablation rows")
    return payload


def copy_template() -> None:
    if OUTPUT_DIR.exists():
        resolved = OUTPUT_DIR.resolve()
        if ARTIFACTS.resolve() not in resolved.parents:
            raise RuntimeError(f"refusing to remove outside artifacts: {resolved}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))


def write_demo_data(payload: dict[str, Any]) -> None:
    data_js = (
        "window.XA202608_DEMO_PAYLOAD = "
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + ";\n"
    )
    (OUTPUT_DIR / "assets" / "demo-data.js").write_text(data_js, encoding="utf-8")


def write_readme() -> None:
    lines = [
        "# XA-202608 动态 HTML Demo",
        "",
        "本目录是可直接打开的离线动态 demo。建议双击 `index.html`，或用浏览器打开该文件。",
        "",
        "## 数据来源",
        "",
        "- `assets/demo-data.js` 由 `competition_artifacts/05_report_assets/demo_payload.json` 自动生成。",
        "- Demo 只读已有实验结果，不现场训练模型，不重新拟合 TC，不使用目标测试标签做校准。",
        "- `PG-STDA-SAC-RSPA` 是 strict raw UDA 迁移模型；`PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final pipeline。",
        "",
        "## 重新生成",
        "",
        "```powershell",
        "& 'D:\\anaconda\\envs\\jiebang\\python.exe' scripts/build_html_demo.py",
        "```",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = load_payload()
    copy_template()
    write_demo_data(payload)
    write_readme()
    print(f"html demo written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
