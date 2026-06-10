from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "competition_artifacts"
REPORT_DIR = ARTIFACTS / "05_report_assets"


CHECKS: list[dict[str, Any]] = [
    {
        "name": "demo_input_manifest",
        "path": "competition_artifacts/05_report_assets/demo_input_manifest.json",
        "kind": "file",
        "min_bytes": 1000,
    },
    {
        "name": "demo_payload",
        "path": "competition_artifacts/05_report_assets/demo_payload.json",
        "kind": "file",
        "min_bytes": 1000,
    },
    {
        "name": "strict_raw_comparison",
        "path": "competition_artifacts/03_results/strict_unsupervised_comparison.md",
        "kind": "contains",
        "required_text": [
            "PG-STDA-SAC-RSPA",
            "raw",
            "不使用目标测试集标签",
        ],
    },
    {
        "name": "tc_ablation_boundary",
        "path": "competition_artifacts/03_results/tc_ablation/tc_ablation_summary.md",
        "kind": "contains",
        "required_text": [
            "time-only TC",
            "validation-only",
            "不使用测试标签",
        ],
    },
    {
        "name": "final_recommendation_boundary",
        "path": "competition_artifacts/03_results/final_recommendation.md",
        "kind": "contains",
        "required_text": [
            "PG-STDA-SAC-RSPA-TC",
            "验证集",
            "不使用目标测试集标签",
        ],
    },
    {
        "name": "pre_demo_boundary_audit",
        "path": "competition_artifacts/05_report_assets/source_docs/PRE_DEMO_BOUNDARY_AUDIT.md",
        "kind": "contains",
        "required_text": [
            "validation-only",
            "Conformal Prediction",
            "Python 的机理约束",
        ],
    },
    {
        "name": "pre_demo_hardening",
        "path": "competition_artifacts/05_report_assets/source_docs/PRE_DEMO_SUBMISSION_HARDENING.md",
        "kind": "contains",
        "required_text": [
            "demo_payload.json",
            "check_demo_inputs.py",
            "Docker 和动态 Demo 前剩余项",
        ],
    },
    {
        "name": "html_demo_design_spec",
        "path": "competition_artifacts/05_report_assets/source_docs/HTML_DEMO_DESIGN_SPEC.md",
        "kind": "contains",
        "required_text": [
            "动态 HTML demo",
            "demo_payload.json",
            "不现场训练",
            "validation-only",
        ],
    },
    {
        "name": "html_demo_index",
        "path": "competition_artifacts/06_html_demo/index.html",
        "kind": "file",
        "min_bytes": 1000,
    },
    {
        "name": "html_demo_data",
        "path": "competition_artifacts/06_html_demo/assets/demo-data.js",
        "kind": "file",
        "min_bytes": 1000,
    },
    {
        "name": "docker_reproducibility",
        "path": "competition_artifacts/05_report_assets/source_docs/DOCKER_REPRODUCIBILITY.md",
        "kind": "contains",
        "required_text": [
            "docker build",
            "scripts/run_docker_smoke.sh",
            "Docker daemon",
        ],
    },
]


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def check_file(spec: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / str(spec["path"])
    result: dict[str, Any] = {"name": spec["name"], "path": rel_path(path), "ok": True, "details": []}
    if not path.exists():
        result["ok"] = False
        result["details"].append("missing")
        return result
    min_bytes = int(spec.get("min_bytes", 1))
    size = path.stat().st_size
    result["details"].append(f"size_bytes={size}")
    if size < min_bytes:
        result["ok"] = False
        result["details"].append(f"below_min_bytes={min_bytes}")
    return result


def check_contains(spec: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / str(spec["path"])
    result: dict[str, Any] = {"name": spec["name"], "path": rel_path(path), "ok": True, "details": []}
    if not path.exists():
        result["ok"] = False
        result["details"].append("missing")
        return result
    text = path.read_text(encoding="utf-8")
    for needle in spec.get("required_text", []):
        if str(needle) not in text:
            result["ok"] = False
            result["details"].append(f"missing_text={needle}")
    if result["ok"]:
        result["details"].append("required_text_present")
    return result


def run_demo_input_check() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "scripts/check_demo_inputs.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "name": "check_demo_inputs",
        "path": "scripts/check_demo_inputs.py",
        "ok": result.returncode == 0,
        "details": [line for line in (result.stdout + result.stderr).splitlines() if line],
    }


def run_html_demo_check() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "scripts/check_html_demo.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "name": "check_html_demo",
        "path": "scripts/check_html_demo.py",
        "ok": result.returncode == 0,
        "details": [line for line in (result.stdout + result.stderr).splitlines() if line],
    }


def build_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# 动态 Demo 前就绪检查",
        "",
        "本报告由 `scripts/check_pre_demo_readiness.py` 生成，用于确认 Docker 收尾和动态 HTML demo 开发前，当前证据包、payload、manifest 和报告口径边界处于可用状态。",
        "",
        "| 检查项 | 状态 | 文件/入口 | 说明 |",
        "|---|---|---|---|",
    ]
    for result in results:
        status = "通过" if result["ok"] else "失败"
        details = "; ".join(result.get("details", []))
        lines.append(f"| {result['name']} | {status} | `{result['path']}` | {details} |")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本检查不运行 Docker，也不启动动态 HTML demo。",
            "- 本检查不训练模型，不重新拟合 TC，不读取目标测试标签做校准。",
            "- 后续 demo 应优先读取 `competition_artifacts/05_report_assets/demo_payload.json`；其他文件只作为追溯证据。",
            "- `PG-STDA-SAC-RSPA` 是 strict raw UDA 迁移模型；`PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final pipeline。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = [run_demo_input_check(), run_html_demo_check()]
    for spec in CHECKS:
        if spec["kind"] == "file":
            results.append(check_file(spec))
        elif spec["kind"] == "contains":
            results.append(check_contains(spec))
        else:
            raise ValueError(f"unknown check kind: {spec['kind']}")
    ok = all(result["ok"] for result in results)
    payload = {"ok": ok, "results": results}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "pre_demo_readiness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "pre_demo_readiness.md").write_text(build_report(results), encoding="utf-8")
    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['name']}: {'; '.join(result.get('details', []))}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
