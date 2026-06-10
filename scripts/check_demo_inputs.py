from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "competition_artifacts" / "05_report_assets" / "demo_input_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), sum(1 for _ in reader)


def resolve_artifact_path(rel_path: str) -> Path:
    return PROJECT_ROOT / rel_path


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_prediction_csv(name: str, spec: dict[str, Any], errors: list[str]) -> None:
    path = resolve_artifact_path(str(spec["path"]))
    require(path.exists(), f"{name}: missing prediction CSV {path}", errors)
    if not path.exists():
        return
    fields, row_count = read_csv_rows(path)
    required = list(spec.get("required_columns", []))
    missing = [col for col in required if col not in fields]
    require(not missing, f"{name}: missing columns {missing} in {path}", errors)
    require(row_count >= int(spec.get("min_rows", 1)), f"{name}: row count {row_count} below minimum", errors)


def check_metrics_json(name: str, spec: dict[str, Any], errors: list[str]) -> None:
    path = resolve_artifact_path(str(spec["path"]))
    require(path.exists(), f"{name}: missing metrics JSON {path}", errors)
    if not path.exists():
        return
    data = load_json(path)
    metrics = data.get("final_metrics") or data.get("test_metrics") or data
    for key in spec.get("required_metrics", []):
        require(key in metrics, f"{name}: missing metric {key} in {path}", errors)
    if "rmse" in metrics:
        rmse = float(metrics["rmse"])
        require(0.0 <= rmse <= 1.0, f"{name}: rmse out of normalized range: {rmse}", errors)


def check_table_csv(name: str, spec: dict[str, Any], errors: list[str]) -> None:
    path = resolve_artifact_path(str(spec["path"]))
    require(path.exists(), f"{name}: missing CSV table {path}", errors)
    if not path.exists():
        return
    fields, row_count = read_csv_rows(path)
    required = list(spec.get("required_columns", []))
    missing = [col for col in required if col not in fields]
    require(not missing, f"{name}: missing columns {missing} in {path}", errors)
    require(row_count >= int(spec.get("min_rows", 1)), f"{name}: row count {row_count} below minimum", errors)
    if name == "tc_ablation_summary":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        variants = {(row["task"], row["variant"]) for row in rows}
        for task in ("第一迁移", "第二迁移"):
            for variant in ("raw", "y_pred-only TC", "time-only TC", "y_pred+time TC"):
                require((task, variant) in variants, f"{name}: missing {task} / {variant}", errors)
        require(
            all(row.get("uses_test_labels_for_fit") == "no" for row in rows),
            f"{name}: TC fit must not use test labels",
            errors,
        )


def check_required_file(name: str, spec: dict[str, Any], errors: list[str]) -> None:
    path = resolve_artifact_path(str(spec["path"]))
    require(path.exists(), f"{name}: missing required file {path}", errors)
    if path.exists():
        min_bytes = int(spec.get("min_bytes", 1))
        require(path.stat().st_size >= min_bytes, f"{name}: file too small {path}", errors)


def main() -> None:
    manifest = load_json(DEFAULT_MANIFEST)
    errors: list[str] = []
    for name, spec in manifest.get("prediction_csvs", {}).items():
        check_prediction_csv(name, spec, errors)
    for name, spec in manifest.get("metrics_jsons", {}).items():
        check_metrics_json(name, spec, errors)
    for name, spec in manifest.get("table_csvs", {}).items():
        check_table_csv(name, spec, errors)
    for name, spec in manifest.get("required_files", {}).items():
        check_required_file(name, spec, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    total_files = sum(len(manifest.get(section, {})) for section in ("prediction_csvs", "metrics_jsons", "table_csvs", "required_files"))
    print(f"demo input check ok: {total_files} manifest entries")


if __name__ == "__main__":
    main()
