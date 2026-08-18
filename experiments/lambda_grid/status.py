"""Progress summaries for independently executed jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .manifests import resolve_manifest
from .utils import read_csv


def collect_status(
    config: ExperimentConfig,
    manifest: str | Path = "all",
) -> dict[str, Any]:
    manifest_path = resolve_manifest(config, manifest)
    rows = read_csv(manifest_path)
    counts = {
        "planned": len(rows),
        "completed": 0,
        "converged": 0,
        "not_converged": 0,
        "failed": 0,
        "pending": 0,
        "audit_passed": 0,
        "full_rank": 0,
    }
    failures: list[dict[str, str]] = []
    for row in rows:
        summary_path = (
            config.output_path / "runs" / row["run_id"] / "summary.json"
        )
        if not summary_path.is_file():
            counts["pending"] += 1
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            counts["pending"] += 1
            continue
        if summary.get("status") == "failed":
            counts["failed"] += 1
            failures.append(
                {
                    "run_id": row["run_id"],
                    "error_type": str(summary.get("error_type", "")),
                    "error_message": str(summary.get("error_message", "")),
                }
            )
            continue
        if summary.get("status") != "completed":
            counts["pending"] += 1
            continue
        counts["completed"] += 1
        if summary.get("converged"):
            counts["converged"] += 1
        else:
            counts["not_converged"] += 1
        if summary.get("audit_all_checks_passed"):
            counts["audit_passed"] += 1
        if summary.get("n_components_effective") == config.model.n_components:
            counts["full_rank"] += 1
    counts["accounted_for"] = counts["completed"] + counts["failed"]
    return {
        "manifest": str(manifest_path),
        "counts": counts,
        "failures": failures,
    }


def print_status(status: dict[str, Any]) -> None:
    counts = status["counts"]
    print(f"Manifest:      {status['manifest']}")
    labels = (
        ("Planned", "planned"),
        ("Completed", "completed"),
        ("Converged", "converged"),
        ("Not converged", "not_converged"),
        ("Audit passed", "audit_passed"),
        ("Full rank", "full_rank"),
        ("Failed", "failed"),
        ("Pending", "pending"),
    )
    for label, key in labels:
        print(f"{label + ':':15s}{counts[key]:6d}")
    if status["failures"]:
        print("Recent failures:")
        for failure in status["failures"][-10:]:
            print(
                f"  {failure['run_id']}: {failure['error_type']}: "
                f"{failure['error_message']}"
            )
