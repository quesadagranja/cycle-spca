"""Aggregate independent runs and compute between-seed stability."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from calendar_gfspca import match_components

from .config import ExperimentConfig
from .manifests import resolve_manifest
from .utils import atomic_write_csv, read_csv


RUN_FIELDS = [
    "run_id",
    "phase",
    "task_id",
    "global_task_id",
    "n_samples",
    "n_components",
    "lambda_l1_index",
    "lambda_l1",
    "lambda_tv_index",
    "lambda_tv",
    "seed",
    "status",
    "success",
    "converged",
    "audit_all_checks_passed",
    "n_components_effective",
    "n_outer_iterations",
    "outer_limit_reached",
    "inner_limit_hits",
    "score_sweep_limit_hits",
    "last_score_sweeps_used",
    "last_score_relative_change",
    "last_score_converged",
    "last_inner_iterations",
    "last_inner_relative_change",
    "last_inner_primal_dual_residual",
    "last_inner_converged",
    "final_objective",
    "explained_variance",
    "reconstruction_error",
    "total_sum_squares",
    "mean_loading_sparsity_active",
    "global_total_variation",
    "global_loading_l1_norm",
    "global_relative_total_variation",
    "condition_number_v_gram",
    "total_reinitializations",
    "components_reinitialized",
    "elapsed_seconds",
    "maximum_rss_mb",
    "started_at_utc",
    "completed_at_utc",
    "git_commit",
    "hostname",
    "slurm_job_id",
    "error_type",
    "error_message",
]


PAIR_FIELDS = [
    "n_samples",
    "lambda_l1_index",
    "lambda_l1",
    "lambda_tv_index",
    "lambda_tv",
    "seed_a",
    "seed_b",
    "run_id_a",
    "run_id_b",
    "active_a",
    "active_b",
    "matched_components",
    "mean_similarity",
    "min_similarity",
    "component_similarities",
]


CELL_FIELDS = [
    "n_samples",
    "lambda_l1_index",
    "lambda_l1",
    "lambda_tv_index",
    "lambda_tv",
    "expected_runs",
    "completed_runs",
    "failed_runs",
    "pending_runs",
    "converged_runs",
    "audit_passed_runs",
    "full_rank_runs",
    "valid_runs",
    "cell_valid_so_far",
    "cell_complete",
    "cell_complete_and_valid",
    "explained_variance_mean",
    "explained_variance_std",
    "mean_loading_sparsity_active_mean",
    "mean_loading_sparsity_active_std",
    "global_relative_total_variation_mean",
    "global_relative_total_variation_std",
    "final_objective_mean",
    "final_objective_std",
    "condition_number_v_gram_mean",
    "condition_number_v_gram_std",
    "elapsed_seconds_mean",
    "elapsed_seconds_std",
    "maximum_rss_mb_mean",
    "maximum_rss_mb_std",
    "stability_pairs",
    "stability_mean",
    "stability_std",
    "stability_min",
]


def _read_summary(config: ExperimentConfig, run_id: str) -> dict[str, Any] | None:
    path = config.output_path / "runs" / run_id / "summary.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _flatten_run(manifest: dict[str, str], summary: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = dict(manifest)
    if summary is None:
        row["status"] = "pending"
        return row
    for field in RUN_FIELDS:
        if field in summary:
            row[field] = summary[field]
    environment = summary.get("environment", {})
    row.update(
        {
            "git_commit": environment.get("git_commit"),
            "hostname": environment.get("hostname"),
            "slurm_job_id": environment.get("slurm_job_id"),
        }
    )
    return row


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    numeric = np.array(
        [number for value in values if (number := _number(value)) is not None],
        dtype=np.float64,
    )
    if numeric.size == 0:
        return None, None
    mean = float(np.mean(numeric))
    std = float(np.std(numeric, ddof=1)) if numeric.size > 1 else 0.0
    return mean, std


def _cell_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["n_samples"]),
        int(row["lambda_l1_index"]),
        int(row["lambda_tv_index"]),
    )


def _load_active_components(config: ExperimentConfig, run_id: str) -> np.ndarray:
    path = config.output_path / "runs" / run_id / "components.npz"
    with np.load(path, allow_pickle=False) as archive:
        components = np.asarray(archive["components"], dtype=np.float64)
        active = np.asarray(archive["active"], dtype=bool)
    return components[:, active]


def _stability_pairs(
    config: ExperimentConfig,
    grouped_runs: dict[tuple[int, int, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, runs in sorted(grouped_runs.items()):
        usable = [
            run
            for run in runs
            if run.get("status") == "completed"
            and (config.output_path / "runs" / run["run_id"] / "components.npz").is_file()
        ]
        for first, second in combinations(usable, 2):
            a = _load_active_components(config, first["run_id"])
            b = _load_active_components(config, second["run_id"])
            if a.shape[1] == 0 or b.shape[1] == 0:
                similarities = np.empty(0, dtype=np.float64)
                mean_similarity = None
                min_similarity = None
                matched = 0
            else:
                match = match_components(a, b)
                similarities = match.absolute_cosines
                mean_similarity = match.mean_similarity
                min_similarity = float(np.min(similarities))
                matched = int(similarities.size)
            output.append(
                {
                    "n_samples": key[0],
                    "lambda_l1_index": key[1],
                    "lambda_l1": first["lambda_l1"],
                    "lambda_tv_index": key[2],
                    "lambda_tv": first["lambda_tv"],
                    "seed_a": first["seed"],
                    "seed_b": second["seed"],
                    "run_id_a": first["run_id"],
                    "run_id_b": second["run_id"],
                    "active_a": a.shape[1],
                    "active_b": b.shape[1],
                    "matched_components": matched,
                    "mean_similarity": mean_similarity,
                    "min_similarity": min_similarity,
                    "component_similarities": similarities.tolist(),
                }
            )
    return output


def _cell_rows(
    config: ExperimentConfig,
    grouped_runs: dict[tuple[int, int, int], list[dict[str, Any]]],
    stability_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_pairs: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in stability_pairs:
        grouped_pairs[_cell_key(pair)].append(pair)

    output: list[dict[str, Any]] = []
    for key, runs in sorted(grouped_runs.items()):
        expected = len(runs)
        completed = [run for run in runs if run.get("status") == "completed"]
        failed = [run for run in runs if run.get("status") == "failed"]
        converged = [run for run in completed if bool(run.get("converged"))]
        audit_passed = [
            run for run in completed if bool(run.get("audit_all_checks_passed"))
        ]
        full_rank = [
            run
            for run in completed
            if int(run.get("n_components_effective") or -1)
            == config.model.n_components
        ]
        valid = [
            run
            for run in completed
            if bool(run.get("converged"))
            and bool(run.get("audit_all_checks_passed"))
            and int(run.get("n_components_effective") or -1)
            == config.model.n_components
        ]
        row: dict[str, Any] = {
            "n_samples": key[0],
            "lambda_l1_index": key[1],
            "lambda_l1": runs[0]["lambda_l1"],
            "lambda_tv_index": key[2],
            "lambda_tv": runs[0]["lambda_tv"],
            "expected_runs": expected,
            "completed_runs": len(completed),
            "failed_runs": len(failed),
            "pending_runs": expected - len(completed) - len(failed),
            "converged_runs": len(converged),
            "audit_passed_runs": len(audit_passed),
            "full_rank_runs": len(full_rank),
            "valid_runs": len(valid),
            "cell_valid_so_far": bool(completed and len(valid) == len(completed)),
            "cell_complete": len(completed) == expected,
            "cell_complete_and_valid": len(valid) == expected,
        }
        metrics = (
            "explained_variance",
            "mean_loading_sparsity_active",
            "global_relative_total_variation",
            "final_objective",
            "condition_number_v_gram",
            "elapsed_seconds",
            "maximum_rss_mb",
        )
        for metric in metrics:
            mean, std = _mean_std(run.get(metric) for run in completed)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        pairs = grouped_pairs.get(key, [])
        pair_values = [pair["mean_similarity"] for pair in pairs]
        stability_mean, stability_std = _mean_std(pair_values)
        numeric_pairs = [
            value for value in (_number(item) for item in pair_values) if value is not None
        ]
        row.update(
            {
                "stability_pairs": len(numeric_pairs),
                "stability_mean": stability_mean,
                "stability_std": stability_std,
                "stability_min": min(numeric_pairs) if numeric_pairs else None,
            }
        )
        output.append(row)
    return output


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        print("Parquet skipped: install the experiment optional dependencies.")
        return False
    frame = pd.DataFrame(rows)
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ModuleNotFoundError) as error:
        print(f"Parquet skipped: {error}")
        return False
    return True


def aggregate_results(config: ExperimentConfig) -> dict[str, Path]:
    manifest_path = resolve_manifest(config, "all")
    manifest_rows = read_csv(manifest_path)
    runs = [
        _flatten_run(row, _read_summary(config, row["run_id"]))
        for row in manifest_rows
    ]
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[_cell_key(run)].append(run)
    pairs = _stability_pairs(config, grouped)
    cells = _cell_rows(config, grouped, pairs)

    target = config.output_path / "aggregated"
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs_csv": target / "runs.csv",
        "runs_parquet": target / "runs.parquet",
        "pairs_csv": target / "stability_pairs.csv",
        "pairs_parquet": target / "stability_pairs.parquet",
        "cells_csv": target / "cells.csv",
        "cells_parquet": target / "cells.parquet",
    }
    atomic_write_csv(paths["runs_csv"], runs, RUN_FIELDS)
    atomic_write_csv(paths["pairs_csv"], pairs, PAIR_FIELDS)
    atomic_write_csv(paths["cells_csv"], cells, CELL_FIELDS)
    _write_parquet(paths["runs_parquet"], runs)
    _write_parquet(paths["pairs_parquet"], pairs)
    _write_parquet(paths["cells_parquet"], cells)
    completed = sum(run.get("status") == "completed" for run in runs)
    failed = sum(run.get("status") == "failed" for run in runs)
    print(
        f"Aggregated planned={len(runs)} completed={completed} failed={failed} "
        f"cells={len(cells)} stability_pairs={len(pairs)}"
    )
    return paths
