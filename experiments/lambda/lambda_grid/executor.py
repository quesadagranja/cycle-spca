"""Execute one manifest row and persist all fit artifacts."""

from __future__ import annotations

from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import scipy

from calendar_gfspca import CalendarGraphFusedSparsePCA

from .config import ExperimentConfig
from .manifests import get_task
from .sampling import sample_paths
from .utils import atomic_write_json, tee_stdout, to_jsonable


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version() -> str:
    try:
        return importlib.metadata.version("calendar-gfspca")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "calendar_gfspca": _package_version(),
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _maximum_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _history_rows(model: CalendarGraphFusedSparsePCA) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer in model.history_as_dicts():
        inner = outer.pop("inner")
        row = dict(outer)
        row["reinitialized"] = json.dumps(row["reinitialized"])
        row.update({f"inner_{key}": value for key, value in inner.items()})
        rows.append(row)
    return rows


def _atomic_history(path: Path, model: CalendarGraphFusedSparsePCA) -> None:
    rows = _history_rows(model)
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if to_jsonable(value) is None else to_jsonable(value)
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _load_master(config: ExperimentConfig, n_samples: int):
    paths = sample_paths(config)
    for name in ("metadata", "master", "indices", "imputed"):
        if not paths[name].is_file():
            raise FileNotFoundError(
                f"Missing prepared sample artifact: {paths[name]}. Run prepare first."
            )
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("config_sha256") != config.sha256:
        raise RuntimeError("Prepared sample and active configuration do not match.")
    master = np.load(paths["master"], mmap_mode="r", allow_pickle=False)
    expected_shape = (config.sampling.master_size, config.input.n_features)
    if master.shape != expected_shape:
        raise ValueError(
            f"Prepared master sample has shape {master.shape}; expected {expected_shape}."
        )
    if n_samples not in config.sampling.sizes:
        raise ValueError(f"n_samples={n_samples} is not a configured sample size.")
    return master[:n_samples], metadata


def _run_metrics(
    model: CalendarGraphFusedSparsePCA,
    audit: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    active = model.active_
    total_l1 = float(np.sum(model.loading_l1_norm_[active]))
    total_tv = float(np.sum(model.total_variation_[active]))
    global_rtv = total_tv / total_l1 if total_l1 > 0 else float("nan")
    final = model.history_[-1]
    history = model.history_
    return {
        "converged": bool(model.converged_),
        "n_outer_iterations": int(model.n_iter_),
        "outer_limit_reached": bool(
            not model.converged_ and model.n_iter_ >= model.outer_max_iter
        ),
        "inner_limit_hits": int(
            sum(item.inner.iterations >= model.inner_max_iter for item in history)
        ),
        "score_sweep_limit_hits": int(
            sum(
                item.score_sweeps_used >= model.score_sweeps
                and not item.score_converged
                for item in history
            )
        ),
        "last_score_sweeps_used": int(final.score_sweeps_used),
        "last_score_relative_change": float(final.score_relative_change),
        "last_score_converged": bool(final.score_converged),
        "last_inner_iterations": int(final.inner.iterations),
        "last_inner_relative_change": float(final.inner.relative_change),
        "last_inner_primal_dual_residual": float(
            final.inner.primal_dual_residual
        ),
        "last_inner_converged": bool(final.inner.converged),
        "n_components_effective": int(model.n_components_effective_),
        "final_objective": float(final.objective),
        "explained_variance": float(model.explained_variance_),
        "reconstruction_error": float(model.reconstruction_error_),
        "total_sum_squares": float(model.total_sum_squares_),
        "mean_loading_sparsity_active": float(
            model.mean_loading_sparsity_active_
        ),
        "global_total_variation": total_tv,
        "global_loading_l1_norm": total_l1,
        "global_relative_total_variation": global_rtv,
        "condition_number_v_gram": float(model.condition_number_),
        "total_reinitializations": int(
            np.sum(model.reinitialization_counts_)
        ),
        "components_reinitialized": int(
            np.count_nonzero(model.reinitialization_counts_)
        ),
        "audit_all_checks_passed": bool(audit["all_checks_passed"]),
        "elapsed_seconds": float(elapsed_seconds),
        "maximum_rss_mb": _maximum_rss_mb(),
    }


def execute_task(
    config: ExperimentConfig,
    task: dict[str, str],
    *,
    mirror_to_terminal: bool = True,
) -> dict[str, Any]:
    run_id = task["run_id"]
    run_dir = config.output_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    success_marker = run_dir / "_SUCCESS"
    failed_marker = run_dir / "_FAILED.json"
    summary_path = run_dir / "summary.json"
    if success_marker.exists() and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        print(f"SKIP run={run_id} status=completed")
        return summary

    log_context = tee_stdout(run_dir / "run.log") if mirror_to_terminal else nullcontext()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    base_summary: dict[str, Any] = {
        "run_id": run_id,
        "phase": task["phase"],
        "task_id": int(task["task_id"]),
        "global_task_id": int(task["global_task_id"]),
        "n_samples": int(task["n_samples"]),
        "n_components": int(task["n_components"]),
        "lambda_l1_index": int(task["lambda_l1_index"]),
        "lambda_l1": float(task["lambda_l1"]),
        "lambda_tv_index": int(task["lambda_tv_index"]),
        "lambda_tv": float(task["lambda_tv"]),
        "seed": int(task["seed"]),
        "config_sha256": config.sha256,
        "started_at_utc": started_at.isoformat(),
        "environment": _environment(),
        "model_parameters": config.model.estimator_kwargs(),
    }

    with log_context:
        try:
            print(f"START run={run_id}")
            print(
                f"N={task['n_samples']} K={task['n_components']} "
                f"lambda_l1={task['lambda_l1']} lambda_tv={task['lambda_tv']} "
                f"seed={task['seed']}"
            )
            x, sample_metadata = _load_master(config, int(task["n_samples"]))
            kwargs = config.model.estimator_kwargs()
            kwargs.update(
                {
                    "lambda_l1": float(task["lambda_l1"]),
                    "lambda_tv": float(task["lambda_tv"]),
                    "random_state": int(task["seed"]),
                }
            )
            model = CalendarGraphFusedSparsePCA(**kwargs).fit(x)
            print("Running independent metric audit")
            audit = model.audit_metrics()
            elapsed = time.perf_counter() - started_clock
            diagnostics = model.diagnostics()
            metrics = _run_metrics(model, audit, elapsed)
            summary = base_summary | {
                "status": "completed",
                "success": True,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "sample_source_indices_sha256": sample_metadata["sampling"][
                    "source_indices_sha256"
                ],
                "sample_master_sha256": sample_metadata["master_sample_sha256"],
            } | metrics

            _atomic_history(run_dir / "history.csv.gz", model)
            atomic_write_json(run_dir / "diagnostics.json", diagnostics)
            atomic_write_json(run_dir / "audit.json", audit)
            _atomic_npz(
                run_dir / "components.npz",
                components=model.components_,
                active=model.active_,
                reinitialization_counts=model.reinitialization_counts_,
            )
            atomic_write_json(summary_path, summary)
            failed_marker.unlink(missing_ok=True)
            success_marker.touch()
            print(
                f"DONE converged={model.converged_} "
                f"outer_iterations={model.n_iter_} "
                f"EV={model.explained_variance_:.6f} "
                f"sparsity={model.mean_loading_sparsity_active_:.6f} "
                f"RTV={metrics['global_relative_total_variation']:.6f} "
                f"elapsed={elapsed:.1f}s"
            )
            return summary
        except Exception as error:
            elapsed = time.perf_counter() - started_clock
            failure = base_summary | {
                "status": "failed",
                "success": False,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "maximum_rss_mb": _maximum_rss_mb(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(summary_path, failure)
            atomic_write_json(failed_marker, failure)
            success_marker.unlink(missing_ok=True)
            print(
                f"FAILED run={run_id} type={type(error).__name__} "
                f"message={error} elapsed={elapsed:.1f}s"
            )
            traceback.print_exc()
            raise


def run_from_manifest(
    config: ExperimentConfig,
    manifest: str | Path,
    task_id: int,
) -> dict[str, Any]:
    task = get_task(config, manifest, task_id)
    return execute_task(config, task)
