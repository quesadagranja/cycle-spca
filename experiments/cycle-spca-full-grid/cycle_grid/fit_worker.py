from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
from typing import Any
import uuid

import numpy as np

from .io_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_rows,
    read_json,
    relative_to,
    sha256_array,
    utc_now,
)


_CONFIG: dict[str, Any] | None = None
_OUTPUT: Path | None = None
_DATASET: np.ndarray | None = None
_INDICES: dict[int, np.ndarray] = {}


class IndexedRowsMatrix:
    """Slice-only matrix view over selected rows of a memory-mapped NPY.

    CycleSPCA requests contiguous row slices.  Each request materializes only
    that batch, while the complete source matrix remains memory-mapped and is
    shared through the operating-system page cache by all workers.
    """

    def __init__(
        self,
        source: np.ndarray,
        row_indices: np.ndarray,
        feature_start: int,
        feature_count: int,
    ) -> None:
        self.source = source
        self.row_indices = row_indices
        self.feature_start = int(feature_start)
        self.feature_stop = self.feature_start + int(feature_count)
        self.shape = (int(row_indices.size), int(feature_count))

    def __getitem__(self, key):
        if not isinstance(key, (slice, int, np.integer)):
            raise TypeError("IndexedRowsMatrix supports row slices and integers only.")
        rows = self.row_indices[key]
        return self.source[rows, self.feature_start : self.feature_stop]


def initialize_fit_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _OUTPUT, _DATASET, _INDICES
    _CONFIG = config
    _OUTPUT = Path(config["output_dir"])
    repo = str(Path(config["cycle_spca_repo"]).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _DATASET = np.load(config["dataset_path"], mmap_mode="r", allow_pickle=False)
    _INDICES = {}

    # Matplotlib is initialized once per long-lived worker.
    import matplotlib

    matplotlib.use("Agg", force=True)


def _require_state() -> tuple[dict[str, Any], Path, np.ndarray]:
    if _CONFIG is None or _OUTPUT is None or _DATASET is None:
        raise RuntimeError("Fit worker was not initialized.")
    return _CONFIG, _OUTPUT, _DATASET


def _sample_indices(task: dict[str, Any]) -> np.ndarray:
    repeat = int(task["repeat"])
    if repeat not in _INDICES:
        _INDICES[repeat] = np.load(
            task["sample_indices_path"], mmap_mode="r", allow_pickle=False
        )
    return np.asarray(_INDICES[repeat][: int(task["N"])], dtype=np.int64)


def _mean_active(values: np.ndarray, active: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[active]
    return float(np.mean(selected)) if selected.size else None


def _median_active(values: np.ndarray, active: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[active]
    return float(np.median(selected)) if selected.size else None


def _component_rows(model, task: dict[str, Any]) -> list[dict[str, Any]]:
    d = model.diagnostics()
    rows: list[dict[str, Any]] = []
    for k in range(int(task["K"])):
        rows.append(
            {
                "fit_id": task["fit_id"],
                "component": k + 1,
                "active": bool(d["active"][k]),
                "conditional_contribution_ss": float(d["conditional_contribution_ss"][k]),
                "conditional_contribution_ratio": float(
                    d["conditional_contribution_ratio"][k]
                ),
                "conditional_contribution_percent": float(
                    d["conditional_contribution_percent"][k]
                ),
                "sparsity_fraction": float(d["loading_sparsity"][k]),
                "sparsity_percent": 100.0 * float(d["loading_sparsity"][k]),
                "loading_l1_norm": float(d["loading_l1_norm"][k]),
                "loading_l2_norm": float(d["loading_l2_norm"][k]),
                "total_variation": float(d["total_variation"][k]),
                "relative_total_variation": float(d["relative_total_variation"][k]),
                "active_cells": int(d["active_cells"][k]),
                "connected_regions": int(d["n_connected_regions"][k]),
                "effective_regions": int(d["n_effective_regions"][k]),
                "largest_region_size": int(d["largest_region_size"][k]),
                "largest_region_active_fraction": float(
                    d["largest_region_active_fraction"][k]
                ),
                "largest_region_l1_fraction": float(
                    d["largest_region_l1_fraction"][k]
                ),
                "reinitializations": int(d["reinitialization_counts"][k]),
            }
        )
    return rows


COMPONENT_FIELDS = [
    "fit_id",
    "component",
    "active",
    "conditional_contribution_ss",
    "conditional_contribution_ratio",
    "conditional_contribution_percent",
    "sparsity_fraction",
    "sparsity_percent",
    "loading_l1_norm",
    "loading_l2_norm",
    "total_variation",
    "relative_total_variation",
    "active_cells",
    "connected_regions",
    "effective_regions",
    "largest_region_size",
    "largest_region_active_fraction",
    "largest_region_l1_fraction",
    "reinitializations",
]


HISTORY_FIELDS = [
    "iteration",
    "objective",
    "reconstruction_error",
    "explained_variance",
    "relative_objective_change",
    "relative_reconstruction_change",
    "lipschitz_u",
    "condition_v",
    "k_eff",
    "reinitialized",
    "inner_iterations",
    "inner_relative_change",
    "inner_primal_dual_residual",
    "inner_initial_objective",
    "inner_final_objective",
    "inner_accepted",
    "inner_converged",
    "inner_tolerance",
    "inner_tau",
    "inner_sigma",
]


def _history_rows(model) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in model.history_as_dicts():
        inner = item.pop("inner")
        item["reinitialized"] = ";".join(map(str, item["reinitialized"]))
        for key, value in inner.items():
            item[f"inner_{key}"] = value
        output.append(item)
    return output


def tensor_to_heatmap(tensor: np.ndarray) -> np.ndarray:
    """Map (hour, day, week) to (hour, chronological day)."""

    if tensor.ndim != 3:
        raise ValueError("Expected one loading tensor with three axes.")
    hours, days, weeks = tensor.shape
    return tensor.transpose(0, 2, 1).reshape(hours, weeks * days)


def _save_component_pngs(
    directory: Path,
    model,
    task: dict[str, Any],
    component_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    if not config["png"].get("enabled", True):
        return
    import matplotlib.pyplot as plt
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    tensors = model.loading_tensors()
    palette_colors = int(config["png"].get("palette_colors", 256))
    compression = int(config["png"].get("compression_level", 6))
    figsize = tuple(map(float, config["png"].get("figsize", [14.0, 6.0])))
    dpi = int(config["png"].get("dpi", 150))
    days = tensors.shape[1]
    weeks = tensors.shape[2]

    for k, metrics in enumerate(component_rows):
        heatmap = tensor_to_heatmap(tensors[..., k])
        vmax = float(np.max(np.abs(heatmap)))
        if vmax <= 1e-15:
            vmax = 1.0
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        image_artist = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        for week in range(1, weeks):
            ax.axvline(week * days - 0.5, color="black", linewidth=0.18, alpha=0.18)
        tick_weeks = list(range(0, weeks, 4))
        ax.set_xticks([week * days + (days - 1) / 2 for week in tick_weeks])
        ax.set_xticklabels([str(week + 1) for week in tick_weeks])
        ax.set_yticks(range(tensors.shape[0]))
        ax.set_xlabel("ISO week (weekdays run from Monday to Sunday within each week)")
        ax.set_ylabel("Hour of day")
        status = "active" if metrics["active"] else "inactive"
        ax.set_title(
            f"{task['fit_id']} · component {k + 1:02d} ({status})\n"
            f"EV={100.0 * model.explained_variance_:.3f}% · "
            f"contribution={metrics['conditional_contribution_percent']:.3f}% · "
            f"sparsity={metrics['sparsity_percent']:.2f}% · "
            f"relative TV={metrics['relative_total_variation']:.5g}"
        )
        colorbar = fig.colorbar(image_artist, ax=ax, shrink=0.92, pad=0.015)
        colorbar.set_label("Loading value")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as source:
            adaptive = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", 1)
            quantized = source.convert(
                "P", palette=adaptive, colors=max(2, min(palette_colors, 256))
            )
            quantized.save(
                directory / f"component_{k + 1:02d}.png",
                format="PNG",
                optimize=False,
                compress_level=max(0, min(compression, 9)),
            )


def _metrics(model, task: dict[str, Any], started: str, elapsed: float) -> dict[str, Any]:
    d = model.diagnostics()
    active = np.asarray(d["active"], dtype=bool)
    history = model.history_as_dicts()
    inner_iterations = np.array([item["inner"]["iterations"] for item in history])
    inner_converged = np.array([item["inner"]["converged"] for item in history])
    last = history[-1]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.  The cluster target is Linux.
    rss_mb = float(rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0**2))
    return {
        "fit_id": task["fit_id"],
        "repeat": int(task["repeat"]),
        "N": int(task["N"]),
        "K": int(task["K"]),
        "K_effective": int(d["effective_rank"]),
        "l1_index": int(task["l1_index"]),
        "ltv_index": int(task["ltv_index"]),
        "lambda_l1": float(task["lambda_l1"]),
        "lambda_tv": float(task["lambda_tv"]),
        "sample_seed": int(task["sample_seed"]),
        "initialization_seed": int(task["initialization_seed"]),
        "sample_master_hash": task["sample_master_hash"],
        "sample_prefix_hash": sha256_array(_sample_indices(task)),
        "explained_variance": float(d["explained_variance"]),
        "explained_variance_percent": 100.0 * float(d["explained_variance"]),
        "mean_sparsity_active": _mean_active(d["loading_sparsity"], active),
        "median_sparsity_active": _median_active(d["loading_sparsity"], active),
        "mean_relative_tv_active": _mean_active(d["relative_total_variation"], active),
        "median_relative_tv_active": _median_active(
            d["relative_total_variation"], active
        ),
        "mean_connected_regions_active": d["mean_connected_regions_active"],
        "mean_effective_regions_active": d["mean_effective_regions_active"],
        "condition_number_v_gram": float(d["condition_number_v_gram"]),
        "converged": bool(d["converged"]),
        "n_outer_iterations": int(d["n_outer_iterations"]),
        "final_objective": float(last["objective"]),
        "final_reconstruction_error": float(last["reconstruction_error"]),
        "final_relative_objective_change": float(last["relative_objective_change"]),
        "final_relative_reconstruction_change": float(
            last["relative_reconstruction_change"]
        ),
        "final_inner_converged": bool(last["inner"]["converged"]),
        "final_inner_iterations": int(last["inner"]["iterations"]),
        "final_inner_relative_change": float(last["inner"]["relative_change"]),
        "final_inner_primal_dual_residual": float(
            last["inner"]["primal_dual_residual"]
        ),
        "all_inner_converged": bool(np.all(inner_converged)),
        "mean_inner_iterations": float(np.mean(inner_iterations)),
        "max_inner_iterations": int(np.max(inner_iterations)),
        "total_reinitializations": int(np.sum(d["reinitialization_counts"])),
        "elapsed_seconds": float(elapsed),
        "worker_peak_rss_mb": rss_mb,
        "started_at": started,
        "finished_at": utc_now(),
        "component_order": "active first; descending conditional contribution",
    }


def _record_from_completed_fit(output: Path, task: dict[str, Any]) -> dict[str, Any]:
    fit_dir = output / task["fit_relative_dir"]
    metrics = read_json(fit_dir / "metrics.json")
    components = list(read_csv_rows(fit_dir / "components.csv"))
    # Restore useful scalar types for database exports after recovery.
    typed_components: list[dict[str, Any]] = []
    for row in components:
        converted: dict[str, Any] = {"fit_id": row["fit_id"]}
        for key, value in row.items():
            if key == "fit_id":
                continue
            if key == "active":
                converted[key] = value.lower() == "true"
            elif key in {
                "component",
                "active_cells",
                "connected_regions",
                "effective_regions",
                "largest_region_size",
                "reinitializations",
            }:
                converted[key] = int(value)
            else:
                converted[key] = float(value)
        typed_components.append(converted)
    return {
        "fit": metrics,
        "components": typed_components,
        "fit_relative_dir": task["fit_relative_dir"],
        "loadings_relative_path": str(
            Path(task["fit_relative_dir"]) / "loadings.npz"
        ),
        "created_at": utc_now(),
    }


def run_fit_task(task: dict[str, Any]) -> dict[str, Any]:
    config, output, dataset = _require_state()
    fit_dir = output / task["fit_relative_dir"]
    record_path = output / "records" / f"{task['fit_id']}.json"
    running_path = output / "running" / f"{task['fit_id']}.json"

    if (fit_dir / "DONE").exists():
        if not record_path.exists():
            atomic_write_json(
                record_path, _record_from_completed_fit(output, task), pretty=False
            )
        return {"status": "already_done", "fit_id": task["fit_id"]}

    started = utc_now()
    atomic_write_json(
        running_path,
        {
            "fit_id": task["fit_id"],
            "pid": os.getpid(),
            "started_at": started,
            "task": task,
        },
        pretty=False,
    )
    temporary = output / "tmp" / f"{task['fit_id']}.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    timer = time.perf_counter()
    try:
        from calendar_gfspca import CalendarGraphFusedSparsePCA

        indices = _sample_indices(task)
        x = IndexedRowsMatrix(
            dataset,
            indices,
            int(config["feature_start"]),
            int(config["feature_count"]),
        )
        model_parameters = dict(config["model"])
        model_parameters.update(
            {
                "n_components": int(task["K"]),
                "lambda_l1": float(task["lambda_l1"]),
                "lambda_tv": float(task["lambda_tv"]),
                "calendar_shape": tuple(map(int, config["calendar_shape"])),
                "order": config["order"],
                "center": bool(config["center"]),
                "random_state": int(task["initialization_seed"]),
                "verbose": 0,
            }
        )
        model = CalendarGraphFusedSparsePCA(**model_parameters).fit(x)
        model.reorder_components("contribution")
        elapsed = time.perf_counter() - timer
        metrics = _metrics(model, task, started, elapsed)
        component_rows = _component_rows(model, task)

        atomic_write_json(temporary / "config.json", {"task": task, "model": model_parameters})
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_csv(
            temporary / "components.csv", COMPONENT_FIELDS, component_rows
        )
        atomic_write_csv(
            temporary / "history.csv.gz",
            HISTORY_FIELDS,
            _history_rows(model),
            gzip_output=True,
        )
        np.savez_compressed(
            temporary / "loadings.npz",
            components=np.asarray(model.components_, dtype=np.float64),
            active=np.asarray(model.active_, dtype=np.bool_),
            conditional_contribution_percent=np.asarray(
                model.conditional_contribution_percent_, dtype=np.float64
            ),
            calendar_shape=np.asarray(config["calendar_shape"], dtype=np.int64),
            order=np.asarray(config["order"]),
        )
        _save_component_pngs(
            temporary / "component_png", model, task, component_rows, config
        )
        atomic_write_json(
            temporary / "run.json",
            {
                "fit_id": task["fit_id"],
                "started_at": started,
                "finished_at": metrics["finished_at"],
                "worker_pid": os.getpid(),
            },
        )
        (temporary / "DONE").touch()
        fit_dir.parent.mkdir(parents=True, exist_ok=True)
        if fit_dir.exists():
            raise RuntimeError(f"Unexpected pre-existing incomplete fit directory: {fit_dir}")
        os.replace(temporary, fit_dir)

        record = {
            "fit": metrics,
            "components": component_rows,
            "fit_relative_dir": task["fit_relative_dir"],
            "loadings_relative_path": str(
                Path(task["fit_relative_dir"]) / "loadings.npz"
            ),
            "created_at": utc_now(),
        }
        atomic_write_json(record_path, record, pretty=False)
        return {
            "status": "done",
            "fit_id": task["fit_id"],
            "elapsed_seconds": elapsed,
            "converged": metrics["converged"],
        }
    except Exception as error:
        failure = {
            "fit_id": task["fit_id"],
            "task": task,
            "failed_at": utc_now(),
            "worker_pid": os.getpid(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path = output / "failures" / (
            f"{task['fit_id']}.{int(time.time())}.{os.getpid()}.{uuid.uuid4().hex[:8]}.json"
        )
        atomic_write_json(failure_path, failure)
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        return {
            "status": "failed",
            "fit_id": task["fit_id"],
            "exception": str(error),
        }
    finally:
        try:
            running_path.unlink()
        except FileNotFoundError:
            pass
