from __future__ import annotations

from itertools import combinations
import hashlib
import os
from pathlib import Path
import time
import traceback
from typing import Any
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io_utils import atomic_write_json, utc_now


_OUTPUT: Path | None = None


def _pair_id(kind: str, fit_a: str, fit_b: str) -> str:
    digest = hashlib.sha256(f"{kind}|{fit_a}|{fit_b}".encode()).hexdigest()[:24]
    return f"{kind}_{digest}"


def build_stability_tasks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_grid = {
        (
            int(record["repeat"]),
            int(record["N"]),
            int(record["K"]),
            int(record["l1_index"]),
            int(record["ltv_index"]),
        ): record
        for record in records
    }
    tasks: list[dict[str, Any]] = []

    # Four-neighbor grid edges are represented once: +lambda_1 and +lambda_TV.
    for key, record in by_grid.items():
        repeat, n_samples, k_value, l1_index, ltv_index = key
        for direction, neighbor_key in (
            ("lambda_l1", (repeat, n_samples, k_value, l1_index + 1, ltv_index)),
            ("lambda_tv", (repeat, n_samples, k_value, l1_index, ltv_index + 1)),
        ):
            neighbor = by_grid.get(neighbor_key)
            if neighbor is not None:
                tasks.append(_make_task("local", direction, record, neighbor))

    by_repetition_key: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            int(record["N"]),
            int(record["K"]),
            int(record["l1_index"]),
            int(record["ltv_index"]),
        )
        by_repetition_key.setdefault(key, []).append(record)
    for group in by_repetition_key.values():
        group.sort(key=lambda record: int(record["repeat"]))
        for first, second in combinations(group, 2):
            tasks.append(_make_task("repeat", "repeat", first, second))
    return tasks


def _make_task(
    kind: str,
    direction: str,
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    task = {
        "kind": kind,
        "direction": direction,
        "fit_a": first["fit_id"],
        "fit_b": second["fit_id"],
        "repeat_a": int(first["repeat"]),
        "repeat_b": int(second["repeat"]),
        "N": int(first["N"]),
        "K": int(first["K"]),
        "l1_index_a": int(first["l1_index"]),
        "l1_index_b": int(second["l1_index"]),
        "ltv_index_a": int(first["ltv_index"]),
        "ltv_index_b": int(second["ltv_index"]),
        "lambda_l1_a": float(first["lambda_l1"]),
        "lambda_l1_b": float(second["lambda_l1"]),
        "lambda_tv_a": float(first["lambda_tv"]),
        "lambda_tv_b": float(second["lambda_tv"]),
        "K_effective_a": int(first["K_effective"]),
        "K_effective_b": int(second["K_effective"]),
        "loadings_a": first["loadings_relative_path"],
        "loadings_b": second["loadings_relative_path"],
    }
    task["pair_id"] = _pair_id(kind, task["fit_a"], task["fit_b"])
    return task


def initialize_stability_worker(output_dir: str) -> None:
    global _OUTPUT
    _OUTPUT = Path(output_dir)


def _cosine_match(
    components_a: np.ndarray,
    active_a: np.ndarray,
    components_b: np.ndarray,
    active_b: np.ndarray,
    nominal_k: int,
) -> dict[str, Any]:
    a = components_a[:, active_a]
    b = components_b[:, active_b]
    if a.shape[1] == 0 or b.shape[1] == 0:
        return {
            "matched_active_components": 0,
            "mean_matched_active_cosine": 0.0,
            "penalized_similarity": 0.0,
        }
    norms = np.maximum(
        np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :],
        1e-30,
    )
    similarity = np.abs(a.T @ b) / norms
    rows, columns = linear_sum_assignment(-similarity)
    values = similarity[rows, columns]
    return {
        "matched_active_components": int(values.size),
        "mean_matched_active_cosine": float(np.mean(values)) if values.size else 0.0,
        # Missing/inactive components contribute zero.  Because comparisons
        # share nominal K, this score cannot be inflated by component collapse.
        "penalized_similarity": float(np.sum(values) / max(int(nominal_k), 1)),
    }


def run_stability_task(task: dict[str, Any]) -> dict[str, Any]:
    if _OUTPUT is None:
        raise RuntimeError("Stability worker was not initialized.")
    record_path = _OUTPUT / "stability_records" / f"{task['pair_id']}.json"
    if record_path.exists():
        return {"status": "already_done", "pair_id": task["pair_id"]}
    try:
        with np.load(_OUTPUT / task["loadings_a"], allow_pickle=False) as first:
            components_a = np.asarray(first["components"], dtype=np.float64)
            active_a = np.asarray(first["active"], dtype=bool)
        with np.load(_OUTPUT / task["loadings_b"], allow_pickle=False) as second:
            components_b = np.asarray(second["components"], dtype=np.float64)
            active_b = np.asarray(second["active"], dtype=bool)
        matched = _cosine_match(
            components_a,
            active_a,
            components_b,
            active_b,
            int(task["K"]),
        )
        record = {
            **{key: value for key, value in task.items() if not key.startswith("loadings_")},
            **matched,
            "computed_at": utc_now(),
        }
        atomic_write_json(record_path, record, pretty=False)
        return {"status": "done", "pair_id": task["pair_id"]}
    except Exception as error:
        failure = {
            "pair_id": task["pair_id"],
            "task": task,
            "failed_at": utc_now(),
            "worker_pid": os.getpid(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }
        path = _OUTPUT / "failures" / (
            f"stability.{task['pair_id']}.{int(time.time())}.{uuid.uuid4().hex[:8]}.json"
        )
        atomic_write_json(path, failure)
        return {"status": "failed", "pair_id": task["pair_id"], "exception": str(error)}
