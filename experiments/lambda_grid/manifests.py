"""Manifest generation and task lookup."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .utils import atomic_write_csv, read_csv, snapshot_config


MANIFEST_FIELDS = [
    "phase",
    "task_id",
    "global_task_id",
    "run_id",
    "n_samples",
    "n_components",
    "lambda_l1_index",
    "lambda_l1",
    "lambda_tv_index",
    "lambda_tv",
    "seed",
    "config_sha256",
]


def _run_id(n: int, l1_index: int, tv_index: int, seed: int) -> str:
    return f"n{n:05d}_l1i{l1_index:02d}_ltvi{tv_index:02d}_seed{seed:04d}"


def build_manifest_rows(config: ExperimentConfig) -> dict[str, list[dict[str, Any]]]:
    pilot: list[dict[str, Any]] = []
    restarts: list[dict[str, Any]] = []
    global_task_id = 0
    for seed_index, seed in enumerate(config.grid.seeds):
        destination = pilot if seed_index == 0 else restarts
        phase = "pilot" if seed_index == 0 else "restarts"
        for n, (l1_index, l1), (tv_index, tv) in product(
            config.sampling.sizes,
            enumerate(config.grid.lambda_l1),
            enumerate(config.grid.lambda_tv),
        ):
            row = {
                "phase": phase,
                "task_id": len(destination),
                "global_task_id": global_task_id,
                "run_id": _run_id(n, l1_index, tv_index, seed),
                "n_samples": n,
                "n_components": config.model.n_components,
                "lambda_l1_index": l1_index,
                "lambda_l1": l1,
                "lambda_tv_index": tv_index,
                "lambda_tv": tv,
                "seed": seed,
                "config_sha256": config.sha256,
            }
            destination.append(row)
            global_task_id += 1
    return {"pilot": pilot, "restarts": restarts, "all": pilot + restarts}


def write_manifests(config: ExperimentConfig) -> dict[str, Path]:
    config.validate()
    snapshot_config(config)
    rows_by_phase = build_manifest_rows(config)
    paths: dict[str, Path] = {}
    for phase, rows in rows_by_phase.items():
        path = config.output_path / "manifests" / f"{phase}.csv"
        atomic_write_csv(path, rows, MANIFEST_FIELDS)
        paths[phase] = path
        print(f"Wrote {len(rows):5d} tasks: {path}")
    return paths


def resolve_manifest(config: ExperimentConfig, manifest: str | Path) -> Path:
    candidate = Path(manifest)
    if candidate.suffix.lower() == ".csv" or candidate.parent != Path("."):
        path = candidate.expanduser().resolve()
    else:
        path = config.output_path / "manifests" / f"{manifest}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    return path


def get_task(
    config: ExperimentConfig,
    manifest: str | Path,
    task_id: int,
) -> dict[str, str]:
    path = resolve_manifest(config, manifest)
    rows = read_csv(path)
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(
            f"task_id={task_id} is outside [0, {max(len(rows) - 1, 0)}] "
            f"for {path}."
        )
    row = rows[task_id]
    if int(row["task_id"]) != task_id:
        raise RuntimeError(f"Manifest task ordering is inconsistent in {path}.")
    if row["config_sha256"] != config.sha256:
        raise RuntimeError(
            "Manifest and experiment configuration have different hashes."
        )
    return row
