from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

import numpy as np

from .config import public_config, scientific_config_hash, total_fits
from .io_utils import (
    atomic_save_npy,
    atomic_write_csv,
    atomic_write_json,
    read_json,
    sha256_array,
    sha256_file,
    utc_now,
)


OUTPUT_DIRS = (
    "fits",
    "records",
    "running",
    "failures",
    "tmp",
    "samples",
    "tables",
    "dashboard",
    "dashboard/heatmaps",
    "stability_records",
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def inspect_repository(config: dict[str, Any]) -> dict[str, Any]:
    repo = Path(config["cycle_spca_repo"])
    if not (repo / "calendar_gfspca" / "model.py").is_file():
        raise FileNotFoundError(
            f"CycleSPCA source was not found at {repo}. "
            "Expected calendar_gfspca/model.py."
        )
    commit = _git(repo, "rev-parse", "HEAD")
    expected = config.get("expected_cycle_spca_commit")
    if expected and commit != expected:
        raise RuntimeError(
            f"CycleSPCA commit mismatch: expected {expected}, found {commit}."
        )
    tracked_status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if config.get("require_clean_tracked_files", True) and tracked_status:
        raise RuntimeError(
            "CycleSPCA has modified tracked files. Commit or restore them before "
            "starting the experiment.\n" + tracked_status
        )
    return {
        "path": str(repo),
        "commit": commit,
        "tracked_files_clean": not bool(tracked_status),
        "remote": _git(repo, "remote", "get-url", "origin") if _has_origin(repo) else None,
    }


def _has_origin(repo: Path) -> bool:
    try:
        _git(repo, "remote", "get-url", "origin")
        return True
    except subprocess.CalledProcessError:
        return False


def inspect_dataset(config: dict[str, Any], *, calculate_hash: bool) -> dict[str, Any]:
    path = Path(config["dataset_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("The dataset NPY must be a two-dimensional numeric array.")
    feature_start = int(config["feature_start"])
    feature_stop = feature_start + int(config["feature_count"])
    if feature_start < 0 or feature_stop > data.shape[1]:
        raise ValueError(
            f"Feature slice [{feature_start}:{feature_stop}] does not fit dataset "
            f"shape {data.shape}."
        )
    imputed_column = int(config["imputed_column"])
    if not (0 <= imputed_column < data.shape[1]):
        raise ValueError("imputed_column is outside the dataset.")
    imputed = np.asarray(data[:, imputed_column], dtype=np.float64)
    eligible = np.flatnonzero(np.isfinite(imputed) & (imputed <= float(config["max_imputed"])))
    if eligible.size < max(map(int, config["N_values"])):
        raise ValueError(
            f"Only {eligible.size} rows satisfy imputed <= {config['max_imputed']}; "
            f"the largest requested N is {max(config['N_values'])}."
        )
    stat = path.stat()
    result = {
        "path": str(path),
        "shape": list(map(int, data.shape)),
        "dtype": str(data.dtype),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "feature_start": feature_start,
        "feature_stop_exclusive": feature_stop,
        "feature_count": int(config["feature_count"]),
        "imputed_column": imputed_column,
        "max_imputed": float(config["max_imputed"]),
        "eligible_rows": int(eligible.size),
        "excluded_rows": int(data.shape[0] - eligible.size),
    }
    if calculate_hash:
        result["sha256"] = sha256_file(path)
    return {"info": result, "eligible": eligible, "data": data}


def _write_sample_metadata(
    path: Path,
    source: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
) -> None:
    id_column = int(config.get("id_column", 0))
    year_column = int(config.get("iso_year_column", 1))
    imputed_column = int(config["imputed_column"])
    metadata = np.asarray(source[indices, : max(id_column, year_column, imputed_column) + 1])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["position", "source_row_index", "id", "iso_year", "imputed"])
            for position, (row_index, row) in enumerate(zip(indices, metadata)):
                writer.writerow(
                    [position, int(row_index), row[id_column], row[year_column], row[imputed_column]]
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_samples(
    config: dict[str, Any],
    dataset: np.ndarray,
    eligible: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    sample_dir = output / "samples"
    master_n = max(map(int, config["N_values"]))
    repetitions: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for repeat, seed in enumerate(config["sample_seeds"], start=1):
        rng = np.random.default_rng(int(seed))
        indices = rng.permutation(eligible)[:master_n].astype(np.int64, copy=False)
        indices_path = sample_dir / f"repeat_{repeat:02d}_master_indices.npy"
        metadata_path = sample_dir / f"repeat_{repeat:02d}_master_metadata.csv.gz"
        atomic_save_npy(indices_path, indices)
        _write_sample_metadata(metadata_path, dataset, indices, config)
        master_hash = sha256_array(indices)
        repetition = {
            "repeat": repeat,
            "sample_seed": int(seed),
            "master_N": master_n,
            "indices_path": str(indices_path),
            "metadata_path": str(metadata_path),
            "master_indices_sha256": master_hash,
        }
        repetitions.append(repetition)
        for n_samples in config["N_values"]:
            prefix = indices[: int(n_samples)]
            sample_rows.append(
                {
                    "repeat": repeat,
                    "N": int(n_samples),
                    "sample_seed": int(seed),
                    "indices_sha256": sha256_array(prefix),
                    "master_indices_path": str(indices_path),
                    "prefix_length": int(n_samples),
                }
            )
    manifest = {
        "created_at": utc_now(),
        "nested_samples": True,
        "eligible_rows": int(eligible.size),
        "master_N": master_n,
        "repetitions": repetitions,
    }
    atomic_write_json(sample_dir / "manifest.json", manifest)
    atomic_write_csv(
        output / "tables" / "samples.csv",
        [
            "repeat",
            "N",
            "sample_seed",
            "indices_sha256",
            "master_indices_path",
            "prefix_length",
        ],
        sample_rows,
    )
    return manifest


def initialize_experiment(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"])
    for relative in OUTPUT_DIRS:
        (output / relative).mkdir(parents=True, exist_ok=True)

    experiment_path = output / "experiment.json"
    config_hash = scientific_config_hash(config)
    repository = inspect_repository(config)

    if experiment_path.exists():
        experiment = read_json(experiment_path)
        if experiment.get("scientific_config_sha256") != config_hash:
            raise RuntimeError(
                "This output directory belongs to a different scientific "
                "configuration. Choose a new output_dir."
            )
        current = inspect_dataset(config, calculate_hash=False)["info"]
        previous = experiment["dataset"]
        for field in ("path", "size_bytes", "mtime_ns", "shape", "dtype"):
            if current.get(field) != previous.get(field):
                raise RuntimeError(
                    f"Dataset identity changed in field {field}. Choose a new "
                    "output_dir or restore the original dataset."
                )
        if repository["commit"] != experiment["repository"]["commit"]:
            raise RuntimeError("CycleSPCA commit changed after experiment initialization.")
        sample_manifest = read_json(output / "samples" / "manifest.json")
        return {"experiment": experiment, "samples": sample_manifest}

    inspected = inspect_dataset(config, calculate_hash=True)
    sample_manifest = _build_samples(
        config, inspected["data"], inspected["eligible"], output
    )
    experiment = {
        "created_at": utc_now(),
        "scientific_config_sha256": config_hash,
        "config": public_config(config),
        "repository": repository,
        "dataset": inspected["info"],
        "sampling": {
            "eligible_rule": f"imputed <= {config['max_imputed']}",
            "nested": True,
            "manifest_path": str(output / "samples" / "manifest.json"),
        },
        "calendar_mapping": {
            "shape": config["calendar_shape"],
            "order": config["order"],
            "heatmap_column_formula": "column = week * n_days + day",
        },
        "total_fits": total_fits(config),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    atomic_write_json(output / "grid_config.snapshot.json", public_config(config))
    atomic_write_json(experiment_path, experiment)
    return {"experiment": experiment, "samples": sample_manifest}


class RunnerLock:
    def __init__(self, output: Path, *, force: bool = False) -> None:
        self.path = output / "RUNNER.lock"
        self.force = force
        self.owned = False

    def __enter__(self) -> "RunnerLock":
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": utc_now(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_json(self.path)
            live = _lock_is_live(existing)
            if live and not self.force:
                raise RuntimeError(
                    f"Another runner appears active: PID {existing.get('pid')} on "
                    f"{existing.get('hostname')}."
                )
            stale = self.path.with_name(
                f"RUNNER.stale.{existing.get('pid', 'unknown')}.{os.getpid()}.json"
            )
            os.replace(self.path, stale)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        self.owned = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.owned and self.path.exists():
            try:
                current = read_json(self.path)
                if int(current.get("pid", -1)) == os.getpid():
                    self.path.unlink()
            except (OSError, ValueError):
                pass


def _lock_is_live(payload: dict[str, Any]) -> bool:
    if payload.get("hostname") != socket.gethostname():
        return True
    try:
        os.kill(int(payload["pid"]), 0)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def clear_stale_running_files(output: Path) -> None:
    running = output / "running"
    for path in running.glob("*.json"):
        try:
            path.unlink()
        except OSError:
            pass
