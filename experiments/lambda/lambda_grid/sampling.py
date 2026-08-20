"""Reproducible filtering and nested random sampling."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from .config import ExperimentConfig
from .utils import (
    atomic_write_json,
    ensure_layout,
    sha256_array,
    sha256_file,
    snapshot_config,
)


SAMPLE_METADATA = "sample_metadata.json"
SOURCE_INDICES = "source_indices.npy"
IMPUTED_VALUES = "imputed_values.npy"


def sample_paths(config: ExperimentConfig) -> dict[str, Path]:
    base = config.output_path / "samples"
    return {
        "metadata": base / SAMPLE_METADATA,
        "master": base / f"master_{config.sampling.master_size}.npy",
        "indices": base / SOURCE_INDICES,
        "imputed": base / IMPUTED_VALUES,
    }


def _load_source(config: ExperimentConfig):
    path = Path(config.input.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input matrix does not exist: {path}")
    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=config.input.allow_pickle,
        )
    except ValueError as error:
        if "memory-mapped" in str(error) and config.input.allow_pickle:
            array = np.load(path, allow_pickle=True)
        else:
            raise
    if array.ndim != 2:
        raise ValueError(f"Input matrix must be two-dimensional; got {array.shape}.")
    required = max(
        config.input.imputed_column_1based,
        config.input.feature_end_column_1based,
    )
    if array.shape[1] < required:
        raise ValueError(
            f"Input matrix has {array.shape[1]} columns; column {required} is required."
        )
    return path, array


def _eligible_rows(source, config: ExperimentConfig) -> np.ndarray:
    n_rows = int(source.shape[0])
    eligible = np.zeros(n_rows, dtype=bool)
    block_size = max(10_000, config.sampling.extraction_batch_rows)
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        imputed = np.asarray(
            source[start:stop, config.input.imputed_index], dtype=np.float64
        )
        eligible[start:stop] = imputed <= config.input.max_imputed_hours
    return np.flatnonzero(eligible)


def prepare_samples(config: ExperimentConfig) -> dict:
    """Create one random 20k master sample and nested prefix subsets.

    Eligibility is determined only by ``imputed <= max_imputed_hours``. The
    extracted matrix contains only the configured inclusive feature range.
    No value-domain, NaN, infinity, or year-distribution checks are performed.
    """

    config.validate()
    ensure_layout(config)
    snapshot_config(config)
    paths = sample_paths(config)
    if all(path.exists() for path in paths.values()):
        with paths["metadata"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("config_sha256") != config.sha256:
            raise RuntimeError(
                "Existing sample artifacts were created with another configuration."
            )
        print(f"Samples already prepared: {paths['master']}")
        return metadata

    partially_existing = [str(path) for path in paths.values() if path.exists()]
    if partially_existing:
        raise RuntimeError(
            "Partial sample artifacts already exist; move them aside or select a "
            f"new output_dir: {partially_existing}"
        )

    started = time.perf_counter()
    input_path, source = _load_source(config)
    print(f"Input: {input_path}")
    print(f"Source shape={source.shape} dtype={source.dtype}")
    print(
        "Eligibility filter: "
        f"column={config.input.imputed_column_1based} "
        f"imputed<={config.input.max_imputed_hours}"
    )
    eligible = _eligible_rows(source, config)
    print(f"Eligible rows: {eligible.size}/{source.shape[0]}")
    if eligible.size < config.sampling.master_size:
        raise ValueError(
            f"Only {eligible.size} rows satisfy imputed <= "
            f"{config.input.max_imputed_hours}; "
            f"{config.sampling.master_size} are required."
        )

    rng = np.random.default_rng(config.sampling.seed)
    selected = rng.choice(
        eligible, size=config.sampling.master_size, replace=False
    ).astype(np.int64, copy=False)
    selected_imputed = np.asarray(
        source[selected, config.input.imputed_index], dtype=np.int64
    )
    np.save(paths["indices"], selected, allow_pickle=False)
    np.save(paths["imputed"], selected_imputed, allow_pickle=False)

    dtype = np.dtype(config.sampling.stored_dtype)
    temporary = paths["master"].with_name(
        f".{paths['master'].stem}.tmp-{os.getpid()}.npy"
    )
    master = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=dtype,
        shape=(config.sampling.master_size, config.input.n_features),
    )
    step = config.sampling.extraction_batch_rows
    feature_slice = config.input.feature_slice
    print(
        "Extracting inclusive columns "
        f"{config.input.feature_start_column_1based}-"
        f"{config.input.feature_end_column_1based} into {paths['master']}"
    )
    for start in range(0, config.sampling.master_size, step):
        stop = min(start + step, config.sampling.master_size)
        master[start:stop] = np.asarray(
            source[selected[start:stop], feature_slice], dtype=dtype
        )
        if stop == config.sampling.master_size or stop % 2_000 == 0:
            print(f"Extracted {stop}/{config.sampling.master_size} rows")
    master.flush()
    del master
    os.replace(temporary, paths["master"])

    print("Computing reproducibility hashes")
    input_sha256 = (
        sha256_file(input_path) if config.sampling.hash_input_file else None
    )
    metadata = {
        "experiment": config.name,
        "config_sha256": config.sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_mtime_ns": input_path.stat().st_mtime_ns,
        "input_sha256": input_sha256,
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "eligibility": {
            "imputed_column_1based": config.input.imputed_column_1based,
            "max_imputed_hours_inclusive": config.input.max_imputed_hours,
            "eligible_rows": int(eligible.size),
            "excluded_rows": int(source.shape[0] - eligible.size),
        },
        "sampling": {
            "method": "simple_random_without_replacement",
            "seed": config.sampling.seed,
            "master_size": config.sampling.master_size,
            "nested_prefix_sizes": list(config.sampling.sizes),
            "source_indices_sha256": sha256_array(selected),
        },
        "features": {
            "start_column_1based_inclusive": (
                config.input.feature_start_column_1based
            ),
            "end_column_1based_inclusive": config.input.feature_end_column_1based,
            "n_features": config.input.n_features,
            "stored_dtype": str(dtype),
        },
        "no_additional_value_checks_performed": True,
        "master_sample_path": str(paths["master"]),
        "master_sample_sha256": sha256_file(paths["master"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(paths["metadata"], metadata)
    print(
        f"DONE sample rows={config.sampling.master_size} "
        f"features={config.input.n_features} "
        f"elapsed={metadata['elapsed_seconds']:.1f}s"
    )
    return metadata
