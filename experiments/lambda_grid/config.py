"""Configuration model for the lambda-grid experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("experiment.json")


@dataclass(frozen=True)
class InputConfig:
    path: str = "/home/cquesada/pca/dataset/matrix_normalized.npy"
    imputed_column_1based: int = 3
    max_imputed_hours: int = 72
    feature_start_column_1based: int = 4
    feature_end_column_1based: int = 8739
    allow_pickle: bool = False

    @property
    def imputed_index(self) -> int:
        return self.imputed_column_1based - 1

    @property
    def feature_slice(self) -> slice:
        # Python's stop is exclusive; the one-based end column is therefore
        # already the correct zero-based stop value.
        return slice(
            self.feature_start_column_1based - 1,
            self.feature_end_column_1based,
        )

    @property
    def n_features(self) -> int:
        return (
            self.feature_end_column_1based
            - self.feature_start_column_1based
            + 1
        )


@dataclass(frozen=True)
class SamplingConfig:
    sizes: tuple[int, ...] = (5_000, 10_000, 15_000, 20_000)
    seed: int = 20_260_809
    extraction_batch_rows: int = 256
    stored_dtype: str = "float64"
    hash_input_file: bool = True

    @property
    def master_size(self) -> int:
        return max(self.sizes)


@dataclass(frozen=True)
class GridConfig:
    lambda_l1: tuple[float, ...] = (
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
    )
    lambda_tv: tuple[float, ...] = (
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
    )
    seeds: tuple[int, ...] = (11, 23, 47, 71, 101)


@dataclass(frozen=True)
class ModelConfig:
    n_components: int = 10
    calendar_shape: tuple[int, int, int] = (24, 7, 52)
    order: str = "C"
    center: bool = True
    score_solver: str = "coordinate"
    score_sweeps: int = 50
    score_tolerance: float = 1e-7
    outer_max_iter: int = 1_000
    outer_objective_tolerance: float = 1e-6
    outer_reconstruction_tolerance: float = 1e-6
    inner_max_iter: int = 20_000
    inner_check_interval: int = 25
    epsilon_0: float = 1e-3
    epsilon_min: float = 1e-6
    epsilon_rho: float = 0.75
    step_safety: float = 0.99
    dual_step_scale: float = 1.0
    loading_collapse_tolerance: float = 1e-10
    score_collapse_tolerance: float = 1e-12
    similarity_threshold: float = 0.995
    condition_threshold: float = 1e12
    max_reinitializations: int = 3
    reinit_power_iterations: int = 5
    initialization: str = "svd"
    svd_power_iterations: int = 2
    svd_oversamples: int = 5
    batch_size: int = 2_048
    sparsity_tolerance: float = 1e-10
    effective_region_min_active_fraction: float = 0.005
    effective_region_min_l1_fraction: float = 0.01
    verbose: int = 1

    def estimator_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "lambda_grid_v1"
    output_dir: str = "/home/cquesada/pca/results/lambda_grid_v1"
    input: InputConfig = field(default_factory=InputConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty.")
        if not self.sampling.sizes or any(size < 1 for size in self.sampling.sizes):
            raise ValueError("sampling.sizes must contain positive integers.")
        if tuple(sorted(set(self.sampling.sizes))) != self.sampling.sizes:
            raise ValueError("sampling.sizes must be unique and increasing.")
        if self.sampling.extraction_batch_rows < 1:
            raise ValueError("sampling.extraction_batch_rows must be positive.")
        np.dtype(self.sampling.stored_dtype)
        if self.input.imputed_column_1based < 1:
            raise ValueError("The imputed column must be one-based and positive.")
        if self.input.feature_start_column_1based < 1:
            raise ValueError("The first feature column must be positive.")
        if (
            self.input.feature_end_column_1based
            < self.input.feature_start_column_1based
        ):
            raise ValueError("The final feature column precedes the first one.")
        expected_features = int(np.prod(self.model.calendar_shape))
        if self.input.n_features != expected_features:
            raise ValueError(
                f"The selected input range has {self.input.n_features} columns, "
                f"but calendar_shape contains {expected_features} cells."
            )
        if not self.grid.lambda_l1 or not self.grid.lambda_tv:
            raise ValueError("Both lambda axes must be non-empty.")
        if any(value < 0 for value in (*self.grid.lambda_l1, *self.grid.lambda_tv)):
            raise ValueError("Lambda values must be non-negative.")
        if len(set(self.grid.seeds)) != len(self.grid.seeds) or not self.grid.seeds:
            raise ValueError("Algorithm seeds must be non-empty and unique.")
        if self.model.n_components < 1:
            raise ValueError("model.n_components must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _as_tuple_values(data: dict[str, Any], key: str) -> None:
    if key in data:
        data[key] = tuple(data[key])


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    input_data = dict(data.get("input", {}))
    sampling_data = dict(data.get("sampling", {}))
    grid_data = dict(data.get("grid", {}))
    model_data = dict(data.get("model", {}))
    _as_tuple_values(sampling_data, "sizes")
    _as_tuple_values(grid_data, "lambda_l1")
    _as_tuple_values(grid_data, "lambda_tv")
    _as_tuple_values(grid_data, "seeds")
    _as_tuple_values(model_data, "calendar_shape")
    config = ExperimentConfig(
        name=data.get("name", ExperimentConfig.name),
        output_dir=data.get("output_dir", ExperimentConfig.output_dir),
        input=InputConfig(**input_data),
        sampling=SamplingConfig(**sampling_data),
        grid=GridConfig(**grid_data),
        model=ModelConfig(**model_data),
    )
    config.validate()
    return config


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return config_from_dict(data)
