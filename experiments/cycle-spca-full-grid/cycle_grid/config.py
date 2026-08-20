from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .io_utils import object_sha256


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)

    for name in ("dataset_path", "cycle_spca_repo", "output_dir"):
        if name not in config:
            raise ValueError(f"Missing required configuration field: {name}")
        config[name] = str(Path(config[name]).expanduser().resolve())

    config.setdefault("feature_start", 3)
    config.setdefault("feature_count", 8736)
    config.setdefault("imputed_column", 2)
    config.setdefault("max_imputed", 72)
    config.setdefault("calendar_shape", [24, 7, 52])
    config.setdefault("order", "F")
    config.setdefault("center", True)
    config.setdefault("task_order_seed", 20260810)
    config.setdefault("require_clean_tracked_files", True)
    config.setdefault("expected_cycle_spca_commit", None)
    config.setdefault("model", {})
    config.setdefault("png", {})
    config.setdefault("runtime", {})
    config.setdefault("stability", {})

    png = config["png"]
    png.setdefault("enabled", True)
    png.setdefault("dpi", 150)
    png.setdefault("figsize", [14.0, 6.0])
    png.setdefault("palette_colors", 256)
    png.setdefault("compression_level", 6)

    runtime = config["runtime"]
    runtime.setdefault("workers", 90)
    runtime.setdefault("refresh_seconds", 60)
    runtime.setdefault("plot_refresh_seconds", 900)
    runtime.setdefault("components_export_seconds", 1800)
    runtime.setdefault("maxtasksperchild", 100)

    stability = config["stability"]
    stability.setdefault("enabled", True)
    stability.setdefault("workers", runtime["workers"])

    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    lambdas_l1 = config.get("lambda_l1_values")
    lambdas_tv = config.get("lambda_tv_values")
    k_values = config.get("K_values")
    n_values = config.get("N_values")
    sample_seeds = config.get("sample_seeds")
    initialization_seeds = config.get("initialization_seeds")
    required_lists = {
        "lambda_l1_values": lambdas_l1,
        "lambda_tv_values": lambdas_tv,
        "K_values": k_values,
        "N_values": n_values,
        "sample_seeds": sample_seeds,
        "initialization_seeds": initialization_seeds,
    }
    for name, values in required_lists.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty JSON list.")
    if len(sample_seeds) != len(initialization_seeds):
        raise ValueError("sample_seeds and initialization_seeds must have equal length.")
    if len(set(sample_seeds)) != len(sample_seeds):
        raise ValueError("sample_seeds must be unique.")
    if any(float(value) < 0 for value in [*lambdas_l1, *lambdas_tv]):
        raise ValueError("Lambda values must be non-negative.")
    if sorted(set(map(float, lambdas_l1))) != sorted(map(float, lambdas_l1)):
        raise ValueError("lambda_l1_values must be sorted and unique.")
    if sorted(set(map(float, lambdas_tv))) != sorted(map(float, lambdas_tv)):
        raise ValueError("lambda_tv_values must be sorted and unique.")
    if sorted(set(map(int, k_values))) != list(map(int, k_values)):
        raise ValueError("K_values must be sorted, positive, and unique.")
    if sorted(set(map(int, n_values))) != list(map(int, n_values)):
        raise ValueError("N_values must be sorted, positive, and unique.")
    if min(map(int, k_values)) < 1 or min(map(int, n_values)) < 1:
        raise ValueError("K_values and N_values must be positive.")
    shape = tuple(map(int, config["calendar_shape"]))
    if len(shape) != 3 or shape[0] * shape[1] * shape[2] != int(config["feature_count"]):
        raise ValueError("calendar_shape product must equal feature_count.")
    if config["order"] not in {"C", "F"}:
        raise ValueError("order must be C or F.")
    if int(config["runtime"]["workers"]) < 1:
        raise ValueError("runtime.workers must be positive.")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("_config_path", None)
    return value


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    value = public_config(config)
    value.pop("runtime", None)
    value.pop("output_dir", None)
    return value


def scientific_config_hash(config: dict[str, Any]) -> str:
    return object_sha256(scientific_config(config))


def total_fits(config: dict[str, Any]) -> int:
    return (
        len(config["lambda_l1_values"])
        * len(config["lambda_tv_values"])
        * len(config["K_values"])
        * len(config["N_values"])
        * len(config["sample_seeds"])
    )


def fit_id(task: dict[str, Any]) -> str:
    return (
        f"r{task['repeat']:02d}_n{task['N']:06d}_k{task['K']:02d}_"
        f"l1{task['l1_index']:02d}_tv{task['ltv_index']:02d}"
    )


def fit_relative_dir(task: dict[str, Any]) -> Path:
    return Path(
        "fits",
        f"repeat_{task['repeat']:02d}",
        f"N_{task['N']:06d}",
        f"K_{task['K']:02d}",
        f"l1_{task['l1_index']:02d}",
        f"ltv_{task['ltv_index']:02d}",
    )


def all_tasks(config: dict[str, Any], sample_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import random

    by_repeat = {int(item["repeat"]): item for item in sample_manifest["repetitions"]}
    tasks: list[dict[str, Any]] = []
    for repeat, initialization_base in enumerate(config["initialization_seeds"], start=1):
        sample = by_repeat[repeat]
        for n_index, n_samples in enumerate(config["N_values"]):
            for k_index, k_value in enumerate(config["K_values"]):
                # All lambda pairs at fixed (repeat, N, K) receive exactly the
                # same randomized-SVD seed.
                initialization_seed = int(initialization_base) + 1000 * n_index + k_index
                for l1_index, lambda_l1 in enumerate(config["lambda_l1_values"]):
                    for ltv_index, lambda_tv in enumerate(config["lambda_tv_values"]):
                        task = {
                            "repeat": repeat,
                            "N": int(n_samples),
                            "K": int(k_value),
                            "l1_index": l1_index,
                            "ltv_index": ltv_index,
                            "lambda_l1": float(lambda_l1),
                            "lambda_tv": float(lambda_tv),
                            "sample_seed": int(config["sample_seeds"][repeat - 1]),
                            "initialization_seed": initialization_seed,
                            "sample_indices_path": sample["indices_path"],
                            "sample_master_hash": sample["master_indices_sha256"],
                        }
                        task["fit_id"] = fit_id(task)
                        task["fit_relative_dir"] = str(fit_relative_dir(task))
                        tasks.append(task)
    random.Random(int(config["task_order_seed"])).shuffle(tasks)
    return tasks
