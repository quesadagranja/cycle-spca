"""Four-by-four heatmap figure for the lambda study."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


METRICS = (
    ("explained_variance_mean", "Explained variance", "viridis"),
    ("mean_loading_sparsity_active_mean", "Loading sparsity", "viridis"),
    (
        "global_relative_total_variation_mean",
        "Relative total variation",
        "magma_r",
    ),
    ("stability_mean", "Between-seed stability", "viridis"),
)


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_cells(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Aggregated cells do not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_matrix(
    rows: list[dict[str, str]],
    n_samples: int,
    metric: str,
    n_l1: int,
    n_tv: int,
) -> np.ndarray:
    matrix = np.full((n_l1, n_tv), np.nan, dtype=np.float64)
    for row in rows:
        if int(row["n_samples"]) != n_samples:
            continue
        # During the pilot, a cell can be valid with one completed seed. Once
        # more seeds arrive, any failed convergence/audit/rank check masks it.
        if row["cell_valid_so_far"].lower() != "true":
            continue
        value = _float(row[metric])
        if np.isfinite(value):
            matrix[int(row["lambda_l1_index"]), int(row["lambda_tv_index"])] = value
    return matrix


def plot_four_maps(config: ExperimentConfig) -> dict[str, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install .[experiments]."
        ) from error

    cells_path = config.output_path / "aggregated" / "cells.csv"
    rows = _load_cells(cells_path)
    sizes = config.sampling.sizes
    n_l1 = len(config.grid.lambda_l1)
    n_tv = len(config.grid.lambda_tv)
    matrices: dict[tuple[str, int], np.ndarray] = {}
    for metric, _, _ in METRICS:
        for size in sizes:
            matrices[(metric, size)] = _metric_matrix(
                rows, size, metric, n_l1, n_tv
            )

    figure, axes = plt.subplots(
        len(METRICS),
        len(sizes),
        figsize=(25, 21),
        constrained_layout=True,
        squeeze=False,
    )
    figure.suptitle(
        f"CycleSPCA lambda grid — K={config.model.n_components}", fontsize=18
    )
    for metric_index, (metric, label, cmap_name) in enumerate(METRICS):
        finite_values = np.concatenate(
            [
                matrix[np.isfinite(matrix)]
                for size in sizes
                if (matrix := matrices[(metric, size)])[np.isfinite(matrix)].size
            ]
        ) if any(np.isfinite(matrices[(metric, size)]).any() for size in sizes) else np.empty(0)
        vmin = float(np.min(finite_values)) if finite_values.size else 0.0
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        image = None
        for size_index, size in enumerate(sizes):
            axis = axes[metric_index, size_index]
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad(color="#d3d3d3")
            image = axis.imshow(
                matrices[(metric, size)],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if metric_index == 0:
                axis.set_title(f"N = {size:,}", fontsize=14)
            if size_index == 0:
                axis.set_ylabel(f"{label}\n$\\lambda_1$", fontsize=12)
            else:
                axis.set_ylabel("$\\lambda_1$")
            if metric_index == len(METRICS) - 1:
                axis.set_xlabel("$\\lambda_{TV}$")
            axis.set_xticks(range(n_tv))
            axis.set_xticklabels(
                [f"{value:g}" for value in config.grid.lambda_tv],
                rotation=90,
                fontsize=7,
            )
            axis.set_yticks(range(n_l1))
            axis.set_yticklabels(
                [f"{value:g}" for value in config.grid.lambda_l1], fontsize=7
            )
        if image is not None:
            figure.colorbar(
                image,
                ax=list(axes[metric_index, :]),
                shrink=0.88,
                location="right",
                label=label,
            )

    target = config.output_path / "figures"
    target.mkdir(parents=True, exist_ok=True)
    png = target / "lambda_grid_four_maps.png"
    pdf = target / "lambda_grid_four_maps.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    print(f"Wrote figure: {png}")
    print(f"Wrote figure: {pdf}")
    return {"png": png, "pdf": pdf}
