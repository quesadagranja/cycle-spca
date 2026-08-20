from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .database import load_fit_records_enriched
from .io_utils import atomic_write_text, utc_now


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return escape(str(value))


def _aggregate_matrix(
    records: list[dict[str, Any]],
    n_samples: int,
    k_value: int,
    n_l1: int,
    n_ltv: int,
    getter: Callable[[dict[str, Any]], float | None],
) -> np.ndarray:
    sums = np.zeros((n_l1, n_ltv), dtype=np.float64)
    counts = np.zeros((n_l1, n_ltv), dtype=np.int64)
    for record in records:
        if int(record["N"]) != n_samples or int(record["K"]) != k_value:
            continue
        value = getter(record)
        if value is None or not np.isfinite(value):
            continue
        row = int(record["l1_index"])
        column = int(record["ltv_index"])
        sums[row, column] += float(value)
        counts[row, column] += 1
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )


def generate_heatmaps(
    output: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]] | None = None,
) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if records is None:
        records = load_fit_records_enriched(output)
    if not records:
        return 0
    l1_values = list(map(float, config["lambda_l1_values"]))
    ltv_values = list(map(float, config["lambda_tv_values"]))
    tick_l1 = sorted(set([0, len(l1_values) - 1, *range(0, len(l1_values), 3)]))
    tick_ltv = sorted(set([0, len(ltv_values) - 1, *range(0, len(ltv_values), 3)]))
    metrics = [
        ("Explained variance (%)", lambda r: r.get("explained_variance_percent"), "viridis", 0, None),
        ("Sparsity (fraction)", lambda r: r.get("mean_sparsity_active"), "magma", 0, 1),
        ("Relative total variation", lambda r: r.get("mean_relative_tv_active"), "cividis", 0, None),
        ("Convergence rate", lambda r: float(bool(r.get("converged"))), "Greens", 0, 1),
        ("Local stability", lambda r: r.get("local_stability_mean"), "plasma", 0, 1),
        ("Repeat stability", lambda r: r.get("repeat_stability_mean"), "plasma", 0, 1),
    ]
    count = 0
    heatmap_dir = output / "dashboard" / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for n_samples in config["N_values"]:
        for k_value in config["K_values"]:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
            for ax, (title, getter, cmap, vmin, vmax) in zip(axes.flat, metrics):
                matrix = _aggregate_matrix(
                    records,
                    int(n_samples),
                    int(k_value),
                    len(l1_values),
                    len(ltv_values),
                    getter,
                )
                masked = np.ma.masked_invalid(matrix)
                colormap = plt.get_cmap(cmap).copy()
                colormap.set_bad("#d9d9d9")
                artist = ax.imshow(
                    masked,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(title)
                ax.set_xlabel(r"$\lambda_{TV}$")
                ax.set_ylabel(r"$\lambda_1$")
                ax.set_xticks(tick_ltv)
                ax.set_xticklabels([f"{ltv_values[i]:g}" for i in tick_ltv], rotation=45)
                ax.set_yticks(tick_l1)
                ax.set_yticklabels([f"{l1_values[i]:g}" for i in tick_l1])
                fig.colorbar(artist, ax=ax, shrink=0.83)
            fig.suptitle(
                f"CycleSPCA grid · N={int(n_samples):,} · K={int(k_value)} · "
                "mean over completed repeats",
                fontsize=15,
            )
            path = heatmap_dir / f"N_{int(n_samples):06d}_K_{int(k_value):02d}.png"
            fig.savefig(path, dpi=130, facecolor="white")
            plt.close(fig)
            count += 1
    return count


def write_dashboard(
    output: Path,
    config: dict[str, Any],
    *,
    total: int,
    include_plots: bool = False,
) -> None:
    records = load_fit_records_enriched(output)
    if include_plots:
        generate_heatmaps(output, config, records)
    completed = len(records)
    converged = sum(bool(record.get("converged")) for record in records)
    failures = len(list((output / "failures").glob("*.json")))
    running = len(list((output / "running").glob("*.json")))
    percent = 100.0 * completed / max(total, 1)
    elapsed_values = [
        float(record["elapsed_seconds"])
        for record in records
        if record.get("elapsed_seconds") is not None
    ]
    mean_elapsed = float(np.mean(elapsed_values)) if elapsed_values else None
    recent = sorted(records, key=lambda item: item.get("finished_at", ""), reverse=True)[:50]

    rows = []
    for record in recent:
        image_link = (
            f"{record['fit_relative_dir']}/component_png/component_01.png"
        )
        rows.append(
            "<tr>"
            f"<td><a href='{escape(image_link)}'>{escape(record['fit_id'])}</a></td>"
            f"<td>{record['repeat']}</td><td>{record['N']}</td><td>{record['K']}</td>"
            f"<td>{record['lambda_l1']:g}</td><td>{record['lambda_tv']:g}</td>"
            f"<td>{_format_number(record.get('explained_variance_percent'))}</td>"
            f"<td>{_format_number(record.get('mean_sparsity_active'))}</td>"
            f"<td>{_format_number(record.get('mean_relative_tv_active'))}</td>"
            f"<td>{'✓' if record.get('converged') else '—'}</td>"
            "</tr>"
        )
    heatmap_links = []
    for n_samples in config["N_values"]:
        links = " ".join(
            f"<a href='dashboard/heatmaps/N_{int(n_samples):06d}_K_{int(k):02d}.png'>K={int(k)}</a>"
            for k in config["K_values"]
            if (output / "dashboard" / "heatmaps" / f"N_{int(n_samples):06d}_K_{int(k):02d}.png").exists()
        )
        heatmap_links.append(f"<li>N={int(n_samples):,}: {links or 'pending'}</li>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="60">
  <title>CycleSPCA full grid</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #202124; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .card {{ border: 1px solid #ddd; border-radius: .5rem; padding: .8rem 1rem; min-width: 10rem; }}
    .big {{ font-size: 1.6rem; font-weight: 650; }}
    progress {{ width: 100%; height: 1.4rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: .35rem .45rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    a {{ color: #1259a7; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>CycleSPCA full-grid experiment</h1>
  <p class="muted">Updated {escape(utc_now())}. This page refreshes every 60 seconds.</p>
  <progress max="{total}" value="{completed}"></progress>
  <div class="cards">
    <div class="card"><div class="big">{completed:,} / {total:,}</div>completed fits ({percent:.2f}%)</div>
    <div class="card"><div class="big">{running}</div>currently running</div>
    <div class="card"><div class="big">{converged:,}</div>converged ({100*converged/max(completed,1):.1f}%)</div>
    <div class="card"><div class="big">{failures:,}</div>failed attempts</div>
    <div class="card"><div class="big">{_format_number(mean_elapsed, 3)} s</div>mean fit time</div>
  </div>

  <h2>Current tables</h2>
  <p><a href="tables/fits.csv">fits.csv</a> ·
     <a href="tables/components.csv.gz">components.csv.gz</a> ·
     <a href="tables/stability_local.csv">local stability</a> ·
     <a href="tables/stability_repeat.csv">repeat stability</a> ·
     <a href="tables/failures.csv">failures</a> ·
     <a href="tables/results.sqlite">read-only SQLite database</a></p>

  <h2>Aggregate heatmaps</h2>
  <p>Gray cells have not finished yet. Stability panels appear during the final
  post-processing phase.</p>
  <ul>{''.join(heatmap_links)}</ul>

  <h2>Most recently completed fits</h2>
  <p>Click a fit identifier to open its first component PNG; the neighboring
  component files are in the same directory.</p>
  <table>
    <thead><tr><th>fit</th><th>repeat</th><th>N</th><th>K</th><th>λ1</th><th>λTV</th>
    <th>EV (%)</th><th>sparsity</th><th>relative TV</th><th>converged</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    atomic_write_text(output / "dashboard.html", html)
