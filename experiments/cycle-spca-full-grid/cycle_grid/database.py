from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from .fit_worker import COMPONENT_FIELDS
from .io_utils import atomic_write_csv, read_json


FIT_EXPORT_FIELDS = [
    "fit_id",
    "repeat",
    "N",
    "K",
    "K_effective",
    "l1_index",
    "ltv_index",
    "lambda_l1",
    "lambda_tv",
    "sample_seed",
    "initialization_seed",
    "sample_master_hash",
    "sample_prefix_hash",
    "explained_variance",
    "explained_variance_percent",
    "mean_sparsity_active",
    "median_sparsity_active",
    "mean_relative_tv_active",
    "median_relative_tv_active",
    "mean_connected_regions_active",
    "mean_effective_regions_active",
    "local_stability_mean",
    "repeat_stability_mean",
    "condition_number_v_gram",
    "converged",
    "n_outer_iterations",
    "final_objective",
    "final_reconstruction_error",
    "final_relative_objective_change",
    "final_relative_reconstruction_change",
    "final_inner_converged",
    "final_inner_iterations",
    "final_inner_relative_change",
    "final_inner_primal_dual_residual",
    "all_inner_converged",
    "mean_inner_iterations",
    "max_inner_iterations",
    "total_reinitializations",
    "elapsed_seconds",
    "worker_peak_rss_mb",
    "started_at",
    "finished_at",
    "component_order",
    "fit_relative_dir",
    "loadings_relative_path",
]


STABILITY_FIELDS = [
    "pair_id",
    "kind",
    "direction",
    "fit_a",
    "fit_b",
    "repeat_a",
    "repeat_b",
    "N",
    "K",
    "l1_index_a",
    "l1_index_b",
    "ltv_index_a",
    "ltv_index_b",
    "lambda_l1_a",
    "lambda_l1_b",
    "lambda_tv_a",
    "lambda_tv_b",
    "K_effective_a",
    "K_effective_b",
    "matched_active_components",
    "mean_matched_active_cosine",
    "penalized_similarity",
    "computed_at",
]


def connect(output: Path, *, readonly: bool = False) -> sqlite3.Connection:
    path = output / "tables" / "results.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    if readonly and path.exists():
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    else:
        connection = sqlite3.connect(path, timeout=60)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def initialize_database(output: Path) -> None:
    with connect(output) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS fits (
                fit_id TEXT PRIMARY KEY,
                repeat INTEGER NOT NULL,
                N INTEGER NOT NULL,
                K INTEGER NOT NULL,
                l1_index INTEGER NOT NULL,
                ltv_index INTEGER NOT NULL,
                lambda_l1 REAL NOT NULL,
                lambda_tv REAL NOT NULL,
                explained_variance REAL,
                mean_sparsity_active REAL,
                mean_relative_tv_active REAL,
                converged INTEGER NOT NULL,
                finished_at TEXT,
                fit_relative_dir TEXT NOT NULL,
                loadings_relative_path TEXT NOT NULL,
                fit_json TEXT NOT NULL,
                components_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS fits_grid
                ON fits(repeat, N, K, l1_index, ltv_index);
            CREATE TABLE IF NOT EXISTS stability (
                pair_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                fit_a TEXT NOT NULL,
                fit_b TEXT NOT NULL,
                similarity REAL NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS stability_kind_a ON stability(kind, fit_a);
            CREATE INDEX IF NOT EXISTS stability_kind_b ON stability(kind, fit_b);
            """
        )


def _insert_fit_record(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    fit = record["fit"]
    connection.execute(
        """
        INSERT OR REPLACE INTO fits (
            fit_id, repeat, N, K, l1_index, ltv_index,
            lambda_l1, lambda_tv, explained_variance,
            mean_sparsity_active, mean_relative_tv_active, converged,
            finished_at, fit_relative_dir, loadings_relative_path,
            fit_json, components_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fit["fit_id"],
            int(fit["repeat"]),
            int(fit["N"]),
            int(fit["K"]),
            int(fit["l1_index"]),
            int(fit["ltv_index"]),
            float(fit["lambda_l1"]),
            float(fit["lambda_tv"]),
            fit.get("explained_variance"),
            fit.get("mean_sparsity_active"),
            fit.get("mean_relative_tv_active"),
            int(bool(fit.get("converged"))),
            fit.get("finished_at"),
            record["fit_relative_dir"],
            record["loadings_relative_path"],
            json.dumps(fit, separators=(",", ":")),
            json.dumps(record["components"], separators=(",", ":")),
        ),
    )


def sync_fit_records(output: Path) -> int:
    initialize_database(output)
    with connect(output) as connection:
        existing = {row[0] for row in connection.execute("SELECT fit_id FROM fits")}
        paths = sorted((output / "records").glob("*.json"))
        new_paths = [path for path in paths if path.stem not in existing]
        for start in range(0, len(new_paths), 250):
            with connection:
                for path in new_paths[start : start + 250]:
                    _insert_fit_record(connection, read_json(path))
        return len(new_paths)


def sync_stability_records(output: Path) -> int:
    initialize_database(output)
    with connect(output) as connection:
        existing = {row[0] for row in connection.execute("SELECT pair_id FROM stability")}
        paths = sorted((output / "stability_records").glob("*.json"))
        new_paths = [path for path in paths if path.stem not in existing]
        for start in range(0, len(new_paths), 500):
            with connection:
                for path in new_paths[start : start + 500]:
                    record = read_json(path)
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO stability
                        (pair_id, kind, fit_a, fit_b, similarity, record_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["pair_id"],
                            record["kind"],
                            record["fit_a"],
                            record["fit_b"],
                            float(record["penalized_similarity"]),
                            json.dumps(record, separators=(",", ":")),
                        ),
                    )
        return len(new_paths)


def load_fit_records(output: Path) -> list[dict[str, Any]]:
    initialize_database(output)
    with connect(output, readonly=True) as connection:
        rows = connection.execute(
            "SELECT fit_json, fit_relative_dir, loadings_relative_path "
            "FROM fits ORDER BY repeat, N, K, l1_index, ltv_index"
        ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        fit = json.loads(row["fit_json"])
        fit["fit_relative_dir"] = row["fit_relative_dir"]
        fit["loadings_relative_path"] = row["loadings_relative_path"]
        records.append(fit)
    return records


def _stability_means(output: Path) -> dict[str, dict[str, float]]:
    accumulator: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with connect(output, readonly=True) as connection:
        for row in connection.execute("SELECT kind, fit_a, fit_b, similarity FROM stability"):
            accumulator[row["fit_a"]][row["kind"]].append(float(row["similarity"]))
            accumulator[row["fit_b"]][row["kind"]].append(float(row["similarity"]))
    result: dict[str, dict[str, float]] = {}
    for fit_id, kinds in accumulator.items():
        result[fit_id] = {
            kind: sum(values) / len(values) for kind, values in kinds.items() if values
        }
    return result


def load_fit_records_enriched(output: Path) -> list[dict[str, Any]]:
    records = load_fit_records(output)
    means = _stability_means(output)
    for record in records:
        values = means.get(record["fit_id"], {})
        record["local_stability_mean"] = values.get("local")
        record["repeat_stability_mean"] = values.get("repeat")
    return records


def export_fits_csv(output: Path) -> int:
    records = load_fit_records_enriched(output)
    atomic_write_csv(output / "tables" / "fits.csv", FIT_EXPORT_FIELDS, records)
    return len(records)


def export_components_csv(output: Path) -> int:
    initialize_database(output)

    def rows():
        with connect(output, readonly=True) as connection:
            cursor = connection.execute(
                "SELECT components_json FROM fits "
                "ORDER BY repeat, N, K, l1_index, ltv_index"
            )
            for database_row in cursor:
                yield from json.loads(database_row["components_json"])

    count = 0

    def counted():
        nonlocal count
        for row in rows():
            count += 1
            yield row

    atomic_write_csv(
        output / "tables" / "components.csv.gz",
        COMPONENT_FIELDS,
        counted(),
        gzip_output=True,
    )
    return count


def export_stability_csv(output: Path) -> int:
    initialize_database(output)
    by_kind: dict[str, list[dict[str, Any]]] = {"local": [], "repeat": []}
    with connect(output, readonly=True) as connection:
        for row in connection.execute(
            "SELECT record_json FROM stability ORDER BY kind, fit_a, fit_b"
        ):
            record = json.loads(row["record_json"])
            by_kind[record["kind"]].append(record)
    for kind, rows in by_kind.items():
        atomic_write_csv(
            output / "tables" / f"stability_{kind}.csv",
            STABILITY_FIELDS,
            rows,
        )
    return sum(map(len, by_kind.values()))


def export_failures_csv(output: Path) -> int:
    paths = sorted((output / "failures").glob("*.json"))
    fields = [
        "fit_id",
        "failed_at",
        "worker_pid",
        "exception_type",
        "exception_message",
        "failure_record",
    ]

    def rows():
        for path in paths:
            record = read_json(path)
            yield {
                **record,
                "failure_record": str(path.relative_to(output)),
            }

    atomic_write_csv(output / "tables" / "failures.csv", fields, rows())
    return len(paths)


def database_counts(output: Path) -> dict[str, int]:
    initialize_database(output)
    with connect(output, readonly=True) as connection:
        fits = int(connection.execute("SELECT COUNT(*) FROM fits").fetchone()[0])
        converged = int(
            connection.execute("SELECT COUNT(*) FROM fits WHERE converged = 1").fetchone()[0]
        )
        local = int(
            connection.execute(
                "SELECT COUNT(*) FROM stability WHERE kind = 'local'"
            ).fetchone()[0]
        )
        repeat = int(
            connection.execute(
                "SELECT COUNT(*) FROM stability WHERE kind = 'repeat'"
            ).fetchone()[0]
        )
    return {"fits": fits, "converged": converged, "local": local, "repeat": repeat}
