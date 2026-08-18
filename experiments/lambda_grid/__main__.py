"""Command-line interface for the lambda-grid experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .aggregate import aggregate_results
from .config import DEFAULT_CONFIG_PATH, load_config
from .executor import run_from_manifest
from .manifests import write_manifests
from .plotting import plot_four_maps
from .sampling import prepare_samples
from .status import collect_status, print_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cycle-spca-lambda-grid",
        description="Reproducible CycleSPCA lambda-grid experiment.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Experiment JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare",
        help="Filter imputed<=72 and create the nested random master sample.",
    )
    subparsers.add_parser(
        "make-manifests",
        help="Write pilot, restart, and complete task manifests.",
    )
    run_parser = subparsers.add_parser("run", help="Execute one manifest task.")
    run_parser.add_argument(
        "--manifest",
        default="pilot",
        help="pilot, restarts, all, or an explicit CSV path.",
    )
    run_parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Zero-based manifest task ID; defaults to SLURM_ARRAY_TASK_ID.",
    )
    status_parser = subparsers.add_parser("status", help="Summarize run progress.")
    status_parser.add_argument(
        "--manifest",
        default="all",
        help="pilot, restarts, all, or an explicit CSV path.",
    )
    subparsers.add_parser(
        "aggregate",
        help="Build run, cell, and between-seed stability tables.",
    )
    subparsers.add_parser("plot", help="Create the four-by-four heatmap figure.")
    subparsers.add_parser("show-config", help="Validate and print resolved config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command == "prepare":
        prepare_samples(config)
    elif arguments.command == "make-manifests":
        write_manifests(config)
    elif arguments.command == "run":
        task_id = arguments.task_id
        if task_id is None:
            slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_task is None:
                raise SystemExit(
                    "--task-id is required outside a Slurm array task."
                )
            task_id = int(slurm_task)
        run_from_manifest(config, arguments.manifest, task_id)
    elif arguments.command == "status":
        print_status(collect_status(config, arguments.manifest))
    elif arguments.command == "aggregate":
        aggregate_results(config)
    elif arguments.command == "plot":
        plot_four_maps(config)
    elif arguments.command == "show-config":
        print(json.dumps(config.as_dict(), indent=2, sort_keys=True))
        print(f"config_sha256={config.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
