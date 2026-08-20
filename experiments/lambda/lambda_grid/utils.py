"""Small I/O and serialization utilities shared by experiment commands."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, TextIO

import numpy as np

from .config import ExperimentConfig


LAYOUT_DIRECTORIES = (
    "config",
    "samples",
    "manifests",
    "runs",
    "aggregated",
    "figures",
    "slurm",
)


def ensure_layout(config: ExperimentConfig) -> None:
    for relative in LAYOUT_DIRECTORIES:
        (config.output_path / relative).mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    converted = to_jsonable(value)
    if converted is None:
        return ""
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, separators=(",", ":"), ensure_ascii=False)
    return converted


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def snapshot_config(config: ExperimentConfig) -> Path:
    ensure_layout(config)
    target = config.output_path / "config" / "experiment.json"
    payload = config.as_dict() | {"config_sha256": config.sha256}
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_sha256") != config.sha256:
            raise RuntimeError(
                f"{target} belongs to a different configuration. Use a new "
                "output_dir for a changed experiment."
            )
    else:
        atomic_write_json(target, payload)
    return target


class Tee(TextIO):
    """Mirror text to the terminal and a persistent run log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        terminal_result = self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return terminal_result

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False


def tee_stdout(log_path: Path):
    """Context manager that mirrors both stdout and stderr."""

    class _TeeContext:
        def __enter__(self):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = log_path.open("a", encoding="utf-8", buffering=1)
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            sys.stdout = Tee(self.old_stdout, self.handle)
            sys.stderr = Tee(self.old_stderr, self.handle)
            return self.handle

        def __exit__(self, exc_type, exc_value, traceback):
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            self.handle.close()
            return False

    return _TeeContext()
