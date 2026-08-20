from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _temp_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    return Path(name)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = _temp_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    if pretty:
        text = json.dumps(jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
    else:
        text = json.dumps(
            jsonable(value), separators=(",", ":"), sort_keys=True, ensure_ascii=False
        )
    atomic_write_text(path, text + "\n")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = _temp_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    gzip_output: bool | None = None,
) -> None:
    if gzip_output is None:
        gzip_output = path.suffix == ".gz"
    temporary = _temp_path(path)
    try:
        if gzip_output:
            handle_context = gzip.open(temporary, "wt", encoding="utf-8", newline="")
        else:
            handle_context = temporary.open("w", encoding="utf-8", newline="")
        with handle_context as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: jsonable(v) for k, v in row.items()})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def relative_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))
