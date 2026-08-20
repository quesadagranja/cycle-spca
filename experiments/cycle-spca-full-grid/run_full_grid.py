#!/usr/bin/env python3
"""Entry point that pins every numerical backend to one thread per worker."""

from __future__ import annotations

import os


for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

from cycle_grid.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
