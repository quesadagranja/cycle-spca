from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from cycle_grid.cli import run_grid
from cycle_grid.config import load_config
from cycle_grid.database import database_counts


class OrchestrationTest(unittest.TestCase):
    def test_parallel_run_live_tables_and_stability(self):
        repo_value = os.environ.get("CYCLE_SPCA_REPO")
        if not repo_value:
            self.skipTest("Set CYCLE_SPCA_REPO to run the orchestration test.")
        repo = Path(repo_value).resolve()
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        with tempfile.TemporaryDirectory() as temporary_string:
            temporary = Path(temporary_string)
            rng = np.random.default_rng(33)
            data = np.zeros((35, 63), dtype=np.float64)
            data[:, 0] = np.arange(35)
            data[:, 1] = 2021
            data[:, 2] = 0
            data[-3:, 2] = 90
            data[:, 3:] = rng.normal(size=(35, 60))
            dataset = temporary / "dataset.npy"
            np.save(dataset, data)
            output = temporary / "results"
            config_path = temporary / "grid.json"
            raw = {
                "cycle_spca_repo": str(repo),
                "expected_cycle_spca_commit": commit,
                "dataset_path": str(dataset),
                "output_dir": str(output),
                "feature_start": 3,
                "feature_count": 60,
                "imputed_column": 2,
                "max_imputed": 72,
                "calendar_shape": [4, 3, 5],
                "order": "F",
                "lambda_l1_values": [0.0],
                "lambda_tv_values": [0.0, 0.05],
                "K_values": [2],
                "N_values": [20],
                "sample_seeds": [101, 202],
                "initialization_seeds": [303, 404],
                "model": {
                    "outer_max_iter": 1,
                    "inner_max_iter": 25,
                    "inner_check_interval": 5,
                    "batch_size": 8
                },
                "png": {"enabled": False},
                "runtime": {
                    "workers": 2,
                    "start_method": "fork",
                    "refresh_seconds": 5,
                    "plot_refresh_seconds": 60,
                    "components_export_seconds": 60,
                    "maxtasksperchild": 10
                },
                "stability": {"enabled": True, "workers": 2}
            }
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(run_grid(config, workers=2, force_lock=False), 0)
            counts = database_counts(output)
            self.assertEqual(counts["fits"], 4)
            self.assertEqual(counts["local"], 2)
            self.assertEqual(counts["repeat"], 2)
            self.assertTrue((output / "tables" / "fits.csv").exists())
            self.assertTrue((output / "tables" / "components.csv.gz").exists())
            self.assertTrue((output / "dashboard.html").exists())


if __name__ == "__main__":
    unittest.main()
