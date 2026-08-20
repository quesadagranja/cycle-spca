from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from cycle_grid.config import all_tasks, load_config
from cycle_grid.database import export_fits_csv, sync_fit_records
from cycle_grid.experiment import initialize_experiment
from cycle_grid.fit_worker import (
    initialize_fit_worker,
    run_fit_task,
    tensor_to_heatmap,
)


class CalendarMappingTests(unittest.TestCase):
    def test_heatmap_column_is_week_times_days_plus_day(self):
        tensor = np.zeros((4, 3, 5))
        for hour in range(4):
            for day in range(3):
                for week in range(5):
                    tensor[hour, day, week] = 100 * hour + 10 * day + week
        heatmap = tensor_to_heatmap(tensor)
        self.assertEqual(heatmap.shape, (4, 15))
        for hour in range(4):
            for day in range(3):
                for week in range(5):
                    self.assertEqual(
                        heatmap[hour, week * 3 + day], tensor[hour, day, week]
                    )


class SmokeTest(unittest.TestCase):
    def test_one_fit_produces_queryable_outputs(self):
        repo_value = os.environ.get("CYCLE_SPCA_REPO")
        if not repo_value:
            self.skipTest("Set CYCLE_SPCA_REPO to run the end-to-end smoke test.")
        repo = Path(repo_value).resolve()
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        with tempfile.TemporaryDirectory() as temporary_string:
            temporary = Path(temporary_string)
            rng = np.random.default_rng(10)
            data = np.zeros((40, 63), dtype=np.float64)
            data[:, 0] = np.arange(40)
            data[:, 1] = 2020
            data[:, 2] = 0
            data[-5:, 2] = 100
            data[:, 3:] = rng.normal(size=(40, 60))
            dataset = temporary / "dataset.npy"
            np.save(dataset, data)
            output = temporary / "results"
            config_path = temporary / "grid.json"
            config = {
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
                "lambda_tv_values": [0.0],
                "K_values": [2],
                "N_values": [20],
                "sample_seeds": [123],
                "initialization_seeds": [456],
                "model": {
                    "outer_max_iter": 2,
                    "inner_max_iter": 50,
                    "inner_check_interval": 5,
                    "batch_size": 8,
                },
                "png": {
                    "enabled": True,
                    "dpi": 50,
                    "figsize": [5, 2.5],
                    "palette_colors": 64,
                },
                "runtime": {"workers": 1},
                "stability": {"enabled": False},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_config(config_path)
            initialized = initialize_experiment(loaded)
            tasks = all_tasks(loaded, initialized["samples"])
            self.assertEqual(len(tasks), 1)
            initialize_fit_worker(loaded)
            result = run_fit_task(tasks[0])
            self.assertEqual(result["status"], "done", result)
            sync_fit_records(output)
            self.assertEqual(export_fits_csv(output), 1)
            fit_dir = output / tasks[0]["fit_relative_dir"]
            self.assertTrue((fit_dir / "DONE").exists())
            self.assertTrue((fit_dir / "component_png" / "component_01.png").exists())
            self.assertTrue((output / "tables" / "fits.csv").exists())


if __name__ == "__main__":
    unittest.main()
