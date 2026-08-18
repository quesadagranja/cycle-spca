# Lambda-grid experiment

This module runs the first empirical CycleSPCA hyperparameter study with fixed
rank `K=10` and no sample-size scaling of either lambda.

## Frozen design

- Input: `/home/cquesada/pca/dataset/matrix_normalized.npy`.
- Eligibility: the one-based `imputed` column 3 must satisfy `imputed <= 72`.
- Features: one-based columns 4 through 8739, both inclusive (8,736 values).
- Sampling: one simple random sample without replacement of 20,000 eligible
  rows using seed `20260809`.
- Nested samples: the first 5k, 10k, 15k, and 20k rows of that one shuffled
  master sample.
- No stratification or inspection of `iso_year` is performed.
- No additional NaN, infinity, or value-range scan is performed by sample
  preparation.
- Grid: 15 values for `lambda_l1` by 15 values for `lambda_tv`.
- Algorithm seeds: 11, 23, 47, 71, and 101.
- Iteration ceilings: 50 score sweeps, 1,000 outer iterations, and 20,000
  loading-solver iterations, all with early stopping.

The complete grid contains `4 * 15 * 15 * 5 = 4,500` independent fits. The
pilot is the first seed only and contains 900 fits. The other four seeds
contain 3,600 fits.

## Installation

From `/home/cquesada/pca`:

```bash
python -m pip install -e '.[experiments]'
```

The experiment configuration is
`experiments/lambda_grid/experiment.json`. Its output directory is
`/home/cquesada/pca/results/lambda_grid_v1`. A resolved, hashed copy of the
configuration is frozen under the output directory when preparation starts.
Changing the scientific design should use a new output directory.

## Prepare the sample and manifests

```bash
python -m experiments.lambda_grid show-config
python -m experiments.lambda_grid prepare
python -m experiments.lambda_grid make-manifests
```

Preparation writes one `master_20000.npy`; the four analyses use read-only
prefix views, so the original multi-column dataset is not re-extracted for
each fit.

## Run one job

Task IDs are zero-based:

```bash
python -m experiments.lambda_grid run --manifest pilot --task-id 137
```

Each job has an independent directory under `runs/<run_id>/`. Standard output
is mirrored to `run.log`. A successful run contains:

- `summary.json`: scalar metrics, limits, versions, time, and memory;
- `history.csv.gz`: complete flattened optimization history;
- `diagnostics.json`: estimator diagnostics;
- `audit.json`: independent metric audit;
- `components.npz`: loadings, active mask, and reinitialization counts;
- `_SUCCESS`: atomic completion marker.

Scores are deliberately omitted. Jobs never append to a common results file.

## Slurm arrays

Templates are supplied under `experiments/lambda_grid/slurm/`. Their default
concurrency is 20 jobs; adjust `%20`, memory, time, and environment activation
to the cluster policy before submission.

```bash
sbatch experiments/lambda_grid/slurm/pilot.sbatch
```

After validating the pilot coverage and convergence behavior:

```bash
sbatch experiments/lambda_grid/slurm/restarts.sbatch
```

Set `CYCLE_SPCA_PYTHON` to an absolute environment Python executable when
`python` is not the intended interpreter.

## Inspect and aggregate

```bash
python -m experiments.lambda_grid status --manifest pilot
python -m experiments.lambda_grid status --manifest all
python -m experiments.lambda_grid aggregate
python -m experiments.lambda_grid plot
```

Aggregation creates CSV and Parquet tables with one row per planned run, one
row per lambda cell and sample size, and one row per seed pair. Component
stability uses Hungarian matching and absolute cosine similarity. The plot
contains four rows (explained variance, loading sparsity, global relative TV,
and between-seed stability) and four columns (5k, 10k, 15k, and 20k).

Cells with a completed run that failed convergence, audit, or effective-rank
checks are gray. During the first-seed pilot, the stability row remains gray
because at least two seeds are required.
