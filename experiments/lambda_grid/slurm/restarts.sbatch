#!/usr/bin/env bash
#SBATCH --job-name=cspca-lambda-restarts
#SBATCH --array=0-3599%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cquesada/pca/results/lambda_grid_v1/slurm/restarts_%A_%a.out
#SBATCH --error=/home/cquesada/pca/results/lambda_grid_v1/slurm/restarts_%A_%a.err

set -euo pipefail

PROJECT_DIR="${CYCLE_SPCA_REPO:-/home/cquesada/pca}"
PYTHON_BIN="${CYCLE_SPCA_PYTHON:-python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m experiments.lambda_grid run \
  --manifest restarts \
  --task-id "${SLURM_ARRAY_TASK_ID}"
