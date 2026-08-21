# Full factorial Calendar-SPCA experiment

This package runs the large-scale factorial experiment used to evaluate
Calendar-SPCA on the GoiEner electricity-consumption dataset.

The experiment invokes the implementation directly from the
`quesadagranja/cycle-spca` repository. The optimizer is neither copied nor
modified by the experimental harness, and the exact source-code commit used for
the experiment is recorded and verified before execution.

## Experimental design

The production configuration evaluates:

- 7 values of `lambda_l1`:
  `0, 0.05, 0.2, 0.5, 1, 3, 10`;
- 7 values of `lambda_tv`:
  `0, 0.05, 0.2, 0.5, 1, 3, 10`;
- nominal component counts
  `K = 5, 10, 15, 20, 25, 30`;
- sample sizes
  `N = 5,000, 10,000, 15,000, 20,000`;
- five reproducible repetitions;
- mandatory filtering with `imputed <= 72`;
- nested samples within each repetition;
- centered data;
- calendar shape `(24, 7, 52)`;
- `outer_max_iter = 500`;
- `inner_max_iter = 50,000`;
- one PNG per nominal component;
- 90 worker processes, with numerical backends restricted to one thread per
  worker.

The complete design therefore contains

```text
7 × 7 × 6 × 4 × 5 = 5,880 fits
````

and, with PNG generation enabled, produces

```text
102,900 component maps
```

when all fits are completed.

## Current production configuration

The default paths in `grid.json` are:

```text
Calendar-SPCA repository:
    /home/cquesada/cycle-spca

Dataset:
    /home/cquesada/pca/dataset/matrix_normalized.npy

Results:
    /home/cquesada/cycle-spca-full-grid/results
```

The experiment is pinned to the Calendar-SPCA source commit

```text
2ea2317fef82efcadefbdaa4a22d709df373ba00
```

and requires tracked repository files to remain clean. This prevents changes
to the scientific implementation while an experiment is in progress.

## Dataset layout

The input NPY matrix contains three metadata columns followed by the annual
electricity-consumption profile:

```text
column 0      identifier
column 1      ISO year
column 2      number of imputed hourly values
columns 3:    8,736 normalized hourly features
```

Only rows satisfying

```text
imputed <= 72
```

are eligible for sampling.

The 8,736 features correspond to

```text
24 hours × 7 weekdays × 52 ISO weeks.
```

The experiment uses

```text
calendar_shape = [24, 7, 52]
order = "F"
center = true
```

so that the flattened feature ordering matches the chronological organization
of the source dataset.

## Validation and preparation

From the experiment directory:

```bash
cd experiments/cycle-spca-full-grid
python -m pip install -r requirements.txt
python run_full_grid.py --config grid.json validate
```

The dataset, repository state, configuration, and requested dimensions are
checked before execution.

Samples and experiment metadata can be prepared without launching the fits:

```bash
python run_full_grid.py --config grid.json prepare
```

Preparation records:

* the SHA-256 hash of the dataset;
* the exact Calendar-SPCA Git commit;
* the scientific configuration hash;
* software and platform information;
* the sampled row indices;
* sample hashes;
* dataset metadata.

If the dataset, scientific configuration, or Calendar-SPCA commit changes, the
existing results directory cannot silently be reused.

## Repeated and nested sampling

Five independent master samples are generated using

```text
11011
22022
33033
44044
55055
```

as sampling seeds.

For each repetition, a random permutation of all eligible observations is
generated and its first 20,000 observations form the master sample.

The smaller samples are nested prefixes:

```text
N = 5,000
    ⊂ 10,000
    ⊂ 15,000
    ⊂ 20,000
```

within the same repetition.

Consequently, comparisons across sample sizes preserve the observations from
the smaller sample while progressively adding new observations.

The master indices and their metadata are stored under:

```text
results/samples/
```

## Initialization control

The five repetitions use the initialization seed bases

```text
111
222
333
444
555
```

For a fixed `(repeat, N, K)`, all 49 combinations of
`(lambda_l1, lambda_tv)` receive exactly the same randomized-SVD
initialization seed.

This design isolates the effect of the regularization parameters from
differences caused by randomized initialization when comparing positions
within the lambda grid.

Initialization seeds vary across sample sizes, nominal ranks, and repetitions.

## Running the experiment

The provided launcher starts 90 workers:

```bash
bash launch_90_workers.sh
```

For a long cluster execution, it can be launched inside `tmux`:

```bash
tmux new -s calendar-spca-grid
cd experiments/cycle-spca-full-grid
bash launch_90_workers.sh
```

Detach with:

```text
Ctrl-b
d
```

The experiment is resumable. Fits whose output directory already contains a
valid `DONE` marker are skipped.

## Parallel execution

`run_full_grid.py` sets the following numerical backends to one thread before
starting the workers:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
BLIS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

This avoids nested BLAS/OpenMP parallelism when using many independent worker
processes.

The source dataset is opened with NumPy memory mapping. Workers materialize
only the selected row batches required by Calendar-SPCA, while the underlying
NPY file remains memory-mapped and can be shared through the operating-system
page cache.

## Fit organization

Each combination receives an exclusive directory of the form

```text
fits/
└── repeat_01/
    └── N_020000/
        └── K_15/
            └── l1_04/
                └── ltv_04/
                    ├── config.json
                    ├── metrics.json
                    ├── components.csv
                    ├── history.csv.gz
                    ├── loadings.npz
                    ├── component_png/
                    │   ├── component_01.png
                    │   ├── component_02.png
                    │   └── ...
                    └── DONE
```

Workers first construct complete fit directories under `tmp/` and publish them
through an atomic rename only after all required files have been written.

Workers do not write directly to the central CSV files or to directories owned
by other fits.

## Recorded fit-level metrics

Each fit records, among other quantities:

* nominal and effective component count;
* explained variance;
* mean and median loading sparsity;
* mean and median relative graph total variation;
* mean number of connected regions;
* mean number of effective regions;
* loading Gram-matrix condition number;
* convergence status;
* number of outer iterations;
* final objective and reconstruction error;
* inner primal-dual convergence diagnostics;
* component reinitializations;
* elapsed execution time;
* peak worker memory usage.

Detailed histories are stored separately for every outer iteration.

## Recorded component-level metrics

For every nominal component, the experiment records:

* active/inactive status;
* conditional reconstruction contribution;
* loading sparsity;
* loading L1 and L2 norms;
* graph total variation;
* relative total variation;
* number of active calendar cells;
* connected regions;
* effective regions;
* dominant-region size;
* dominant-region active fraction;
* dominant-region L1 fraction;
* number of reinitializations.

The complete component table is exported to:

```text
results/tables/components.csv.gz
```

## Calendar mapping and component maps

`grid.json` uses:

```text
order = "F"
```

The loading tensor returned by Calendar-SPCA has dimensions

```text
(hour, weekday, week, component).
```

For visualization, each component is transformed using

```python
heatmap = tensor.transpose(0, 2, 1).reshape(24, 52 * 7)
```

so that the horizontal coordinate follows chronological calendar days:

```text
column = week * 7 + weekday
```

and the vertical coordinate corresponds to the hour of day.

Each component PNG displays the loading values together with fit-level and
component-level summary statistics.

## Stability analysis

After the Calendar-SPCA fits are available, the experiment performs two
complementary stability analyses.

### Local lambda-grid stability

Each fit is compared with its immediate neighbours along the two regularization
axes:

```text
lambda_l1 direction
lambda_tv direction
```

Only adjacent grid positions are compared.

For a complete experiment this produces:

```text
10,080 local stability comparisons
```

### Repetition stability

For each fixed combination

```text
(N, K, lambda_l1, lambda_tv)
```

all pairs among the five repetitions are compared.

With five repetitions there are

```text
C(5,2) = 10
```

pairwise comparisons per configuration, giving

```text
11,760 repetition-stability comparisons
```

for the complete experiment.

### Component matching

Components from two fits are matched using the Hungarian assignment algorithm
applied to the absolute cosine similarity matrix of the active loading vectors.

Two summary quantities are recorded:

```text
mean_matched_active_cosine
penalized_similarity
```

The penalized similarity is

```text
sum of matched cosine similarities / nominal K
```

so missing or inactive components contribute zero. Component collapse
therefore cannot artificially increase the reported stability.

## Central result tables

The experiment maintains the following main outputs:

```text
results/tables/fits.csv
results/tables/components.csv.gz
results/tables/stability_local.csv
results/tables/stability_repeat.csv
results/tables/failures.csv
results/tables/results.sqlite
```

`fits.csv` contains one row per completed Calendar-SPCA fit.

`components.csv.gz` contains one row per nominal component.

The two stability tables contain pairwise loading-comparison results.

`results.sqlite` stores the consolidated experiment state and uses SQLite WAL
mode so that it can be queried while the experiment is being processed.

## Monitoring

Current progress can be inspected with:

```bash
python run_full_grid.py --config grid.json status
```

or refreshed continuously with:

```bash
python run_full_grid.py --config grid.json watch --interval 60
```

The experiment also generates:

```text
results/dashboard.html
```

with execution status, recently completed fits, and aggregate heatmaps.

To regenerate aggregate tables and plots:

```bash
python run_full_grid.py --config grid.json aggregate --components --plots
```

## Querying individual fits

A completed configuration can be located with, for example:

```bash
python run_full_grid.py --config grid.json query \
    --N 20000 \
    --K 15 \
    --lambda-l1 1 \
    --lambda-tv 1
```

The lambda values supplied to the query should correspond to values present in
the configured grid.

## Serving the dashboard

The dashboard can be served from the cluster with:

```bash
python run_full_grid.py --config grid.json serve --port 8765
```

and accessed remotely through an SSH tunnel such as:

```bash
ssh -L 8765:127.0.0.1:8765 USER@SERVER
```

The local browser can then open:

```text
http://127.0.0.1:8765/dashboard.html
```

## Reproducibility safeguards

The experimental harness is designed to preserve the complete provenance of
every fit. In particular, it records or verifies:

* the exact dataset identity and SHA-256 hash;
* the exact Calendar-SPCA Git commit;
* cleanliness of tracked source files;
* the complete scientific configuration;
* sampling and initialization seeds;
* sampled row indices and their hashes;
* calendar shape and flattening order;
* software versions;
* optimization histories;
* fit and component diagnostics.

The output directory is tied to a hash of the scientific configuration. A
different scientific configuration must therefore use a different experiment
directory.

This design makes the full factorial evaluation deterministic, resumable, and
auditable while allowing the computational workload to be distributed over a
large number of worker processes.