[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21611751.svg)](https://doi.org/10.5281/zenodo.21611751)
# Calendar Graph-Fused Sparse PCA

`calendar-gfspca` is a research implementation of sparse principal component
analysis for data observed on several periodic calendar axes. It combines:

- an L1 penalty to obtain sparse loading vectors;
- graph total variation to encourage locally coherent regions; and
- a Cartesian product of cycle graphs to represent periodic boundaries.

The validated calendar configuration is

```text
24 hours × 7 weekdays × 52 weeks = 8,736 features
```

and uses the graph

\[
G = C_{24} \mathbin{\square} C_7 \mathbin{\square} C_{52}.
\]

Consequently, the regularization treats 23:00 and 00:00, Sunday and Monday,
and week 52 and week 1 as neighbors, while retaining all ordinary
within-axis neighbor relations.

> **Status:** version 0.1.0 is an initial research release. The software is
> suitable for reproducible experimentation, but the API may evolve in later
> versions.

## Installation

Python 3.10 or later is required.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For an editable development installation:

```bash
python -m pip install -e .
python -m pip install pytest
```

The core package depends only on NumPy and SciPy. If the synthetic example
creates figures, install Matplotlib as well:

```bash
python -m pip install matplotlib
```

## Quick start

The estimator follows a scikit-learn-like interface:

```python
import numpy as np

from calendar_gfspca import CalendarGraphFusedSparsePCA

# Rows are observations; columns are calendar locations.
X = np.load("prepared_matrix.npy")

model = CalendarGraphFusedSparsePCA(
    n_components=15,
    lambda_l1=3.0,
    lambda_tv=2.0,
    outer_max_iter=500,
    random_state=42,
)

scores = model.fit_transform(X)
X_reconstructed = model.inverse_transform(scores)
```

See [`examples/synthetic_demo.py`](examples/synthetic_demo.py) for the complete
executable example and the exact configuration used by this release.

## Input data

The input matrix has shape `(n_samples, n_features)`. In the validated
configuration:

```text
n_features = 24 × 7 × 52 = 8,736.
```

Each feature must correspond consistently to one `(hour, weekday, week)`
calendar position. Use the same flattening convention as the graph constructor
and the synthetic example. Values supplied to the estimator must be finite.

Centering, scaling, normalization, missing-value treatment, and the definition
of a yearly observation are experimental choices. They should be applied
consistently before fitting and documented whenever results are reported.

## Model

For \(X \in \mathbb{R}^{n \times M}\), the estimator factorizes

\[
X \approx U V^\top
\]

by minimizing

\[
\frac{1}{2}\lVert X-U V^\top\rVert_F^2
+ \lambda_1 \sum_{k=1}^{K}\lVert v_k\rVert_1
+ \lambda_{\mathrm{TV}}\sum_{k=1}^{K}\lVert D_G v_k\rVert_1,
\]

where \(D_G\) is an oriented incidence matrix of the product-of-cycles graph.
The columns of \(V\) are sparse, calendar-structured loadings, while the rows
of \(U\) provide the low-dimensional representation of the observations.

The implementation alternates between score estimation and regularized loading
estimation. The loading subproblem is solved using a primal-dual three-operator
splitting procedure. See [`METHOD.md`](METHOD.md) for the formulation,
optimization details, conventions, and diagnostics.

## Main parameters

- `n_components`: number \(K\) of components.
- `lambda_l1`: loading sparsity strength \(\lambda_1\).
- `lambda_tv`: graph-fusion strength \(\lambda_{\mathrm{TV}}\).
- `outer_max_iter`: maximum number of alternating-optimization iterations.
- `inner_max_iter`: maximum number of iterations in the loading subproblem.
- `random_state`: seed used by stochastic or randomized initialization steps.

The exact constructor signature and all available diagnostics are documented
in the estimator source. Larger regularization values do not have an absolute
interpretation independent of data preprocessing.

## Synthetic demonstration

Run:

```bash
python examples/synthetic_demo.py
```

The example is self-contained and does not require the private research
dataset. Its purpose is to verify installation and demonstrate the estimator;
it is not a benchmark of scientific superiority over alternative methods.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The tests check the periodic graph construction, numerical validity, expected
array shapes, and core estimator behavior.

## Research data

The large empirical dataset used during development is not distributed in this
repository. In particular, no normalized `.npy` or `.csv` matrix is included.
Reproducible empirical studies should document:

- the source and permitted use of the original data;
- preprocessing and normalization;
- calendar indexing and retained features;
- exclusions and missing-value handling;
- sample identifiers or sampling seed;
- software version and full model configuration; and
- cryptographic hashes of immutable input files, where permitted.

## Reproducibility

When reporting a result, record at least:

```text
calendar-gfspca version
Python, NumPy, and SciPy versions
calendar shape and flattening convention
number of components
lambda_l1 and lambda_tv
outer and inner iteration limits
convergence tolerances
random seed
input-data hash
```

Component order and sign are not identifiable. Comparisons between fits should
therefore align components and account for sign changes.

## Citation

If you use this software, cite the archived release corresponding to the code
you ran. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
After the first Zenodo archive is created, the release-specific DOI should be
added here and to the citation metadata.

## License

This project is released under the BSD 3-Clause License. See
[`LICENSE`](LICENSE).

