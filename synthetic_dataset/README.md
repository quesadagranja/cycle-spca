# Synthetic Calendar-SPCA Dataset

This directory contains the reproducible synthetic benchmark used to evaluate
Calendar-SPCA under controlled conditions with known latent structure.

The purpose of this dataset is to complement the experiments on real
electricity-consumption profiles with a setting in which the true latent
components are explicitly known. This makes it possible to evaluate component
recovery objectively rather than relying only on reconstruction,
regularization diagnostics, or qualitative interpretation.

## 1. Data-generating model

The synthetic centered data matrix is generated as

\[
X = U_{\mathrm{true}} V_{\mathrm{true}}^\top + E,
\]

where:

- \(X \in \mathbb{R}^{N \times M}\) is the observed synthetic dataset;
- \(U_{\mathrm{true}} \in \mathbb{R}^{N \times K}\) contains the true latent
  scores;
- \(V_{\mathrm{true}} \in \mathbb{R}^{M \times K}\) contains the true loading
  patterns;
- \(E\) is additive Gaussian noise.

The current benchmark uses

\[
N = 5000,
\qquad
K_{\mathrm{true}} = 5,
\qquad
M = 24 \times 7 \times 52 = 8736.
\]

The feature domain therefore has the same hour--weekday--week organization used
in the annual electricity-consumption experiments.

The generated matrix is centered by construction. No artificial mean
consumption profile is added, since the benchmark is intended to evaluate
recovery of the latent structure modeled by Calendar-SPCA.

---

## 2. Ground-truth calendar components

Five latent loading patterns are defined on the
\(24 \times 7 \times 52\) calendar domain.

### Component 1: interior

A control component located entirely inside the calendar domain.

It crosses no periodic boundary and provides a reference case in which cyclic
boundary closure should have little influence on recovery.

### Component 2: daily crossing

A connected region spanning the end and beginning of the daily cycle.

For example, its within-day support includes positions around

\[
22{:}00 \rightarrow 23{:}00 \rightarrow 00{:}00 \rightarrow 01{:}00
\rightarrow 02{:}00.
\]

This component tests whether the daily cycle is represented continuously
across midnight.

### Component 3: weekly crossing

A connected region spanning the weekly boundary between Sunday and Monday.

This component tests whether the weekly cycle is represented continuously
across the end and beginning of the week.

### Component 4: annual crossing

A connected region spanning the end and beginning of the annual
representation.

Its support includes positions from both the final and first weeks of the
52-week calendar axis.

### Component 5: multi-boundary crossing

A more challenging component that simultaneously crosses the daily, weekly,
and annual boundaries.

In the toroidal calendar graph, this support forms a coherent connected
region even though ordinary vectorization places parts of it near different
edges of the feature array.

---

## 3. Loading structure

The five ground-truth loadings are fixed across all observations.

Their active regions contain mild within-region variation instead of perfectly
constant blocks. This prevents the synthetic components from being constructed
as ideal solutions of the total-variation penalty.

The interior component is separated from the boundary-crossing components.

Components 2--5 deliberately have moderate support overlap. Consequently,
different latent mechanisms may contribute to some of the same calendar
positions while remaining distinguishable as separate loading patterns.

The overlap is intentional and is quantified automatically by the generation
script.

Each ground-truth loading is normalized to unit Euclidean norm.

---

## 4. Latent-score generation

Observations are mixtures of the five latent components.

All

\[
2^5 - 1 = 31
\]

non-empty combinations of active components are represented.

The number of observations according to the number of active components is:

| Active components | Number of observations |
|---:|---:|
| 1 | 500 |
| 2 | 1500 |
| 3 | 1750 |
| 4 | 1000 |
| 5 | 250 |

Combinations of the same cardinality are represented approximately equally
often.

Latent scores vary continuously across observations, so observations sharing
the same activation pattern still have different component intensities.

Several latent components are moderately correlated. This deliberately avoids
an artificially simple setting with mutually independent latent factors and
tests the joint-factor formulation used by Calendar-SPCA.

The active scores are centered so that each complete score column has zero
sample mean.

---

## 5. Noise

Independent Gaussian noise is added after constructing the low-rank signal.

The default target signal-to-noise ratio is

\[
\mathrm{SNR} = 5\ \mathrm{dB}.
\]

The generation script computes and records the actual SNR obtained in the
generated realization.

---

## 6. Reproducibility

The dataset is generated with a fixed random seed:

```text
20260818
````

The benchmark configuration should remain frozen once the generation script
has been validated.

In particular, the following quantities define the benchmark and should not
be adjusted in response to the performance of Calendar-SPCA:

* random seed;
* number of observations;
* calendar dimensions;
* number and geometry of the five true components;
* component-overlap structure;
* latent-score correlation structure;
* signal-to-noise ratio.

This ensures that the synthetic experiment remains an independent benchmark
rather than a dataset tuned to favor a particular fitted model.

---

## 7. Files

Running the generation script creates:

```text
synthetic_dataset/
├── generate_synthetic.py
├── README.md
├── X.npy
├── U_true.npy
├── V_true.npy
├── active_mask.npy
├── truth_metadata.npz
└── config.json
```

### `X.npy`

Synthetic data matrix with shape

```text
(5000, 8736)
```

This is the only array required as input when fitting Calendar-SPCA.

### `U_true.npy`

Ground-truth latent-score matrix with shape

```text
(5000, 5)
```

This file is used only for evaluation.

### `V_true.npy`

Ground-truth loading matrix with shape

```text
(8736, 5)
```

This file is used to evaluate recovery of the five true calendar components.

### `active_mask.npy`

Boolean matrix indicating which latent components are active for each
observation.

### `truth_metadata.npz`

Numerical metadata describing the generated benchmark, including:

* calendar shape;
* target score-correlation matrix;
* realized score correlations;
* pairwise loading similarities;
* support-overlap counts;
* support-overlap fractions;
* component strengths;
* target and realized SNR.

### `config.json`

Human-readable record of the complete generation configuration.

---

## 8. Intended evaluation

The synthetic benchmark is designed to answer a specific question:

> Can Calendar-SPCA recover known sparse and coherent latent patterns defined
> on a multi-periodic calendar domain?

Because (U_{\mathrm{true}}) and (V_{\mathrm{true}}) are known, fitted
components can be evaluated objectively.

Relevant metrics include:

* absolute cosine similarity between true and estimated loadings;
* support precision and recall;
* support Jaccard similarity;
* reconstruction error or explained variance;
* connected-region recovery;
* stability across repeated initializations.

Estimated and true components should be aligned before component-wise
comparison, for example using Hungarian matching based on absolute loading
cosine similarity.

---

## 9. Cyclic-boundary ablation

An important use of this benchmark is to isolate the effect of periodic
boundary closure.

Calendar-SPCA uses a calendar graph based on the Cartesian product of cycle
graphs,

[
C_{24} \square C_7 \square C_{52}.
]

A controlled ablation can replace these cycles with open path graphs,

[
P_{24} \square P_7 \square P_{52},
]

while keeping the data, optimization procedure, number of components, and
regularization settings unchanged.

The interior component provides a control case, while the daily, weekly,
annual, and multi-boundary components directly test the effect of cyclic
connectivity.

This comparison isolates the contribution of the calendar topology from the
remaining elements of the model.

---

## 10. Scope

This benchmark is intentionally controlled.

The ground-truth loadings remain fixed across observations and no
sample-specific deformation, temporal displacement, or boundary perturbation
is introduced.

The dataset already combines several sources of difficulty:

* mixed latent components;
* continuous component intensities;
* correlated latent scores;
* partially overlapping loading supports;
* additive noise;
* single-boundary-crossing components;
* a component crossing several periodic boundaries simultaneously.

The objective is therefore to provide a challenging but exactly evaluable
ground-truth experiment.

