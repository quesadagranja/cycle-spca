#!/usr/bin/env python3

from pathlib import Path
from itertools import combinations
import json

import numpy as np


# ============================================================
# Configuration
# ============================================================

SEED = 20260818

N = 5000

H = 24
D = 7
W = 52

K = 5
M = H * D * W

# Global signal-to-noise ratio:
#
#   SNR = 10 log10(P_signal / P_noise)
#
# 5 dB gives a clearly noisy but still recoverable problem.
SNR_DB = 5.0

DTYPE = np.float32
BATCH_SIZE = 250

OUTPUT_DIR = Path(__file__).resolve().parent

rng = np.random.default_rng(SEED)


# ============================================================
# Cyclic support windows
# ============================================================

def cyclic_window(n, start, length, taper=0):
    """
    Construct a compact-support window on a cyclic axis.

    Parameters
    ----------
    n : int
        Number of positions on the cyclic axis.

    start : int
        First active position.

    length : int
        Number of consecutive active positions.

    taper : int
        Number of positions at each edge receiving a smooth taper.

    Notes
    -----
    Indices wrap modulo n. Therefore, for n=24,

        start=22, length=5

    produces support at

        22, 23, 0, 1, 2.
    """

    x = np.zeros(n, dtype=DTYPE)

    idx = (start + np.arange(length)) % n

    weights = np.ones(length, dtype=DTYPE)

    edge = min(taper, length // 2)

    if edge > 0:
        ramp = np.sin(
            np.linspace(0, np.pi / 2, edge + 2)[1:-1]
        ).astype(DTYPE)

        weights[:edge] = ramp
        weights[-edge:] = ramp[::-1]

    x[idx] = weights

    return x


def make_component(h_spec, d_spec, w_spec, phase=0.0):
    """
    Construct one ground-truth loading on the H x D x W toroidal
    calendar domain.

    The support is a connected Cartesian region. A small smooth texture
    is included so that the active region is not a perfectly constant
    block specifically tailored to total-variation regularization.
    """

    wh = cyclic_window(H, **h_spec)
    wd = cyclic_window(D, **d_spec)
    ww = cyclic_window(W, **w_spec)

    grid = (
        wh[:, None, None]
        * wd[None, :, None]
        * ww[None, None, :]
    )

    # --------------------------------------------------------
    # Mild within-region variation.
    #
    # This deliberately prevents V_true from consisting of
    # perfectly flat blocks.
    # --------------------------------------------------------

    hh = np.arange(H)[:, None, None]
    dd = np.arange(D)[None, :, None]
    ww_idx = np.arange(W)[None, None, :]

    texture = (
        1.0
        + 0.07 * np.sin(2 * np.pi * hh / H + phase)
        + 0.05 * np.cos(2 * np.pi * dd / D + 0.5 * phase)
        + 0.05 * np.sin(2 * np.pi * ww_idx / W + 0.3 * phase)
    )

    grid = grid * texture.astype(DTYPE)

    v = grid.reshape(-1, order="C").astype(DTYPE)

    # Unit-norm ground-truth direction.
    v /= np.linalg.norm(v)

    return v


# ============================================================
# Ground-truth components
# ============================================================
#
# Day convention:
#
#   0 = Monday
#   1 = Tuesday
#   ...
#   6 = Sunday
#
# Week convention:
#
#   0, ..., 51
#
# The semantic labels are secondary; what matters experimentally
# is which periodic boundary each component crosses.
#
# ------------------------------------------------------------
#
# 1. interior
#       crosses no boundary
#
# 2. daily_crossing
#       crosses hour 23 -> 0
#
# 3. weekly_crossing
#       crosses Sunday -> Monday
#
# 4. annual_crossing
#       crosses week 51 -> 0
#
# 5. multi_boundary
#       crosses all three simultaneously
#
# Components 2--5 deliberately overlap moderately.
# Component 1 acts as an interior control.
# ============================================================

component_specs = [

    {
        "name": "interior",
        "description": "Interior control; crosses no periodic boundary.",

        # 16--21
        "h": dict(start=16, length=6, taper=1),

        # Tue--Fri
        "d": dict(start=1, length=4, taper=1),

        # Weeks 20--33
        "w": dict(start=20, length=14, taper=2),
    },

    {
        "name": "daily_crossing",
        "description": "Crosses the within-day boundary.",

        # 22, 23, 0, 1, 2
        "h": dict(start=22, length=5, taper=1),

        # Mon--Fri
        "d": dict(start=0, length=5, taper=1),

        # Weeks 0--13
        "w": dict(start=0, length=14, taper=2),
    },

    {
        "name": "weekly_crossing",
        "description": "Crosses the Sunday--Monday boundary.",

        # 00--05
        "h": dict(start=0, length=6, taper=1),

        # Sunday, Monday
        "d": dict(start=6, length=2, taper=0),

        # Weeks 0--13
        "w": dict(start=0, length=14, taper=2),
    },

    {
        "name": "annual_crossing",
        "description": "Crosses the end--beginning annual boundary.",

        # 01--06
        "h": dict(start=1, length=6, taper=1),

        # Mon--Thu
        "d": dict(start=0, length=4, taper=1),

        # Weeks 48--51, 0--5
        "w": dict(start=48, length=10, taper=2),
    },

    {
        "name": "multi_boundary",
        "description": (
            "Crosses the daily, weekly, and annual boundaries "
            "simultaneously."
        ),

        # 21, 22, 23, 0, 1, 2, 3
        "h": dict(start=21, length=7, taper=1),

        # Sat, Sun, Mon, Tue
        "d": dict(start=5, length=4, taper=1),

        # Weeks 48--51, 0--4
        "w": dict(start=48, length=9, taper=2),
    },
]


V_true = np.zeros((M, K), dtype=DTYPE)

for k, spec in enumerate(component_specs):

    V_true[:, k] = make_component(
        spec["h"],
        spec["d"],
        spec["w"],
        phase=0.8 * k,
    )


# ============================================================
# Ground-truth support overlap
# ============================================================

support = V_true != 0

overlap_count = np.zeros((K, K), dtype=np.int32)
overlap_jaccard = np.zeros((K, K), dtype=np.float64)
overlap_smaller_fraction = np.zeros((K, K), dtype=np.float64)

for i in range(K):
    for j in range(K):

        a = support[:, i]
        b = support[:, j]

        inter = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)

        smaller = min(
            np.count_nonzero(a),
            np.count_nonzero(b)
        )

        overlap_count[i, j] = inter

        overlap_jaccard[i, j] = (
            inter / union if union > 0 else 0.0
        )

        overlap_smaller_fraction[i, j] = (
            inter / smaller if smaller > 0 else 0.0
        )


# ============================================================
# Loading cosine similarities
# ============================================================

loading_cosine = np.abs(V_true.T @ V_true)


# ============================================================
# Balanced activation combinations
# ============================================================
#
# There are 2^5 - 1 = 31 non-empty activation combinations.
#
# We deliberately favor mixtures involving 2--4 components.
#
# Every combination of the same cardinality is represented
# approximately equally often.
# ============================================================

counts_by_n_active = {
    1: 500,
    2: 1500,
    3: 1750,
    4: 1000,
    5: 250,
}

assert sum(counts_by_n_active.values()) == N


sample_subsets = []

for r, total in counts_by_n_active.items():

    subsets = list(combinations(range(K), r))

    # Randomize which subsets receive the few remainder samples.
    rng.shuffle(subsets)

    base = total // len(subsets)
    remainder = total % len(subsets)

    for j, subset in enumerate(subsets):

        n_subset = base + (1 if j < remainder else 0)

        sample_subsets.extend(
            [subset] * n_subset
        )


assert len(sample_subsets) == N

# Shuffle observations globally.
rng.shuffle(sample_subsets)


active_mask = np.zeros((N, K), dtype=bool)

for i, subset in enumerate(sample_subsets):
    active_mask[i, list(subset)] = True


n_active = active_mask.sum(axis=1)


# ============================================================
# Correlated latent scores
# ============================================================
#
# This matrix defines the intended correlation between latent
# components when the corresponding pair is simultaneously active.
#
# It is deliberately moderate: enough to make factor separation
# non-trivial without making the components almost redundant.
# ============================================================

R_target = np.array(
    [
        [1.00, 0.25, 0.10, 0.05, 0.10],
        [0.25, 1.00, 0.20, 0.10, 0.35],
        [0.10, 0.20, 1.00, 0.20, 0.30],
        [0.05, 0.10, 0.20, 1.00, 0.30],
        [0.10, 0.35, 0.30, 0.30, 1.00],
    ],
    dtype=np.float64,
)


# Verify positive definiteness.
eigvals = np.linalg.eigvalsh(R_target)

if eigvals.min() <= 0:
    raise RuntimeError(
        "R_target must be positive definite."
    )


U_true = np.zeros((N, K), dtype=np.float64)


# Draw correlated scores conditional on the active subset.
#
# This is preferable to generating five correlated scores and
# zeroing some of them afterwards, because the desired correlation
# structure is then preserved among simultaneously active factors.

unique_subsets = sorted(set(sample_subsets))

for subset in unique_subsets:

    rows = np.array([
        s == subset
        for s in sample_subsets
    ])

    idx = np.array(subset, dtype=int)

    n_rows = rows.sum()

    if len(idx) == 1:

        z = rng.standard_normal((n_rows, 1))

    else:

        covariance = R_target[np.ix_(idx, idx)]

        z = rng.multivariate_normal(
            mean=np.zeros(len(idx)),
            cov=covariance,
            size=n_rows,
        )

    # Keep total signal energy from increasing automatically
    # just because more latent components are active.
    z /= np.sqrt(len(idx))

    U_true[np.ix_(rows, idx)] = z


# ============================================================
# Observation and component amplitudes
# ============================================================

# Mild observation-to-observation amplitude variability.
sample_scale = rng.lognormal(
    mean=0.0,
    sigma=0.15,
    size=N,
)

U_true *= sample_scale[:, None]


# The five population components do not all have exactly
# identical strength.
component_strength = np.array(
    [1.00, 0.90, 1.05, 0.85, 0.95],
    dtype=np.float64,
)

U_true *= component_strength[None, :]


# ============================================================
# Exact centering of latent scores
# ============================================================
#
# Calendar-SPCA models centered data X_c.
#
# Inactive scores remain exactly zero. For every component, the
# active scores are centered so that each full score column has
# exactly zero sample mean.
# ============================================================

for k in range(K):

    rows = active_mask[:, k]

    U_true[rows, k] -= U_true[rows, k].mean()


U_true = U_true.astype(DTYPE)


# Check centering.
assert np.allclose(
    U_true.mean(axis=0),
    0.0,
    atol=1e-6,
)


# ============================================================
# Observed score correlations
# ============================================================
#
# Correlation is measured only over observations in which both
# corresponding components are active.
# ============================================================

score_correlation = np.eye(K, dtype=np.float64)

for i in range(K):
    for j in range(i + 1, K):

        rows = active_mask[:, i] & active_mask[:, j]

        corr = np.corrcoef(
            U_true[rows, i],
            U_true[rows, j],
        )[0, 1]

        score_correlation[i, j] = corr
        score_correlation[j, i] = corr


# ============================================================
# Determine noise level from exact signal power
# ============================================================
#
# Signal:
#
#       S = U_true @ V_true.T
#
# Instead of allocating the whole signal matrix merely to estimate
# its RMS, use
#
#   || U V^T ||_F^2
#       = trace((U^T U)(V^T V)).
# ============================================================

UtU = U_true.T.astype(np.float64) @ U_true.astype(np.float64)
VtV = V_true.T.astype(np.float64) @ V_true.astype(np.float64)

signal_power_total = np.trace(UtU @ VtV)

signal_power_mean = signal_power_total / (N * M)

signal_rms = np.sqrt(signal_power_mean)

noise_std = signal_rms / (10.0 ** (SNR_DB / 20.0))


# ============================================================
# Generate X in batches
# ============================================================
#
# X is generated directly as centered synthetic data:
#
#       X = U_true V_true^T + E
#
# No artificial mean consumption profile is added. The experiment
# is intended to test recovery of latent structure from X_c.
#
# Writing through open_memmap avoids holding X, signal, and noise
# simultaneously in RAM.
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

x_path = OUTPUT_DIR / "X.npy"

X = np.lib.format.open_memmap(
    x_path,
    mode="w+",
    dtype=DTYPE,
    shape=(N, M),
)


noise_ss = 0.0
signal_ss_generated = 0.0


for start in range(0, N, BATCH_SIZE):

    stop = min(start + BATCH_SIZE, N)

    signal_batch = (
        U_true[start:stop]
        @ V_true.T
    ).astype(DTYPE)

    noise_batch = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal_batch.shape,
    ).astype(DTYPE)

    X[start:stop] = signal_batch + noise_batch

    signal_ss_generated += np.sum(
        signal_batch.astype(np.float64) ** 2
    )

    noise_ss += np.sum(
        noise_batch.astype(np.float64) ** 2
    )


X.flush()

del X


actual_snr_db = (
    10.0
    * np.log10(
        signal_ss_generated / noise_ss
    )
)


# ============================================================
# Save exact ground truth
# ============================================================

np.save(
    OUTPUT_DIR / "U_true.npy",
    U_true,
)

np.save(
    OUTPUT_DIR / "V_true.npy",
    V_true,
)

np.save(
    OUTPUT_DIR / "active_mask.npy",
    active_mask,
)


# Additional numerical metadata in one compact NumPy archive.
np.savez(
    OUTPUT_DIR / "truth_metadata.npz",

    calendar_shape=np.array(
        [H, D, W],
        dtype=np.int32,
    ),

    R_target=R_target,

    score_correlation=score_correlation,

    loading_cosine=loading_cosine,

    overlap_count=overlap_count,

    overlap_jaccard=overlap_jaccard,

    overlap_smaller_fraction=overlap_smaller_fraction,

    component_strength=component_strength,

    n_active=n_active,

    snr_db_target=np.array(SNR_DB),

    snr_db_actual=np.array(actual_snr_db),

    noise_std=np.array(noise_std),
)


# Human-readable configuration.
config = {
    "seed": SEED,
    "N": N,
    "M": M,
    "K_true": K,

    "calendar_shape": [H, D, W],

    "flattening_order": "C",

    "dtype": "float32",

    "snr_db_target": SNR_DB,
    "snr_db_actual": float(actual_snr_db),

    "counts_by_n_active": counts_by_n_active,

    "components": component_specs,

    "component_strength": component_strength.tolist(),

    "score_correlation_target": R_target.tolist(),

    "notes": [
        "X is generated directly in centered form.",
        "Ground-truth loadings are fixed across observations.",
        "No sample-specific loading deformation is applied.",
        "Components 2--5 have deliberately overlapping supports.",
        "Component 1 is a non-boundary-crossing interior control.",
        "Component 5 crosses all three periodic boundaries."
    ],
}


with open(
    OUTPUT_DIR / "config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# Console audit
# ============================================================

names = [
    spec["name"]
    for spec in component_specs
]


print()
print("=" * 72)
print("SYNTHETIC CALENDAR-SPCA DATASET")
print("=" * 72)

print(f"N                  : {N}")
print(f"M                  : {M}")
print(f"K_true             : {K}")
print(f"calendar shape     : ({H}, {D}, {W})")

print(
    f"target SNR         : {SNR_DB:.3f} dB"
)

print(
    f"actual SNR         : {actual_snr_db:.3f} dB"
)

print(
    f"noise std          : {noise_std:.6g}"
)


print()
print("Samples by number of active components:")

for r in sorted(counts_by_n_active):

    actual = np.count_nonzero(
        n_active == r
    )

    print(
        f"  {r}: {actual}"
    )


print()
print("Ground-truth support sizes:")

for k, name in enumerate(names):

    print(
        f"  {name:20s}: "
        f"{support[:, k].sum():4d}"
    )


print()
print(
    "Pairwise overlap "
    "(fraction of smaller support):"
)

for i in range(K):
    for j in range(i + 1, K):

        print(
            f"  {names[i]:20s} "
            f"<-> {names[j]:20s}: "
            f"{overlap_smaller_fraction[i, j]:.3f}"
        )


print()
print("Absolute loading cosine similarities:")

for i in range(K):
    for j in range(i + 1, K):

        print(
            f"  {names[i]:20s} "
            f"<-> {names[j]:20s}: "
            f"{loading_cosine[i, j]:.3f}"
        )


print()
print(
    "Observed score correlations "
    "(conditional on joint activation):"
)

for i in range(K):
    for j in range(i + 1, K):

        print(
            f"  {names[i]:20s} "
            f"<-> {names[j]:20s}: "
            f"{score_correlation[i, j]:+.3f}"
        )


print()
print("Files written:")

for path in sorted(OUTPUT_DIR.iterdir()):

    size_mb = path.stat().st_size / (1024 ** 2)

    print(
        f"  {path.name:24s} "
        f"{size_mb:8.2f} MiB"
    )


print()
print("Done.")