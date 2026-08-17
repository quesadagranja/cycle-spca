# Method

This document specifies the mathematical model and computational conventions implemented by `calendar-gfspca` version 0.1.0. The description below follows the estimator source code and its default configuration.

## 1. Data representation

Let

[
X\in\mathbb{R}^{n\times M}
]

contain (n) observations and (M) ordered features.

The current implementation represents the feature domain using exactly three periodic axes,

[
(m_1,m_2,m_3),
\qquad
M=m_1m_2m_3,
]

with each (m_r\geq 2).

The validated configuration for annual hourly profiles is

[
(m_1,m_2,m_3)=(24,7,52),
\qquad
M=8736,
]

with the axes interpreted as hour of day, weekday, and week of year.

The mathematical product-of-cycles construction extends naturally to more than three periodic axes, but version 0.1.0 of the software implements exactly three.

### 1.1 Flattening convention

A feature corresponds to one location

[
(i_1,i_2,i_3)
\in
{0,\ldots,m_1-1}
\times
{0,\ldots,m_2-1}
\times
{0,\ldots,m_3-1}.
]

The estimator supports two flattening conventions:

* `order="C"`: NumPy/C order, which is the default; for the calendar tensor `(hour, day, week)`, the week index varies fastest;
* `order="F"`: Fortran/R-compatible order.

The same convention is used for reshaping, graph differences, graph adjoints, region extraction, and loading visualization.

### 1.2 Centering

By default, the estimator uses

```text
center=True
```

and internally computes the column mean

[
\mu
===

\frac{1}{n}X^\top\mathbf{1}_n
]

and the centered matrix

[
X_c
===

X-\mathbf{1}_n\mu^\top.
]

If `center=False`, then

[
\mu=0,
\qquad
X_c=X.
]

The centered matrix need not be materialized. Matrix products involving (X_c) and (X_c^\top) are evaluated in row batches while applying the centering correction algebraically.

Scaling, normalization, missing-value treatment, and the definition of an observation are not performed automatically and remain experimental preprocessing choices.

All supplied values must be finite.

---

## 2. Toroidal calendar graph

The feature domain is represented by the Cartesian product of three cycle graphs,

[
G
=

C_{m_1}\square C_{m_2}\square C_{m_3}.
]

For every location

[
i=(i_1,i_2,i_3),
]

the implementation evaluates one forward cyclic difference along each axis:

[
(\Delta_r v)(i)
===============

## v(i)

v(i+e_r\ \mathrm{mod}\ m_r),
\qquad
r=1,2,3.
]

Thus every axis is closed periodically.

For the validated calendar,

[
G=C_{24}\square C_7\square C_{52},
]

so the regularizer explicitly includes the boundary relations

[
23\leftrightarrow0,
\qquad
6\leftrightarrow0,
\qquad
51\leftrightarrow0.
]

The implementation does not materialize an incidence matrix. Instead, the graph operator is evaluated using cyclic tensor shifts.

If (D_G) denotes the equivalent oriented incidence operator, then for a loading matrix

[
V\in\mathbb{R}^{M\times K},
]

the implementation represents

[
D_GV
]

as an array of shape

```text
(3, M, K).
```

There are therefore (3M) forward cyclic differences per component. For the validated calendar,

[
3M
==

# 3(8736)

26208.

]

The orientation of an edge is irrelevant to the regularizer because absolute differences are used.

---

## 3. Graph total variation

For a loading vector (v\in\mathbb{R}^M), the implemented anisotropic graph total variation is

[
\operatorname{TV}_G(v)
======================

|D_Gv|_1.
]

Equivalently,

[
\operatorname{TV}_G(v)
======================

\sum_{r=1}^{3}
\sum_i
\left|
v(i)-v(i+e_r\ \mathrm{mod}\ m_r)
\right|.
]

For

[
V=[v_1,\ldots,v_K],
]

the complete loading penalty is

[
|D_GV|_{1,1}
============

\sum_{k=1}^{K}|D_Gv_k|_1.
]

The graph penalty is therefore anisotropic: absolute differences along the three axes are summed independently rather than grouped through an isotropic Euclidean norm.

### 3.1 Exact operator norm

The squared spectral norm used by the solver is computed analytically as

[
|D_G|_2^2
=========

\sum_{r=1}^{3}\ell(m_r),
]

where

[
\ell(m)
=======

\begin{cases}
4, & m\text{ even},[0.3em]
2+2\cos(\pi/m), & m\text{ odd}.
\end{cases}
]

For

[
(m_1,m_2,m_3)=(24,7,52),
]

this gives

[
\begin{aligned}
|D_G|_2^2
&=
4+
\left(2+2\cos\frac{\pi}{7}\right)
+4\
&=
10+2\cos\frac{\pi}{7}\
&\approx11.80194.
\end{aligned}
]

This value is used directly in the primal-dual step-size calculation.

---

## 4. CycleSPCA factorization

For a nominal number of components (K), define

[
U=[u_1,\ldots,u_K]\in\mathbb{R}^{n\times K},
]

and

[
V=[v_1,\ldots,v_K]\in\mathbb{R}^{M\times K}.
]

The estimator minimizes

[
\boxed{
\mathcal J(U,V)
===============

\frac12
|X_c-UV^\top|*F^2
+
\lambda_1|V|*{1,1}
+
\lambda_{\mathrm{TV}}|D_GV|_{1,1}
}
]

subject, for every active component, to

[
\boxed{
|u_k|_2=1.
}
]

Here

[
|V|_{1,1}
=========

\sum_{k=1}^{K}|v_k|_1.
]

There is no ridge or elastic-net term in version 0.1.0.

There is no orthogonality constraint between distinct columns of (U), and there is no orthogonality constraint between distinct columns of (V).

CycleSPCA is therefore a joint, non-orthogonal low-rank factorization rather than a sequential deflation method.

The two regularization parameters have distinct roles:

* (\lambda_1) promotes exact zeros in the loading matrix;
* (\lambda_{\mathrm{TV}}) promotes equal or similar values between neighboring locations of the periodic graph.

The objective is not jointly convex in ((U,V)). The loading problem is convex for fixed (U), while each individual score-coordinate problem has an exact solution under the unit-norm constraint.

The algorithm therefore seeks a stationary solution through alternating optimization; it does not claim convergence to the global optimum of the complete non-convex problem.

---

## 5. Initialization

The default initialization is

```text
initialization="svd"
```

and uses a randomized truncated SVD of (X_c).

If

[
X_c
\approx
\widetilde U_K
\Sigma_K
\widetilde V_K^\top,
]

the initialization has the form

[
U^{(0)}
=======

\widetilde U_K,
]

[
V^{(0)}
=======

\widetilde V_K\Sigma_K.
]

Thus the initial score columns have unit norm and the loading columns contain the corresponding singular-value amplitudes.

The implementation uses Gaussian randomized range finding, QR orthogonalization, and the configured number of power iterations.

The relevant defaults are

```text
initialization = "svd"
svd_power_iterations = 2
svd_oversamples = 5
```

A random initialization is also available with

```text
initialization = "random"
```

and uses randomized loading directions followed by data-dependent score and amplitude initialization.

---

## 6. Score update

### 6.1 Default coordinate solver

The default score solver is

```text
score_solver = "coordinate"
```

and is the score update used by the reference method.

For fixed (V), consider active component (k). Define the reconstruction excluding that component as

[
R_{-k}
======

X_c-\sum_{j\neq k}u_jv_j^\top.
]

The score-coordinate problem is

[
\min_{|u|*2=1}
\frac12
|R*{-k}-uv_k^\top|_F^2.
]

Its exact solution is obtained from

[
\widetilde u_k
==============

R_{-k}v_k.
]

Using the currently stored factors,

[
\boxed{
\widetilde u_k
==============

## X_cv_k

\sum_{\substack{j\neq k\j\text{ active}}}
u_j(v_j^\top v_k).
}
]

If

[
|\widetilde u_k|_2
]

is greater than the configured collapse tolerance, the update is

[
\boxed{
u_k
===

\frac{\widetilde u_k}
{|\widetilde u_k|_2}.
}
]

This update does not rescale (v_k).

Several coordinate sweeps may be performed. With the default configuration,

```text
score_sweeps = 10
score_tolerance = 1e-7
```

and the sweeps stop early when

[
\frac{
|U_A^{\mathrm{new}}-U_A^{\mathrm{old}}|_F
}{
|U_A^{\mathrm{old}}|_F+\varepsilon
}
<
\texttt{score_tolerance},
]

where (A) denotes the set of active components.

### 6.2 Experimental rescaled least-squares solver

The implementation also retains

```text
score_solver = "rescaled_ls"
```

for experimentation.

For the active loading block (V_A), it computes

[
\widetilde U_A
==============

X_cV_A
(V_A^\top V_A)^\dagger.
]

Each resulting score column is normalized and the corresponding loading column is multiplied by the removed scale.

Although this preserves the rank-one reconstruction term during the rescaling, it changes the magnitude of the L1 and graph-TV penalties. The implementation therefore emits a runtime warning that this update is not guaranteed to decrease the complete penalized objective.

`rescaled_ls` is not the default CycleSPCA score solver.

---

## 7. Loading subproblem

For fixed (U), the estimator solves

[
\min_V
\frac12
|X_c-UV^\top|*F^2
+
\lambda_1|V|*{1,1}
+
\lambda_{\mathrm{TV}}|D_GV|_{1,1}.
]

Define

[
f(V)
====

\frac12
|X_c-UV^\top|_F^2.
]

Its gradient is

[
\boxed{
\nabla f(V)
===========

V(U^\top U)-X_c^\top U.
}
]

The gradient Lipschitz constant is

[
\boxed{
L_U
===

\lambda_{\max}(U^\top U).
}
]

The proximal operator associated with the L1 penalty is elementwise soft thresholding:

[
\operatorname{soft}(z,a)
========================

\operatorname{sign}(z)
\max(|z|-a,0).
]

The dual feasible set associated with graph TV is the elementwise box

[
[-\lambda_{\mathrm{TV}},
\lambda_{\mathrm{TV}}].
]

---

## 8. Primal-dual loading solver

The loading subproblem is solved by the following Condat--Vu-type primal-dual iterations.

Let (Y) denote the graph dual variable.

At inner iteration (\ell),

[
G^{(\ell)}
==========

V^{(\ell)}(U^\top U)-X_c^\top U.
]

The primal update is

[
\boxed{
V^{(\ell+1)}
============

\operatorname{soft}
\left(
V^{(\ell)}
----------

\tau
\left[
G^{(\ell)}
+
D_G^\top Y^{(\ell)}
\right],
,
\tau\lambda_1
\right).
}
]

The extrapolated primal variable is

[
\boxed{
\overline V^{(\ell+1)}
======================

2V^{(\ell+1)}-V^{(\ell)}.
}
]

The dual update is

[
\boxed{
Y^{(\ell+1)}
============

\operatorname{clip}
\left(
Y^{(\ell)}
+
\sigma
D_G\overline V^{(\ell+1)},
-\lambda_{\mathrm{TV}},
\lambda_{\mathrm{TV}}
\right).
}
]

The dual variable is retained between outer iterations unless a component is reinitialized or an inner loading update is rejected.

### 8.1 Step sizes

Let

[
d_G^2
=====

|D_G|_2^2.
]

The implementation sets

[
\boxed{
\sigma
======

\frac{
\texttt{dual_step_scale}
}{
\sqrt{d_G^2}
}
}
]

and

[
\boxed{
\tau
====

\frac{
\texttt{step_safety}
}{
L_U/2+\sigma d_G^2
}.
}
]

The defaults are

```text
dual_step_scale = 1.0
step_safety = 0.99
```

Before starting the iterations, the implementation verifies the strict condition

[
\boxed{
\frac1\tau
----------

\sigma|D_G|_2^2

>

\frac{L_U}{2}.
}
]

Failure of this internal check raises an exception rather than continuing with invalid step sizes.

---

## 9. Inner convergence, residual, and rollback

At outer iteration (t), the requested loading-solver tolerance is

[
\boxed{
\varepsilon_t
=============

\max
\left{
\varepsilon_{\min},
\varepsilon_0\rho^t
\right}.
}
]

The defaults are

```text
epsilon_0 = 1e-3
epsilon_min = 1e-6
epsilon_rho = 0.75
```

The solver monitors the loading iterate every

```text
inner_check_interval = 25
```

iterations by default.

Two numerical criteria are evaluated.

### 9.1 Relative loading change

If (V_{\mathrm{check}}) is the loading matrix at the preceding monitoring point,

[
r_V
===

\frac{
|V-V_{\mathrm{check}}|*F
}{
|V*{\mathrm{check}}|_F+\varepsilon
}.
]

### 9.2 Normalized primal-dual fixed-point residual

The implementation constructs the fixed-point primal map

[
P(V,Y)
======

\operatorname{soft}
\left[
V-\tau
\left(
\nabla f(V)+D_G^\top Y
\right),
\tau\lambda_1
\right]
]

and the dual map

[
Q(V,Y)
======

\operatorname{clip}
\left(
Y+\sigma D_GV,
-\lambda_{\mathrm{TV}},
\lambda_{\mathrm{TV}}
\right).
]

The normalized residual is

[
r_{\mathrm{PD}}
===============

\frac{
\sqrt{
\left(
\frac{|V-P(V,Y)|_F}{\tau}
\right)^2
+
\left(
\frac{|Y-Q(V,Y)|_F}{\sigma}
\right)^2
}
}{
1+|V|_F+|Y|_F
}.
]

The inner solver declares convergence only when

[
r_V<\varepsilon_t,
]

[
r_{\mathrm{PD}}<\varepsilon_t,
]

and the loading-subproblem objective has not increased relative to its value at the beginning of the current outer loading update.

The default maximum number of inner iterations is

```text
inner_max_iter = 20000
```

### 9.3 Acceptance and rollback

Even if the inner convergence criterion is not reached, the final loading iterate is accepted when its loading-subproblem objective does not exceed the initial value, up to numerical tolerance.

If the final loading objective has increased, the implementation restores both:

* the loading matrix present before the inner solve; and
* the corresponding previous dual variable.

Thus an unsuccessful loading solve is rolled back.

---

## 10. Degeneracy and redundancy handling

Regularization and non-orthogonality can produce collapsed or redundant factors. The implementation explicitly monitors these conditions.

### 10.1 Collapsed score direction

If

[
|\widetilde u_k|_2
<
\texttt{score_collapse_tolerance},
]

component (k) is flagged.

The default is

```text
score_collapse_tolerance = 1e-12
```

### 10.2 Collapsed loading

If

[
|v_k|_2
<
\texttt{loading_collapse_tolerance},
]

the component is flagged.

The default is

```text
loading_collapse_tolerance = 1e-10
```

### 10.3 Pairwise redundancy

For active loading columns, the implementation computes absolute cosine similarity.

A pair is considered nearly duplicated when

[
\frac{
|v_i^\top v_j|
}{
|v_i|_2|v_j|_2
}

>

\texttt{similarity_threshold}.
]

The default threshold is

```text
similarity_threshold = 0.995
```

When a redundant pair is detected, the component with the smaller conditional reconstruction contribution is selected for reinitialization.

### 10.4 General linear dependence

Pairwise similarities do not detect every multicolumn dependency. Therefore the estimator also monitors the Gram matrix

[
V_A^\top V_A
]

of the active loading block.

If its condition number exceeds

```text
condition_threshold = 1e12
```

the eigenvector corresponding to the smallest Gram eigenvalue is used to identify the loading columns most strongly implicated in the dependence.

Among those columns, the component with the smallest conditional reconstruction contribution is selected for reinitialization.

---

## 11. Component reinitialization and deactivation

A flagged component is reinitialized using a residual-oriented randomized power procedure.

A random candidate loading direction is first projected away from the span of the remaining active loadings.

The implementation then alternates residual-oriented multiplication by (X_c) and (X_c^\top), repeatedly projecting the candidate away from the remaining loading directions.

The default number of power iterations is

```text
reinit_power_iterations = 5
```

The resulting direction is assigned a data-dependent amplitude and a unit-norm score vector.

The dual variable associated with a reinitialized component is reset to zero.

Each component may be reinitialized at most

```text
max_reinitializations = 3
```

times by default.

Once this limit is exceeded, the component is deactivated and its score and loading columns are set to zero.

The effective fitted rank is therefore

[
\boxed{
K_{\mathrm{eff}}
================

\sum_{k=1}^{K}
\mathbf{1}{k\text{ active}}.
}
]

Consequently,

[
K_{\mathrm{eff}}\leq K.
]

---

## 12. Outer alternating optimization

One outer iteration performs:

1. score updates for the currently active components;
2. computation of the current inner tolerance (\varepsilon_t);
3. one primal-dual solution attempt for the complete loading block;
4. loading-solver acceptance or rollback;
5. collapsed-component and redundancy checks;
6. component reinitialization or deactivation when necessary;
7. objective, reconstruction, explained-variance, conditioning, and convergence diagnostics.

The default outer iteration limit is

```text
outer_max_iter = 100
```

although applications may explicitly request a larger value.

### 12.1 Outer convergence

Let

[
J_t
===

\mathcal J(U_t,V_t).
]

The implementation computes the relative objective change

[
r_J
===

\frac{
|J_t-J_{t-1}|
}{
|J_{t-1}|+\varepsilon
}.
]

It also computes the relative change of the fitted reconstruction,

[
r_R
===

\frac{
|U_tV_t^\top-U_{t-1}V_{t-1}^\top|*F
}{
|U*{t-1}V_{t-1}^\top|_F+\varepsilon
}.
]

The estimator declares outer convergence only when all of the following hold:

[
r_J
<
\texttt{outer_objective_tolerance},
]

[
r_R
<
\texttt{outer_reconstruction_tolerance},
]

the current loading subproblem was declared converged,

no component was reinitialized during the current outer iteration,

and at least two outer iterations have been completed.

The defaults are

```text
outer_objective_tolerance = 1e-6
outer_reconstruction_tolerance = 1e-6
```

Reaching `outer_max_iter` without satisfying these conditions leaves

```text
converged_ = False
```

even though the fitted factors and diagnostics remain available.

---

## 13. Reconstruction and explained variance

Define

[
\mathrm{SSE}
============

|X_c-UV^\top|_F^2.
]

If centering is enabled,

[
\mathrm{TSS}
============

|X_c|_F^2.
]

If centering is disabled,

[
\mathrm{TSS}
============

|X|_F^2.
]

The reported explained-variance or reconstruction fraction is

[
\boxed{
\mathrm{EV}
===========

1-
\frac{\mathrm{SSE}}{\mathrm{TSS}}.
}
]

This is a reconstruction-based quantity.

Because CycleSPCA components are not constrained to be mutually orthogonal, this value is not decomposed additively into ordinary PCA component variances.

---

## 14. Conditional component contribution

For component (k), define the reconstruction with that rank-one term removed while leaving all other fitted factors fixed:

[
\widehat X_{-k}
===============

\sum_{j\neq k}u_jv_j^\top.
]

The implementation reports

[
\boxed{
\Delta_k
========

## |X_c-\widehat X_{-k}|_F^2

|X_c-UV^\top|_F^2.
}
]

This is the increase in squared reconstruction error caused by deleting component (k).

It is a conditional reconstruction contribution, not an orthogonal variance decomposition.

The estimator exposes:

* the raw sum-of-squares contribution;
* the ratio (\Delta_k/\mathrm{TSS}); and
* the corresponding percentage.

These contributions may be used to reorder fitted components without changing the fitted reconstruction.

---

## 15. Sign and order identifiability

The factorization is sign-indeterminate because

[
u_kv_k^\top
===========

(-u_k)(-v_k)^\top.
]

After fitting, each active component is assigned a deterministic sign by locating the largest-magnitude element of (v_k) and making that element positive.

Component order is also not intrinsically identifiable.

The estimator does not automatically sort active components during fitting. An explicit `reorder_components()` operation can subsequently order them by either:

* conditional reconstruction contribution; or
* loading (\ell_2) norm.

Inactive components are moved to the end.

Comparisons between independent fits should align components before comparing column indices.

---

## 16. Out-of-sample transform

For new observations

[
X_{\mathrm{new}}
\in
\mathbb{R}^{n_{\mathrm{new}}\times M},
]

the estimator applies the stored training mean:

[
X_{\mathrm{new},c}
==================

X_{\mathrm{new}}-\mathbf{1}\mu^\top.
]

Let (V_A) contain only active loading columns.

The out-of-sample least-squares scores are

[
\boxed{
Z_A
===

X_{\mathrm{new},c}
V_A
(V_A^\top V_A)^\dagger.
}
]

Inactive score coordinates are set to zero.

These new scores are not normalized across the new sample.

This distinction is intentional: the unit-norm constraint on the training columns of (U) is an identifiability convention of the fitting procedure, whereas `transform()` computes ordinary least-squares coordinates relative to the fitted loading basis.

### 16.1 Training transform

`fit_transform(X)` returns the fitted training matrix

[
U,
]

not a second least-squares transformation of the training observations.

---

## 17. Inverse transform

For any supplied score matrix

[
Z\in\mathbb{R}^{n_{\mathrm{new}}\times K},
]

the estimator reconstructs

[
\widehat X_c
============

ZV^\top.
]

By default, `inverse_transform()` adds the stored mean,

[
\boxed{
\widehat X
==========

ZV^\top+\mathbf{1}\mu^\top.
}
]

The mean addition can be disabled with

```text
add_mean=False.
```

---

## 18. Sparsity and graph diagnostics

A loading coefficient is considered numerically inactive when

[
|v_{jk}|
\leq
\tau_{\mathrm{supp}},
]

where the default support threshold is

```text
sparsity_tolerance = 1e-10.
```

For each component, the implementation reports:

### Loading sparsity

[
\mathrm{sparsity}_k
===================

\frac1M
\sum_{j=1}^{M}
\mathbf{1}
\left{
|v_{jk}|
\leq
\tau_{\mathrm{supp}}
\right}.
]

### Active cells

[
N_k^{\mathrm{active}}
=====================

\sum_{j=1}^{M}
\mathbf{1}
\left{
|v_{jk}|

>

\tau_{\mathrm{supp}}
\right}.
]

### Loading norms

[
|v_k|_1,
\qquad
|v_k|_2.
]

### Graph total variation

[
\operatorname{TV}_k
===================

|D_Gv_k|_1.
]

### Relative graph total variation

[
\operatorname{relativeTV}_k
===========================

\frac{
|D_Gv_k|_1
}{
|v_k|_1
}
]

when (|v_k|_1>0).

The estimator also reports the absolute cosine-similarity matrix between loading columns and the empirical correlation matrix between active training score columns.

---

## 19. Connected periodic regions

For every loading (v_k), define the numerical active support

[
\mathcal S_k
============

\left{
i:
|v_k(i)|>
\tau_{\mathrm{supp}}
\right}.
]

The implementation decomposes this support into connected components using the six toroidal neighbors

[
(i_1\pm1,i_2,i_3),
]

[
(i_1,i_2\pm1,i_3),
]

[
(i_1,i_2,i_3\pm1),
]

with modular arithmetic along all three axes.

Thus a support region crossing the end of an axis remains a single connected region.

For every connected region (R), the estimator records:

* number of cells;
* absolute loading mass;
* fraction of the component's total active (\ell_1) mass;
* loading (\ell_2) energy;
* fraction of the component's total active (\ell_2) energy.

Regions are ordered by decreasing number of cells.

The current implementation defines regions using

[
|v_k|>\tau_{\mathrm{supp}}.
]

It does not automatically split a connected region according to loading sign, and version 0.1.0 does not automatically generate axis-projection labels.

---

## 20. Effective regions

Small connected numerical islands can be filtered from the interpretive summary.

A connected region (R) of component (k) is classified as effective when either

[
\frac{|R|}
{N_k^{\mathrm{active}}}
\geq
\eta_{\mathrm{cells}}
]

or

[
\frac{
\sum_{i\in R}|v_k(i)|
}{
|v_k|*1
}
\geq
\eta*{\mathrm{mass}}.
]

The defaults are

```text
effective_region_min_active_fraction = 0.005
effective_region_min_l1_fraction = 0.01
```

corresponding to 0.5% of the active support or 1% of the component's L1 loading mass.

The estimator reports:

* all connected-region sizes;
* number of connected regions;
* number of effective regions;
* size of the largest region;
* fraction of active cells in the largest region;
* fraction of L1 mass in the largest region.

These region quantities are diagnostics derived directly from the fitted loadings; they do not alter the optimization objective.

---

## 21. Loading conditioning

For the active loading matrix

[
V_A,
]

the estimator monitors the Gram matrix

[
V_A^\top V_A.
]

It reports its singular values and condition number.

This diagnostic is especially relevant because CycleSPCA is non-orthogonal and therefore permits correlated loading directions.

A large condition number indicates poor identifiability or near linear dependence between components.

---

## 22. Computational implementation

The graph incidence matrix is not explicitly stored.

For a loading tensor, the forward graph operation is evaluated by cyclic shifts along each of the three axes, and its exact adjoint is evaluated by the corresponding reverse shifts.

The graph-working array has size

[
3MK.
]

The estimator can also operate on memory-mapped input matrices.

Centered products

[
X_cB
]

and

[
X_c^\top A
]

are evaluated in row batches, whose default size is

```text
batch_size = 2048.
```

The full centered data matrix therefore need not be stored separately in memory.

---

## 23. Reference algorithm

The default CycleSPCA fitting procedure implemented in version 0.1.0 can be summarized as follows.

1. Validate the data, calendar dimensions, and numerical parameters.
2. Construct the implicit three-axis toroidal graph.
3. Compute the training mean if `center=True`.
4. Initialize (U) and (V) with randomized truncated SVD by default.
5. Mark all (K) components active and initialize the graph dual variable to zero.
6. For each outer iteration:

   1. perform exact unit-sphere coordinate updates of the active score columns;
   2. compute the scheduled inner tolerance;
   3. compute (L_U=\lambda_{\max}(U^\top U));
   4. compute valid primal and dual step sizes;
   5. solve the complete convex loading subproblem using the primal-dual updates;
   6. roll back the loading update if its objective increased;
   7. detect collapsed or redundant components;
   8. reinitialize or deactivate flagged components;
   9. compute reconstruction, objective, conditioning, effective-rank, and solver diagnostics;
   10. stop if all outer convergence conditions are satisfied.
7. Apply the deterministic sign convention.
8. Compute final reconstruction, sparsity, graph-TV, region, contribution, conditioning, and score diagnostics.
9. Return the fitted training scores (U), loadings (V), active-component mask, and diagnostics.

---

## 24. Scope of version 0.1.0

Version 0.1.0 implements and validates the following specific estimator:

* a joint non-orthogonal rank-(K) factorization;
* unit-norm active training score columns;
* L1-penalized loadings;
* anisotropic graph total variation;
* a three-axis Cartesian product of cycle graphs;
* exact cyclic boundaries on every implemented axis;
* exact coordinate score updates by default;
* a Condat--Vu-type primal-dual solution of the convex loading subproblem;
* component reinitialization and effective-rank reduction;
* least-squares out-of-sample transformation;
* reconstruction-based explained variance;
* conditional leave-one-component-out reconstruction contributions;
* connected toroidal support-region diagnostics;
* batched centered matrix products; and
* deterministic component sign normalization.

The validated calendar configuration is

[
C_{24}\square C_7\square C_{52}.
]

The software does not currently implement an arbitrary number (P) of periodic axes: version 0.1.0 uses exactly three axes, although their lengths can be configured.

The model contains no loading ridge penalty, no mutual orthogonality constraint, no sequential deflation, and no automatic signed-region or semantic-label extraction.

The central modeling assumption is that feature adjacency is known in advance and is periodic along each of the three supplied axes.
