# Method

This document specifies the mathematical model and computational conventions
implemented in `calendar-gfspca` version 0.1.0.

## 1. Data representation

Let

\[
X \in \mathbb{R}^{n \times M}
\]

contain \(n\) observations and \(M\) ordered calendar features. Suppose the
features are indexed by \(P\) periodic axes of lengths

\[
m_1,\ldots,m_P,
\qquad
M=\prod_{p=1}^{P}m_p.
\]

The configuration validated for annual hourly profiles is

\[
(m_1,m_2,m_3)=(24,7,52),
\qquad M=8736,
\]

with axes interpreted as hour of day, weekday, and week of year.

The estimator operates on the matrix supplied by the user. Data preprocessing
is not part of the mathematical objective below and must be kept fixed when
models or hyperparameters are compared.

## 2. Periodic calendar graph

The feature domain is represented by the Cartesian product of cycle graphs

\[
G=C_{m_1}\mathbin{\square}C_{m_2}
\mathbin{\square}\cdots\mathbin{\square}C_{m_P}.
\]

A vertex is a tuple

\[
i=(i_1,\ldots,i_P),
\qquad 0\leq i_p<m_p.
\]

For every axis \(p\), the graph contains the edge

\[
i \longleftrightarrow
(i_1,\ldots,(i_p+1)\bmod m_p,\ldots,i_P).
\]

Thus the graph contains all ordinary local neighbor relations and closes every
axis periodically. In the \(24\times7\times52\) case, this includes:

- 23:00--00:00 hour adjacency;
- Sunday--Monday weekday adjacency; and
- week 52--week 1 seasonal adjacency.

Each vertex has two neighbors per axis. When every \(m_p>2\), the undirected
graph has

\[
|E|=P M
\]

edges. Hence the validated graph has

\[
3(24\cdot7\cdot52)=26\,208
\]

edges.

Choose an arbitrary but fixed orientation for every edge and let

\[
D_G\in\mathbb{R}^{|E|\times M}
\]

be the corresponding incidence matrix. For a loading \(v\), each entry of
\(D_Gv\) is the signed difference between two adjacent calendar positions.
Because the penalty uses an L1 norm, reversing an edge orientation does not
change the objective.

For a Cartesian product of graphs, the Laplacian eigenvalues add. In the
validated configuration,

\[
\lVert D_G\rVert_2^2
=\lambda_{\max}(D_G^\top D_G)
=4+4+\left(2+2\cos\frac{\pi}{7}\right)
=10+2\cos\frac{\pi}{7}
\approx 11.80194.
\]

This value is used when selecting valid primal-dual step sizes.

## 3. Factorization objective

For \(K\) components, let

\[
U=[u_1,\ldots,u_K]\in\mathbb{R}^{n\times K},
\qquad
V=[v_1,\ldots,v_K]\in\mathbb{R}^{M\times K}.
\]

The model minimizes

\[
\mathcal{J}(U,V)
=
\frac{1}{2}\lVert X-U V^\top\rVert_F^2
+\lambda_1\sum_{k=1}^{K}\lVert v_k\rVert_1
+\lambda_{\mathrm{TV}}\sum_{k=1}^{K}
\lVert D_Gv_k\rVert_1,
\]

subject to the component-normalization convention used by the implementation,
in particular

\[
\lVert u_k\rVert_2=1
\]

for every nondegenerate component.

The two penalties have different roles:

- \(\lambda_1\lVert v_k\rVert_1\) encourages exact zeros and therefore sparse
  calendar support;
- \(\lambda_{\mathrm{TV}}\lVert D_Gv_k\rVert_1\) encourages neighboring
  calendar positions to share the same loading value and therefore produces
  piecewise-coherent regions on the periodic domain.

Setting \(\lambda_{\mathrm{TV}}=0\) removes graph fusion. Setting
\(\lambda_1=0\) removes direct loading sparsity while retaining graph total
variation.

The objective is not jointly convex in \(U\) and \(V\), although each block
subproblem is convex when the other block is fixed. The implementation
therefore seeks a stationary solution through alternating optimization rather
than a guaranteed global optimum.

## 4. Alternating optimization

### 4.1 Initialization

The loading and score factors are initialized using a PCA-based configuration.
A fixed random seed should be used whenever an initialization step can be
randomized. Initialization, preprocessing, and hyperparameters must be held
constant in controlled method comparisons.

### 4.2 Score update

For fixed \(V\), the unconstrained least-squares score estimate is

\[
\widetilde U
=X V(V^\top V)^\dagger,
\]

where \({}^\dagger\) denotes the Moore--Penrose pseudoinverse. The
implementation checks for degenerate columns before normalization and applies
its component scaling convention to obtain \(U\).

The condition number of \(V^\top V\) is monitored because a large value signals
nearly redundant or poorly identified loading directions even when no column
has collapsed exactly to zero.

### 4.3 Loading update

For fixed \(U\), the loading subproblem is

\[
\min_V\;
\frac{1}{2}\lVert X-U V^\top\rVert_F^2
+\lambda_1\lVert V\rVert_{1,1}
+\lambda_{\mathrm{TV}}\lVert D_GV\rVert_{1,1}.
\]

The smooth reconstruction term has gradient

\[
\nabla_V
\frac{1}{2}\lVert X-U V^\top\rVert_F^2
=V(U^\top U)-X^\top U.
\]

The implementation solves this composite convex subproblem using primal-dual
three-operator splitting (PD3O). The L1 proximal operator is elementwise
soft-thresholding. The graph term is handled through the incidence operator
\(D_G\) and its transpose. Step sizes are chosen to satisfy the numerical
conditions associated with the smooth-gradient Lipschitz constant and
\lVert D_G\rVert_2^2.

The inner accuracy may follow a decreasing schedule

\[
\varepsilon_t
=\max\{\varepsilon_{\min},\varepsilon_0\rho^t\},
\qquad 0<\rho<1,
\]

so early outer iterations do not oversolve an inaccurate subproblem while later
iterations receive a tighter loading solution.

## 5. Convergence and diagnostics

The implementation records optimization diagnostics and stops when its outer
convergence criterion is met or the configured iteration limit is reached.
Relevant diagnostics include:

- objective history;
- outer and inner iteration counts;
- convergence status;
- reconstruction error;
- sparsity and graph-fusion measures;
- checks for zero or non-finite components; and
- the condition number of \(V^\top V\).

For large empirical fits, convergence should be assessed from the recorded
diagnostics rather than inferred solely from reaching a preselected number of
iterations.

Because the overall problem is non-convex, results can depend on initialization
and regularization. Stability across resamples, nearby hyperparameters, or
repeated initializations is an empirical property to be measured.

## 6. Scores, reconstruction, and embedding

After fitting the loading matrix \(V\), new observations

\[
X_{\mathrm{new}}\in\mathbb{R}^{n_{\mathrm{new}}\times M}
\]

are mapped to component scores using the fitted least-squares transformation
and the estimator's stored scaling convention. These scores form a
\(K\)-dimensional embedding.

Reconstruction is obtained from

\[
\widehat X=U V^\top.
\]

The embedding may be used for visualization, clustering, or downstream
prediction, provided evaluation is performed out of sample. When several rows
belong to the same physical entity, train/test splitting should be performed by
entity rather than by row to prevent information leakage.

## 7. Identifiability and comparison of solutions

As in other matrix factorizations, component signs and order are not
identifiable:

\[
u_kv_k^\top=(-u_k)(-v_k)^\top.
\]

Consequently, two valid fits must not be compared by raw column position alone.
Components should first be paired, for example by maximum absolute cosine
similarity or Hungarian matching, and their signs should then be aligned.

Nearly collinear components are also possible. The condition number diagnostic
and stability analysis help detect this situation.

## 8. Scope of version 0.1.0

Version 0.1.0 freezes the validated research implementation of:

- the product-of-cycles calendar graph;
- L1-sparse and graph-fused PCA loadings;
- alternating score/loading optimization;
- PD3O loading updates; and
- numerical diagnostics and stability checks.

The release does not claim that adding graph total variation to sparse PCA is,
by itself, a new general principle. Its distinctive modeling choice is the
explicit use of several periodic axes and all their cyclic boundaries. The
scientific benefit of that choice must be established through controlled
comparisons with non-cyclic alternatives.

