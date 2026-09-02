# Phase 3 — Scalable Estimators and a Scale-Aware Optimizer

**Status**: Approved design, pre-implementation
**Date**: 2026-09-02
**Branch**: `feat/scalable-estimators`
**Builds on**: Phases 1, 2a, 2b (`2026-09-02-*-design.md`)

## TL;DR

Three changes, one vocabulary. (A) The Gaussian-kernel support-point optimizer
gets a scale-aware first step and a minimum iteration count so it no longer
reports convergence at the initial sample. (B) and (C) introduce a
`DiscrepancyEstimator` type hierarchy — `Exact`, `Subsample`, `Slices`,
`RandomFeatures` — selected by an `estimator` keyword on `energydistance`,
`mmd`, `splitquality`, and `HerdingSplitter`; which combinations exist is
expressed by method dispatch, not runtime checks. `Slices` estimates the
energy distance (and herding's energy data term) from random 1-D projections
in $O(kN\log N)$; `RandomFeatures` estimates Gaussian MMD (and herding's
Gaussian data term) with random Fourier features in $O(NDp)$. Both are
candidate-symmetric, so herding regains a large-$N$ mode without the bias
that removed `kappa` in Phase 2b. Exact estimators are threaded and the exact
threshold of `splitquality` rises; the automatic fallback above it is chosen
by a recorded selection experiment. Non-breaking, v0.5.0.

## Source of truth

- Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel
  Machines. *NIPS 20*. — random Fourier features for shift-invariant kernels.
- Székely, G. J., & Rizzo, M. L. (2013). Energy statistics. *JSPI* 143(8). —
  energy distance; Gretton et al. (2012) — MMD; Chen, Welling & Smola (2010)
  — herding; Mak & Joseph (2018) — support points.
- The projection identity used by `Slices` is elementary and derived below.

## Estimator vocabulary (B, C)

```julia
abstract type DiscrepancyEstimator end
struct Exact          <: DiscrepancyEstimator end
struct Subsample      <: DiscrepancyEstimator; m::Int; repeats::Int; end
struct Slices         <: DiscrepancyEstimator; k::Int; end
struct RandomFeatures <: DiscrepancyEstimator; D::Int; end
```

Defined combinations (every other combination is a `MethodError` turned into
an `ArgumentError` by one fallback method):

| estimator        | `energydistance` / `EnergyKernel` | `mmd` / `GaussianKernel` | herding data term |
|------------------|-----------------------------------|--------------------------|-------------------|
| `Exact`          | yes (threaded)                    | yes (threaded)           | yes (threaded)    |
| `Subsample`      | yes (existing)                    | yes (existing)           | no                |
| `Slices`         | yes                               | no                       | `EnergyKernel`    |
| `RandomFeatures` | no                                | yes                      | `GaussianKernel`  |

Keyword form: `energydistance(X, Y; estimator = Exact(), rng)`,
`mmd(X, Y, kernel; estimator = Exact(), rng)`,
`splitquality(data, result; kernel, estimator = nothing, rng)` — `nothing`
means the automatic rule below —, `HerdingSplitter(; kernel, estimator =
Exact(), ratio, n_threads, rng)`. The existing `subsample = m, repeats = r`
keywords stay as a thin compatibility path mapping to `Subsample(m, r)`.
Estimator objects hold specifications only; random draws (directions,
features) come from the call's `rng`, so results are reproducible and the
objects are cheap, immutable, and printable.

### Slices — the projection identity

For $\theta$ uniform on the unit sphere $S^{p-1}$ and any $u \in \mathbb{R}^p$,

```math
\mathbb{E}_\theta\,|\langle \theta, u \rangle| = \kappa_p \|u\|, \qquad
\kappa_p = \frac{\Gamma(p/2)}{\sqrt{\pi}\,\Gamma\!\left((p+1)/2\right)}
```

($\kappa_2 = 2/\pi$, $\kappa_3 = 1/2$). The energy distance is linear in the
pairwise norms, so with $u^\theta = X\theta$, $v^\theta = Y\theta$,

```math
\mathrm{ED}(X, Y) = \kappa_p^{-1}\, \mathbb{E}_\theta\, \mathrm{ED}_1(u^\theta, v^\theta),
```

an unbiased estimator is the average over $k$ directions drawn with `rng`.
The one-dimensional energy distance is computed exactly in $O((n+m)\log(n+m))$
from sorted projections: for a sorted sample $a_{(1)} \le \dots \le a_{(n)}$,
$\sum_{i<j} (a_{(j)} - a_{(i)}) = \sum_i (2i - n - 1)\, a_{(i)}$ gives the
within-sample mean, and the cross term $\sum_{i,j} |a_i - b_j|$ follows from
prefix sums of one sorted sample and the ranks of the other. Same $\theta$ for
all three terms, so their Monte-Carlo errors partially cancel. Cost
$O(k (n+m) \log(n+m) + k(n+m)p)$.

Herding data term under `EnergyKernel`: $d_i = -\frac{1}{N}\sum_l \|x_i - x_l\|
\approx -\frac{1}{k\kappa_p N}\sum_\theta \sum_l |u^\theta_i - u^\theta_l|$, and
with the projections sorted and prefix-summed ($P_r = \sum_{l \le r} u_{(l)}$),
a point of rank $r$ has $\sum_l |u_i - u_l| = u_i(2r - N) - 2P_r + P_N$ —
$O(N\log N)$ per direction, every candidate treated identically.

### RandomFeatures — random Fourier features

For the Gaussian kernel $k(x,y) = \exp(-\|x-y\|^2/2\sigma^2)$, with
$\omega_j \sim \mathcal{N}(0, \sigma^{-2} I_p)$ and $b_j \sim U[0, 2\pi]$
(Rahimi & Recht 2007),

```math
z(x) = \sqrt{2/D}\,\big[\cos(\omega_j^\top x + b_j)\big]_{j=1}^{D}, \qquad
\mathbb{E}\big[z(x)^\top z(y)\big] = k(x, y).
```

Hence $\|\bar z_X - \bar z_Y\|^2$ with $\bar z_X = \frac{1}{n}\sum_i z(x_i)$ is
an unbiased estimator of the V-statistic $\mathrm{MMD}^2(X, Y)$, cost
$O((n+m)Dp)$; and the herding data term $d_i = \frac1N\sum_l k(x_i, x_l)
\approx z(x_i)^\top \bar z$ is unbiased for every candidate. The drawn
$(\omega, b)$ live in an internal callable `FourierFeatureMap` created once
per call from `rng` (the `resolve` pattern of Phase 2a); `RandomFeatures(D)`
itself stores only $D$.

### Exact — threaded

`_mean_pairwise` and `_mean_kernel` split their outer block loop across
`n_threads` tasks with one accumulator per task, summed in a fixed order, so
results are identical for every thread count. `_data_term(::Exact, …)` uses a
column-major `permutedims(X)` copy for contiguous row access.

### Automatic rule for `splitquality`

`estimator = nothing` selects `Exact()` when the total row count is at most
`exact_threshold` (new default **20,000**, up from 4,000 — exact evaluation at
that size takes under a second on 16 threads), and otherwise the **fallback
chosen by the selection experiment** below (`Slices(k)` for the energy kernel,
`RandomFeatures(D)` for Gaussian kernels, or `Subsample(2_000, 8)` if the
experiment does not justify the change). The chosen fallback and its
parameters are constants in `quality.jl` with a comment citing the experiment.

### Selection experiment (recorded on the Benchmarks page)

On the four Phase 2b datasets at $N = 10{,}000$ with the exact value as
reference: absolute error and wall time of `Subsample(2000, 8)`,
`Slices(64)`, `Slices(256)`, `Slices(1024)`, `RandomFeatures(512)`,
`RandomFeatures(2048)` for the split produced by `support points · energy`
and `herding · energy` (ED) and `herding · gaussian` (MMD), over 5 rng seeds
(mean and max error). Decision rule: an estimator becomes the automatic
fallback if, at equal or lower wall time, its max error is at most one third
of `Subsample`'s. The table, the rule, and the decision go on
`docs/src/20-benchmarks.md` under "Estimators".

## Scale-aware Gaussian optimizer (A)

Diagnosis (Phase 2b review): the objective carries $1/n^2$ and $1/(nN)$
factors, so gradient row norms are $O(10^{-6})$ at $N = 10^4$ and the first
squared displacement falls below the absolute `tolerance = 10^{-10}` — the
optimizer reports convergence at the initial sample although further steps
decrease the objective.

Changes to `support_points(::GaussianKernel, …)`:

1. **Scale-aware first step**: $t_0 = 0.1\,\bar w / \max_m \|\nabla_m f\|$,
   where $\bar w$ is the median per-dimension data range, so the first trial
   move is a tenth of the data scale regardless of $n$, $N$; Armijo
   backtracking and the $2t$ warm start are unchanged.
2. **No convergence before the second accepted step**, and convergence when
   *either* the largest squared displacement is below `tolerance` *or* the
   relative objective decrease $|f_{t-1} - f_t| / \max(|f_t|, 10^{-12})$ is
   below `rtol = 10^{-8}` (new keyword, default as stated). `converged`
   remains honest (a failed line search still reports `false`).

Tests: on `normal-10d` at $N = 10^4$ the optimizer runs more than one
iteration and lowers the objective below its initial value; monotone descent,
reproducibility, and gradient tests unchanged. The benchmark page's
early-stop disclosure is rewritten once the new numbers are in.

## Documentation (deliverable)

- Methods page: "Estimators" section with the projection identity (with
  $\kappa_p$ and the sorted-sample formulas), random Fourier features, the
  threaded exact path, and the herding data terms; the optimizer section
  updated for the scale-aware step and the two-part convergence rule. Every
  formula names its function.
- Benchmarks page: "Estimators" table and decision; re-run of the existing
  benchmark with the new optimizer (support points · gaussian cells change).
- README/index: `estimator` keyword and `HerdingSplitter(estimator = …)`
  snippets; AGENTS.md gotchas for the estimator dispatch contract and the new
  convergence rule.

## Testing (property style)

1. Slices: $\mathbb{E}_\theta|\langle\theta,u\rangle| = \kappa_p\|u\|$ checked
   numerically for $p \in \{2, 3, 10\}$; sliced ED converges to exact ED as
   $k$ grows (error decreasing over $k \in \{16, 64, 256\}$, seeded); 1-D ED
   from the sorted formulas equals the pairwise definition on small data.
2. RandomFeatures: $z(x)^\top z(y) \to k(x,y)$ as $D$ grows; RFF MMD converges
   to exact MMD; feature map reproducible under `rng`.
3. Herding with `Slices`/`RandomFeatures`: data term converges to the exact
   data term; selections not concentrated (no subset bias); approximate
   herding beats random and is within a factor 2 of exact herding on ED/MMD
   at $N = 2{,}000$; an $N = 10^5$ smoke run finishes.
4. Dispatch: undefined combinations raise `ArgumentError` with both names;
   compatibility keywords map to `Subsample`.
5. Threaded exact estimators are bit-identical across `n_threads`.
6. Optimizer: the `normal-10d`-style regression test above; existing tests.
7. `splitquality` automatic rule picks `Exact` at $\le 20{,}000$ rows and the
   chosen fallback above.

## Non-goals

- `Slices` for Gaussian MMD, `RandomFeatures` for the energy kernel.
- Learning bandwidths or feature counts.
- $N = 10^5$ benchmark tables (smoke test only).

## Breaking changes

None. New exports: `DiscrepancyEstimator`, `Exact`, `Subsample`, `Slices`,
`RandomFeatures`. `splitquality`'s default exact threshold changes from 4,000
to 20,000 (more exact results, no API change). Version 0.4.0 → 0.5.0.
