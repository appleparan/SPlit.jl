# Phase 2a — Gaussian Kernel Support Points

**Status**: Approved design, pre-implementation
**Date**: 2026-09-02
**Branch**: `feat/gaussian-kernel`
**Builds on**: `2026-09-02-paper-aligned-redesign-design.md` (Phase 1)

## TL;DR

Add `GaussianKernel` as the first non-energy `SplitKernel`. Support points
for it are found by minimizing the squared maximum mean discrepancy (MMD²)
with gradient descent and Armijo backtracking, so the objective is
non-increasing at every iteration — the same property the energy kernel's
MM step has, and the same test. Quality diagnostics gain an `mmd` estimator
and a `kernel` keyword on `splitquality`/`compare`. Public API grows by
`GaussianKernel` and `mmd`; nothing existing changes (v0.3.0, non-breaking).
Greedy kernel herding is deferred to Phase 2b as a separate splitter.

## Source of truth

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
  (2012). A Kernel Two-Sample Test. *JMLR*, 13, 723–773. — MMD definition
  and its biased (V-statistic) estimator.
- Mak & Joseph (2018) — support points as the energy-kernel special case;
  the Phase 1 MM path stays as is.

The objective and its gradient are standard consequences of the MMD
definition and are written out below so the implementation can be checked
line by line against them.

## Objective and gradient

Data rows $x_1, \dots, x_N$ (standardized by `preprocess`), support points
$\xi_1, \dots, \xi_n$, kernel $k$. Up to a constant independent of $\xi$,

```math
\mathrm{MMD}^2(\xi) = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} k(\xi_i, \xi_j)
  \;-\; \frac{2}{nN} \sum_{i=1}^{n} \sum_{l=1}^{N} k(\xi_i, x_l)
```

For the Gaussian kernel $k(u, v) = \exp\!\left(-\|u - v\|^2 / 2\sigma^2\right)$,
$\nabla_u k(u, v) = -k(u, v)\,(u - v)/\sigma^2$, so

```math
\nabla_{\xi_m} \mathrm{MMD}^2 = \frac{2}{n^2} \sum_{j \ne m} \nabla_u k(\xi_m, \xi_j)
  \;-\; \frac{2}{nN} \sum_{l=1}^{N} \nabla_u k(\xi_m, x_l)
```

(the $j = m$ term has zero gradient). The full gradient costs O(n(n+N)p)
per evaluation, the same order as one energy-kernel MM sweep.

## Architecture

### Kernel interface (`src/kernels.jl`)

```julia
struct GaussianKernel{B} <: SplitKernel
  bandwidth::B          # Float64, or :median
end
GaussianKernel() = GaussianKernel(:median)

kernelvalue(k::GaussianKernel, u, v)           -> Float64
kernelgrad!(g, k::GaussianKernel, u, v)        # g .= ∇_u k(u, v)
resolve(k::GaussianKernel, data, rng)          -> GaussianKernel{Float64}
```

- `resolve` turns `:median` into a number by the median heuristic: the
  median pairwise Euclidean distance over `min(N, 1_000)` rows sampled with
  `rng` (all rows when N ≤ 1_000). A numeric bandwidth resolves to itself.
  `bandwidth` must be positive.
- `EnergyKernel` keeps its Phase 1 MM path; it does not implement this
  interface (nothing needs it to).

### Optimizer (`src/optimizer.jl`)

`support_points(kernel::GaussianKernel, data, n; max_iterations, tolerance,
n_threads, rng, verbose)` — same keyword set as the energy method except:

- `kappa` other than `nothing` raises `ArgumentError` ("stochastic mode is
  not available for GaussianKernel yet"). Stochastic descent needs its own
  convergence design and lands with Phase 2b at the earliest.
- Initialization, bounds, jitter, chunked threading and the convergence rule
  (largest squared per-point displacement `< tolerance`) are shared with the
  energy path via the existing helpers.

Per iteration:

1. Evaluate `∇MMD²` for all points (chunked over `n_threads`; each chunk
   writes only its rows of the gradient matrix).
2. Armijo backtracking on the projected step: start from `t = 2·t_prev`
   (`t_0 = 1`), clamp `ξ_new = ξ − t∇` to the bounding box, and halve until
   `MMD²(ξ_new) ≤ MMD²(ξ) − c·⟨∇, ξ − ξ_new⟩` with `c = 1e-4`, at most 30
   halvings; if none succeeds, stop and report `converged = false`.
3. Clamp to the data bounding box (as in Phase 1).
4. Convergence check on the accepted step.

MMD² is evaluated exactly (block-wise, no n×N matrix) — it is needed for the
line search anyway, so `verbose` can print it.

### Splitter (`src/splitter.jl`)

`datasplit` resolves the kernel once (`resolve(kernel, X, rng)`) and stores
the resolved kernel in `SplitResult.method` so the σ actually used is
recoverable. `SupportPointSplitter` accepts any `SplitKernel`; validation
rejects `kappa !== nothing` with a `GaussianKernel` at construction time.

### Quality diagnostics (`src/quality.jl`)

```julia
mmd(X, Y, kernel; subsample = nothing, repeats = 8, rng)   # exported
splitquality(data, result; kernel = EnergyKernel(), kwargs...)
compare(methods, data; kernel = EnergyKernel(), kwargs...)
```

- `mmd` mirrors `energydistance`: block-wise exact V-statistic, optional
  subsampled estimate with the same positive `O(1/subsample)` bias caveat.
  With `kernel = EnergyKernel()` `splitquality` calls `energydistance`
  unchanged, so existing behavior and tests are untouched.
- A `:median` bandwidth in `mmd`/`splitquality` is resolved on the pooled
  rows of `X` and `Y` with the supplied `rng`.

## Testing (paper-property style)

1. Monotone descent: the MMD² trajectory of `support_points(GaussianKernel(1.0), …)`
   is non-increasing (slack 1e-10).
2. Optimality: a `GaussianKernel` split has lower `mmd` (same kernel) and
   lower `energydistance` than random splits of the same ratio (averaged
   over repetitions).
3. Reproducibility: same `rng` ⇒ same resolved σ and same indices,
   independent of `n_threads`.
4. Gradient correctness: finite-difference check of `∇MMD²` on a tiny
   problem (relative error < 1e-5).
5. `kappa` with `GaussianKernel` raises at construction.
6. `mmd`: identical samples → 0; shift increases it; subsampled estimate
   agrees with exact on a skewed split (as in the ED test).
7. `compare(…; kernel = GaussianKernel())` scores with MMD and `best` picks
   the minimum.

## Documentation (deliverable, not optional)

The Documenter site (`docs/src/`) gains a **Methods** page written with
`math` blocks, and every new public name gets a docstring with a worked
example:

- `docs/src/methods.md`: (1) the energy distance and the support-point MM
  update of Mak & Joseph (2018) as implemented in `optimizer.jl`, including
  the stochastic running-average variant; (2) the MMD objective, the
  Gaussian gradient above, the Armijo line search, and the median-heuristic
  bandwidth; (3) the sequential nearest-neighbor assignment; (4) the
  V-statistic estimators (`energydistance`, `mmd`) and their
  $O(1/\text{subsample})$ bias. Each formula names the function that
  implements it.
- `docs/src/95-reference.md`: `@docs` blocks for every exported name
  (`GaussianKernel`, `mmd` added).
- `README.md`: one paragraph and one snippet showing
  `SupportPointSplitter(kernel = GaussianKernel())` and
  `splitquality(data, result; kernel = GaussianKernel())`.
- `AGENTS.md`: one gotcha line for the Gaussian path (no `kappa`; bandwidth
  resolved at fit time and stored in `result.method`).

`julia --project=docs docs/make.jl` must build without warnings (Documenter
`:missing_docs` and `:docs_block` errors count as failures).

## Non-goals (Phase 2a)

- Kernel herding (Phase 2b: separate splitter type, selects rows directly).
- Stochastic (`kappa`) mode for non-energy kernels.
- Other kernels (Laplacian, IMQ) — the interface admits them; add on demand.
- Bandwidth learning beyond the median heuristic.

## Breaking changes

None. New exports: `GaussianKernel`, `mmd`. New keywords: `kernel` on
`splitquality` and `compare`. Version 0.2.0 → 0.3.0.
