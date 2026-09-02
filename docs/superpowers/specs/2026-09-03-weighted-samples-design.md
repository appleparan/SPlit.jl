# Weighted samples (roadmap M1)

**Status**: Approved design, pre-implementation
**Date**: 2026-09-03
**Branch**: `feat/weighted-samples`
**Roadmap**: M1 on the Roadmap page (`docs/src/85-roadmap.md`)

## TL;DR

Add per-row sample weights to SPlit.jl and splitiq. A `weights` vector turns
the data into the weighted empirical measure `Σ wᵢ δ(xᵢ)`; every splitter
then chooses its (uniformly weighted) subset to approximate that measure,
and every discrepancy accepts weights on each side. Existing code paths are
untouched: `weights = nothing` (the default) dispatches to the current
methods, so unweighted results are bit-identical. Weighted behavior is added
as new methods, and documented as one new section per page rather than by
rewriting the existing explanations.

Decisions taken with the user on 2026-09-03:

1. Weights are a keyword beside the data (`datasplit(s, data; weights)`),
   not a splitter field.
2. The rule for combining weights with `kappa` subsampling is chosen by a
   benchmark in this PR (two rules implemented, one becomes the default).
3. Preprocessing and the `:median` bandwidth use the weights when given.

## Semantics

Let `w ∈ ℝ^N`, `wᵢ ≥ 0`, `Σ wᵢ > 0`, and write `w̄ᵢ = wᵢ / Σ wₗ` for the
normalized weights. The data distribution is `P_w = Σ w̄ᵢ δ(xᵢ)`.

- A splitter selects `n` rows so that the uniform measure on the selected
  rows is close to `P_w`. The rule that decides which side is `test` (the
  smaller side when `ratio ≤ 0.5`) is unchanged.
- `energydistance(X, Y; weights_x, weights_y)` and
  `mmd(X, Y, kernel; weights_x, weights_y)` compare `P_{w}` on `X` with
  `P_{v}` on `Y`; either side may be `nothing` (uniform).
- `splitquality(data, result; weights)` compares the weighted train rows with
  the weighted test rows, each renormalized. This keeps its meaning
  ("discrepancy between train and test under the same preprocessing"). The
  quantity a splitter minimizes, uniform selected rows against `P_w`, is
  available directly as `energydistance(X[test, :], X; weights_y = w)`.
- Weights proportional to duplication counts are equivalent to duplicating
  rows. This is the invariance the tests use throughout.

Validation (shared helper `_check_weights(w, N)`): length `N`, all finite,
none negative, sum positive; otherwise `ArgumentError`. Zero weights are
allowed; such a row contributes nothing to the objective but may still be
selected by `select_nearest`, which is a rounding step, not an objective.

## Formulas (weighted deltas only)

Unweighted formulas are on the Methods page and stay as written. With
normalized weights `w̄` on `X` and `v̄` on `Y`:

Energy distance

```math
\mathrm{ED}_{w,v}(X, Y) = 2 \sum_{i,j} \bar w_i \bar v_j \|x_i - y_j\|
  - \sum_{i,k} \bar w_i \bar w_k \|x_i - x_k\|
  - \sum_{j,l} \bar v_j \bar v_l \|y_j - y_l\|
```

Squared MMD

```math
\mathrm{MMD}^2_{w,v}(X, Y) = \sum_{i,k} \bar w_i \bar w_k k(x_i, x_k)
  + \sum_{j,l} \bar v_j \bar v_l k(y_j, y_l)
  - 2 \sum_{i,j} \bar w_i \bar v_j k(x_i, y_j)
```

Support points, energy kernel. Objective and MM update (Mak & Joseph 2018,
Theorem 3 with the empirical measure replaced by `P_w`; the majorizer is the
same quadratic bound applied term by term, so monotone descent holds for any
non-negative weights):

```math
E(\xi) = \frac{2}{n} \sum_{m} \sum_{i} \bar w_i \|\xi_m - x_i\|
  - \frac{1}{n^2} \sum_{m,o} \|\xi_m - \xi_o\|
```

```math
\xi_m^{\text{new}} = \frac{\frac{1}{n} \sum_{o \ne m} \frac{\xi_m - \xi_o}{\|\xi_m - \xi_o\|}
  + \sum_i \bar w_i \frac{x_i}{\|x_i - \xi_m\|}}
  {\sum_i \frac{\bar w_i}{\|x_i - \xi_m\|}}
```

Implementation scales the weights to mean one, `ŵᵢ = N w̄ᵢ`, so the update
is the existing one with `ŵᵢ` multiplying the two data sums: the `(N/n)`
factor on the point-repulsion term stays as it is. In stochastic mode the
scaling is over the subsample, `ŵᵢ = κ wᵢ / Σ_{l ∈ sub} wₗ`.

Support points, Gaussian kernel. Objective (up to the constant data
self-term) and gradient row `m`:

```math
f(\xi) = \frac{1}{n^2} \sum_{m,o} k(\xi_m, \xi_o)
  - \frac{2}{n} \sum_m \sum_l \bar w_l\, k(\xi_m, x_l)
```

```math
\nabla_{\xi_m} f = \frac{2}{n^2} \sum_{o \ne m} \nabla k(\xi_m, \xi_o)
  - \frac{2}{n} \sum_l \bar w_l\, \nabla k(\xi_m, x_l)
```

Same `ŵ` trick: the `2/(nN)` factor stays and `ŵₗ` multiplies each data
term. Armijo backtracking, the scale-aware first step, and the convergence
rules are unchanged.

Herding. The data term becomes `dᵢ = Σₗ w̄ₗ k(xᵢ, xₗ)`; the selected-set
term and the greedy rule (Chen, Welling & Smola 2010, Eq. 8) are unchanged.

Estimators. Each existing `_energydistance` / `_mmd` method gets a weighted
counterpart (a new method with `weights_x, weights_y` positional arguments,
never an `if` inside the existing one):

| estimator        | weighted form                                                                 |
|------------------|-------------------------------------------------------------------------------|
| `Exact`          | block-wise `w_blkᵀ D v_blk` instead of `sum(D)`, with no `1/(nm)` division since the weights are normalized |
| `Subsample`      | rows drawn uniformly as now; weights renormalized within each subsample; exact weighted statistic averaged over `repeats` |
| `RandomSlices`   | weighted 1-D energy distance per direction from sorted samples and prefix sums of `w` and `w·a` (derivation below) |
| `RandomFeatures` | weighted feature means `Σ w̄ᵢ z(xᵢ)`; `‖z̄_w(X) − z̄_v(Y)‖²`               |

Weighted 1-D energy distance. For a sorted sample `a₍₁₎ ≤ … ≤ a₍ₙ₎` with
weights `w₍ᵢ₎`, prefix sums `Wᵢ = Σ_{l ≤ i} w₍ₗ₎` and `Aᵢ = Σ_{l ≤ i} w₍ₗ₎ a₍ₗ₎`:

```math
\sum_{i,k} w_i w_k |a_i - a_k| = 2 \sum_{k} w_{(k)} \left( a_{(k)} W_{k-1} - A_{k-1} \right)
```

and for the cross term with `r(b) = #\{i : a₍ᵢ₎ ≤ b\}`:

```math
\sum_{i,j} w_i v_j |a_i - b_j| = \sum_j v_j \left[ (b_j W_{r} - A_{r}) + (A_n - A_{r}) - b_j (W_n - W_{r}) \right]
```

Both reduce to the existing `_within_mean_abs` / `_cross_mean_abs` when
every weight is `1/n`.

Preprocessing. `preprocess(data, weights)` is a new method: the same
Helmert encoding and constant-column rule, then weighted standardization
with `μⱼ = Σ w̄ᵢ xᵢⱼ` and `σⱼ² = Σ w̄ᵢ (xᵢⱼ − μⱼ)²`. The unweighted method is
unchanged (it keeps `std`'s `n − 1` denominator). `resolve(kernel, X, rng,
weights)` draws the `MEDIAN_HEURISTIC_ROWS` rows for the median heuristic
with probability proportional to `w` (without replacement) and is otherwise
unchanged.

## Weighted `kappa` subsampling: benchmark

Two rules, both implemented behind an internal keyword of
`support_points(::EnergyKernel, …)` (like `_n0_factor`, not exposed on
`SupportPointSplitter`):

- `:uniform`: draw `κ` rows uniformly without replacement, as now, and
  rescale their weights to mean one within the subsample.
- `:proportional`: draw `κ` rows without replacement with probability
  proportional to `w` (`StatsBase.sample(rng, 1:N, Weights(w), κ;
  replace = false)`) and treat the subsample as uniform (`ŵ ≡ 1`).

Benchmark script `benchmark/weighted_kappa.jl`, reusing `benchmark/datasets.jl`:
`normal-10d` and `uniform-5d` at `N = 10_000`, two weight profiles
(log-normal weights with `σ_log = 1`, and a 10:1 two-cluster profile on
`normal-10d` split by the sign of the first coordinate), `κ ∈ {500, 2_000}`,
five seeds each. Metric: weighted energy distance of the selected rows
(uniform) against the full data under `w`, plus wall time. The rule with the
lower mean discrepancy at `κ = 500` becomes the default; the table goes to
`docs/src/assets/benchmarks/weighted_kappa.md` and a short section on the
Design experiments page records the choice. If the two rules are within one
standard error of each other, `:uniform` wins for being the simpler rule.

## API

Julia (all new keywords default to `nothing`, preserving every existing
signature):

```julia
datasplit(s::AbstractSplitter, data; weights = nothing)
splitquality(data, result; weights = nothing, kwargs...)
compare(methods, data; weights = nothing, kwargs...)
energydistance(X, Y; weights_x = nothing, weights_y = nothing, kwargs...)
mmd(X, Y, kernel; weights_x = nothing, weights_y = nothing, kwargs...)
support_points(kernel, data, n; weights = nothing, kwargs...)   # internal
herd(kernel, X, n; weights = nothing, n_threads)                # internal
preprocess(data, weights)                                       # internal
```

`SplitResult` is unchanged. `compare` forwards `weights` to both `datasplit`
and `splitquality`.

Python (splitiq), mirroring the names:

```python
datasplit(data, ratio, *, weights=None, ...)
splitquality(data, result, *, weights=None, ...)
energydistance(x, y, *, weights_x=None, weights_y=None, ...)
mmd(x, y, kernel, *, weights_x=None, weights_y=None, ...)
```

`weights` is any 1-D array-like; `_convert.to_weights` produces a
contiguous `float64` vector and rejects anything else with `ValueError`.
Julia-side `ArgumentError`s pass through `_translate_error` as now.

## Tests

Julia, added to the existing files (no existing test is edited):

- `test_quality.jl`: uniform weights give exactly the unweighted value
  (`==`); duplication invariance (`weights_x = [2, 1, 1]` equals the
  unweighted value on the data with row 1 duplicated) for `Exact`,
  `Subsample` (same `rng`, exact-path sizes), `RandomSlices` (same `rng`,
  same directions), `RandomFeatures` (same `rng`); validation errors.
- `test_estimators.jl`: weighted 1-D energy distance against a brute-force
  double loop.
- `test_optimizer.jl`: `weights = nothing` and `weights = ones(N)` produce
  identical points and iteration counts; weighted full-data MM descent is
  monotone (`_objective_trajectory` with weights); a two-cluster dataset
  with 9:1 weights places more support points in the heavy cluster than the
  unweighted run does (both kernels); one `_mm_sweep!` on weighted data
  matches one sweep on the duplicated data from the same starting points.
- `test_herding.jl`: duplication invariance of the selected rows; heavy
  cluster attracts more selections.
- `test_splitter.jl`: `datasplit(s, data; weights = ones(N))` equals
  `datasplit(s, data)` for both splitters (same `rng` seed); `splitquality`
  and `compare` accept `weights`; wrong length errors.
- `test_preprocessing.jl`: weighted standardization has weighted mean 0 and
  weighted variance 1; uniform weights match the unweighted result up to the
  `n − 1` factor.
- `test_properties.jl`: weighted split beats random splits under the
  weighted energy distance on a skewed-weight dataset.

Python (`splitiq/tests`): parity tests for each new keyword (uniform weights
reproduce the unweighted result; duplication invariance for
`energydistance`; a weighted split runs and returns valid indices; bad
weights raise `ValueError`).

## Docs

Nothing existing is rewritten. Additions:

- Methods page: a new section "Weighted samples" stating the definitions
  above and which steps of the procedure read the weights.
- Design experiments page: the `kappa` rule benchmark and the decision.
- Roadmap page: M1 status moves to done, with the `kappa` open question
  resolved in place.
- Python page and splitiq docs: the new keywords.
- `AGENTS.md`: a Workflow bullet stating that every capability exposed in
  Julia must be exposed in splitiq in the same change, with tests.

## Non-goals

- A reference distribution other than the data itself (M2).
- Weights on the selected subset (the output is always uniform).
- Exposing the `kappa` sampling rule on `SupportPointSplitter`.

## References

- Mak, S., & Joseph, V. R. (2018). Support points. *Annals of Statistics*,
  46(6A). MM update; the weighted majorizer is the same bound.
- Joseph, V. R., & Vakayil, A. (2021). SPlit. *Technometrics*, 63(4).
  Stochastic MM.
- Chen, Y., Welling, M., & Smola, A. (2010). Super-samples from kernel
  herding. *UAI*. Eq. 8.
- Gretton, A., et al. (2012). A kernel two-sample test. *JMLR*, 13. MMD.
