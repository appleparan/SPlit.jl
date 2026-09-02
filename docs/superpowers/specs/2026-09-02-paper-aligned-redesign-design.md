# SPlit.jl Paper-Aligned Redesign

**Status**: Approved design, pre-implementation
**Date**: 2026-09-02
**Branch**: `refactor/paper-aligned-core`

## TL;DR

Rebuild SPlit.jl around a single kernel-centric API whose correctness is defined
by the published papers. Phase 1 collapses the current dual API into one
implementation, fixes placeholder convergence reporting, makes split-quality
diagnostics safe at large n via subsampled estimation, and replaces brute-force
nearest-neighbor subsampling with a k-d tree. Phases 2–3 extend the package
beyond the original method: MMD kernels and large-scale estimators. All legacy
API is deleted (v0.1.0, unregistered — breaking changes are free).

## Source of truth

Correctness is judged against the papers, not any prior implementation:

1. Mak, S., & Joseph, V. R. (2018). Support points. *Annals of Statistics*,
   46(6A), 2562–2592. — MM update, energy-distance objective.
2. Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data
   Splitting. *Technometrics*, 63(4), 492–502. — splitting procedure,
   stochastic MM for large n, nearest-neighbor subsampling.
3. Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Stat. Anal. Data
   Mining*, 15(4), 537–546. — optimal test ratio γ = 1/(√p + 1).

Tests encode properties guaranteed by these papers (see Testing) rather than
matching outputs of other implementations.

## Problems being fixed

- Two parallel APIs (`split_data` vs `SupportPointSplitter`/`datasplit`) with
  three overlapping preprocessing paths and two quality modules; they disagree
  on constant-column detection and on `kappa` semantics (multiplier vs
  absolute size).
- `SplitResult.convergence`/`iterations` are hardcoded placeholders
  (`interface.jl`), so results always misreport optimization status.
- The optimizer accepts a `metric` parameter but the MM update is derived only
  for the Euclidean energy distance; the parameter silently does nothing in
  optimization. Passing `EnergyDistance` as a point-to-point metric is a no-op
  relative to Euclidean (between two points it reduces to `2‖x−y‖`).
- Quality assessment materializes full n×n pairwise-distance matrices —
  O(n²) memory; unusable beyond n ≈ 10⁴. It also indexes the raw input, which
  breaks for `DataFrame`s (no `adjoint`) and categorical columns.
- Subsampling is O(k·n) brute force; the paper's procedure is a sequential
  nearest-neighbor query, efficiently served by a k-d tree.
- `optimal_split_ratio`'s parameter-count estimate (`√(unique rows)`) has no
  basis in Joseph (2022), and the regression method is unimplemented.
- Reproducibility: a hardcoded `Random.seed!(rng, 42)` clobbers the caller's
  RNG; stochastic sampling ignores the configured RNG; `n_threads` is accepted
  but not enforced; progress output is unconditional.
- `Base.summary(::SplitComparison)` returns a `DataFrame`, violating
  `summary`'s `String` contract.

## Goals

- One implementation, one preprocessing path, one public API.
- Kernel-centric type skeleton so Phase 2 (MMD kernels) adds methods without
  breaking the API.
- Honest convergence reporting; RNG threaded through every random draw;
  `n_threads` actually bounds parallelism; quiet by default (`verbose` opt-in).
- Split-quality diagnostics that stay correct and bounded in memory at large n.
- k-d tree subsampling.
- `optimal_split_ratio` implemented from Joseph (2022).

## Non-goals (Phase 1)

- MLJ.jl integration.
- GPU support.
- Any kernel other than `EnergyKernel` (the skeleton exists; implementations
  land in Phase 2).

## Architecture (Phase 1)

### Module layout

```text
src/
├── SPlit.jl          # module definition, exports
├── kernels.jl        # SplitKernel abstract type, EnergyKernel
├── preprocessing.jl  # single path: Matrix/DataFrame, Helmert encoding,
│                     # constant-column removal, standardization
├── support_points.jl # MM optimizer, dispatched on kernel
├── subsampling.jl    # k-d tree sequential nearest-neighbor (NearestNeighbors.jl)
├── splitter.jl       # SupportPointSplitter, SplitResult, datasplit
├── quality.jl        # energy-distance estimators, splitquality
├── ratio.jl          # optimal_split_ratio (Joseph 2022)
└── comparison.jl     # compare, SplitComparison
```

Deleted: `main.jl`, `interface.jl`, `types.jl`, `energy_distance.jl`,
`split_quality.jl`, `data_preprocessing.jl` (contents absorbed or dropped).
Removed exports: `split_data`, `split_data_r`, `splitratio`,
`split_data_with_quality`, `evaluate_split_quality`, `compare_split_methods`,
and all internal helpers (`jitter_data!`, `compute_bounds`,
`encode_categorical!`, …). Public surface is only what is listed below.

### Types and core API

```julia
abstract type SplitKernel end
struct EnergyKernel <: SplitKernel end   # k(x,y) = −‖x−y‖ ⇒ MMD² = energy distance

splitter = SupportPointSplitter(;
    kernel = EnergyKernel(),
    ratio = 0.2,
    kappa = nothing,              # absolute per-iteration subsample size
    max_iterations = 500,
    tolerance = 1e-10,
    n_threads = Threads.nthreads(),
    rng = Random.default_rng(),
    verbose = false,
)

result = datasplit(splitter, data)   # data: AbstractMatrix | DataFrame | AbstractVector
train, test = train_indices(result), test_indices(result)
train_view = data[result, :train]    # getindex sugar retained
```

- `optimize!(points, data, ::EnergyKernel; ...)` implements the closed-form MM
  update of Mak & Joseph (2018) — full-data or stochastic (Joseph & Vakayil
  2021) depending on `kappa` — and returns `(converged::Bool,
  iterations::Int)`, stored truthfully in `SplitResult`.
- `SplitResult` fields: `train_indices`, `test_indices`, `converged`,
  `iterations`, `method`. The `quality` field is removed; quality is a
  diagnostic function, not stored state.
- `kappa` is an absolute subsample size everywhere; `nothing` means full-data
  MM. Documented on the struct.
- All random draws (initialization, jitter, stochastic subsampling) go through
  `splitter.rng`. No internal seeding.
- The optimizer partitions work into `n_threads` chunks explicitly instead of
  relying on `@threads` over all available threads.
- Iterator interface on `SplitResult` (`train, test = result`) retained.

### Quality diagnostics

```julia
energydistance(X, Y)                                   # exact
energydistance(X, Y; subsample = 2_000, repeats = 8, rng)  # V-statistic estimate
splitquality(data, result; kwargs...)                  # preprocess like the
                                                       # splitter, then ED
```

- `splitquality` applies the same preprocessing as `datasplit` (fixes the
  DataFrame/categorical breakage), computes exactly when
  `n ≤ exact_threshold` (default 4,000, keyword-tunable), otherwise averages
  `repeats` subsampled V-statistic estimates of size `subsample`.
- Exact computation accumulates block-wise; no n×n matrix is materialized.
- The estimator is a V-statistic: within-sample means include the zero
  diagonal (matching the optimizer's objective), so the subsampled estimate
  carries a positive bias of order `1/subsample`. This makes it suitable for
  comparing splits rather than as an absolute value; tests check agreement
  with the exact value at moderate n within a tolerance that accounts for
  the bias.

### Comparison

```julia
comparison = compare([splitter_a, splitter_b], data)   # uses splitquality
df = DataFrame(comparison)                             # replaces Base.summary piracy
best(comparison)                                       # lowest energy distance
```

### Subsampling

Build one `KDTree` (NearestNeighbors.jl) over the preprocessed data. For each
support point in sequence, query nearest neighbors and take the closest not
yet selected (re-query with growing k when all returned neighbors are used).
Equivalence with brute force is asserted by tests at small n.

### Optimal ratio

Implement γ = 1/(√p + 1) from Joseph (2022):

- `method = :simple` — p from the model-matrix column count.
  **Verify against the paper before implementing** (the current
  `√(unique rows)` estimate is known-wrong and will be discarded).
- `method = :regression` — estimate p by the paper's regression-based
  procedure (polynomial/stepwise fit). **Derive the exact procedure from the
  paper at implementation time**; until then the keyword errors instead of
  silently falling back.

## Testing

Property tests derived from the papers replace output-matching tests:

1. **MM monotone descent** — in full-data mode the energy-distance objective
   is non-increasing across iterations (uses the subsampled estimator at
   large n; exact at small n).
2. **Optimality** — support-point splits achieve lower train/test energy
   distance than random splits of the same ratio (averaged over repetitions).
3. **Reproducibility** — identical `rng` ⇒ identical result, independent of
   `n_threads`.
4. **Estimator agreement** — subsampled ED estimate within tolerance of the
   exact value at moderate n.
5. **Subsampling equivalence** — k-d tree path equals brute force at small n.
6. **Ratio spot checks** — γ = 1/(√p + 1) for known p.
7. Unit tests for preprocessing (Helmert encoding, constant columns, missing
   values, mixed types) carried over from the existing suite.

Existing tolerance-relaxed output-matching tests are deleted.

## Phase 2 — MMD kernels (direction only)

Add kernels (`GaussianKernel(σ)`, …) as `SplitKernel` subtypes with their own
`optimize!` methods. The closed-form MM update is specific to the energy
kernel; other kernels require their own minimization scheme (derived from the
kernel-herding / MMD-minimization literature at design time). Public API is
unchanged: `SupportPointSplitter(kernel = GaussianKernel(1.0))`.

## Phase 3 — large-scale estimation (direction only)

- Sliced energy-distance estimator (random 1D projections, O(k·n log n)) as an
  alternative `splitquality` backend — adopt only if it beats subsampled
  estimation in accuracy-per-cost (needs verification).
- Performance work: allocation-free MM inner loop, thread scaling, n ≈ 10⁶
  benchmarks.

## Breaking changes

Everything legacy is removed in one release (target v0.2.0): `split_data`,
`split_data_r`, `splitratio`, `split_data_with_quality`,
`evaluate_split_quality`, `compare_split_methods`, the stored
`SplitResult.quality` field, `Base.summary(::SplitComparison)`, and all
internal-helper exports. The package is unregistered with no known downstream
users; no deprecation shims.

## Open items to verify from the papers during implementation

- Joseph (2022): exact definition of p for the simple method; exact
  regression-based estimation procedure for p.
- Mak & Joseph (2018): confirm the MM regularization/step constants used in
  the stochastic variant (current `n0 = 0.2·k` is an inherited implementation
  detail, not from the paper); choose by convergence experiments and document.
