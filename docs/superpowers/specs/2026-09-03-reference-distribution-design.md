# Reference (target) distribution and `selectrows` (roadmap M2)

**Status**: Approved design, pre-implementation
**Date**: 2026-09-03
**Branch**: `feat/reference-distribution`
**Roadmap**: M2 on the Roadmap page (`docs/src/85-roadmap.md`)
**Builds on**: `docs/superpowers/specs/2026-09-03-weighted-samples-design.md` (M1)

## TL;DR

Let a splitter choose rows of `data` so that the chosen subset approximates
a *different* distribution: a `reference` sample (optionally weighted by
`reference_weights`) instead of the data itself. Preprocessing becomes
fit/apply, fit on the reference and applied to both sets. A new public
`selectrows(splitter, data, n; ...)` returns the chosen row indices without
building a train/test partition; `datasplit` is `selectrows` plus the
complement. `splitquality(...; reference)` scores the selected rows against
the reference. splitiq mirrors `reference`, `reference_weights`, and
`select_rows`. Without `reference`, every existing code path is untouched and
results are bit-identical.

Decisions taken with the user on 2026-09-03:

1. With a `reference`, only the reference carries weights
   (`reference_weights`); passing `weights` for the data together with
   `reference` is an `ArgumentError`.
2. Standardization, Helmert levels, and the `:median` bandwidth are fit on
   the reference and applied to both sets.
3. `splitquality(data, result; reference)` measures the selected rows
   against the reference.
4. A selection-only public API, `selectrows` (Python: `select_rows`), ships in the same change. It is not called `select` because `DataFrames`, a dependency, exports `select`.

## Semantics

Let `X` be the data (`N` rows, the candidates) and `R` the reference (`M`
rows) with normalized weights `v̄` (`reference_weights`, `nothing` = uniform).
The target distribution is `P_R = Σ_l v̄_l δ(r_l)`.

- A splitter chooses `n` rows of `X` whose uniform empirical measure is
  close to `P_R` under the splitter's discrepancy (energy distance or MMD²).
  Candidates are always rows of `X`; the reference only defines the target.
- `reference = nothing` (default) means `P_R` is the data's own
  distribution, `P_w` from M1 (`weights` allowed). This is the existing
  behavior and stays bit-identical.
- `reference` accepts what `data` accepts (matrix, `DataFrame`, vector) and
  must have the same number of columns; for a `DataFrame`, the same column
  names in the same order, with matching column kinds (numeric vs
  categorical). Mismatch is an `ArgumentError`.
- `weights !== nothing && reference !== nothing` is an `ArgumentError`
  ("with a reference, weight the reference (`reference_weights`), not the
  data").
- The train/test labeling rule of `datasplit` is unchanged: the selected
  `n_small` rows are the test set when `ratio ≤ 1/2`, the training set
  otherwise.

## Preprocessing: fit/apply

New internal type and functions in `src/preprocessing.jl`:

```julia
struct Preprocessor
  columns::Vector{ColumnSpec}   # one per input column, in input order
  μ::Vector{Float64}            # per encoded column
  σ::Vector{Float64}
end
# ColumnSpec is one of
#   NumericColumn(keep::Bool)
#   CategoricalColumn(levels::Vector{String})   # canonical order, union of sets
fit_preprocessor(data; weights = nothing, extra = nothing) -> Preprocessor
apply_preprocessor(prep, data) -> Matrix{Float64}
```

- `fit_preprocessor(R; weights = v, extra = X)`: encoding is decided from
  `R` (constant columns of `R` are dropped), except that categorical levels
  are the canonical-order union of the levels present in `R` and in
  `extra = X`, so rows of `X` with a level absent from `R` still encode.
  `μ`, `σ` come from the encoded `R` (weighted as in M1 when `v` is given,
  using the unbiased weighted variance). A column whose encoded values are
  constant on `R` but not on `X` is dropped (it carries no information about
  the target); this is documented.
- `apply_preprocessor(prep, Y)` encodes `Y` with the stored specs and
  standardizes with the stored `μ`, `σ`. A level of a categorical column in
  `Y` outside `prep`'s levels is an `ArgumentError`.
- The existing `preprocess(data)` and `preprocess(data, weights)` become
  `apply_preprocessor(fit_preprocessor(data; weights), data)`, and the
  arithmetic must stay identical: `μ = mean(col)`, `σ = std(col)` (or the
  weighted forms), applied as `(col .- μ) ./ σ`. Tests assert `==` between
  the old and new results on random data, DataFrames with categoricals, and
  weighted inputs.
- `_encode` stays as the shared encoder; `fit_preprocessor` calls it to
  learn `μ`, `σ`, and `apply_preprocessor` re-encodes with fixed specs.

Canonical level order for the union: plain string columns use
`sort(union(levels_R, levels_X))`; `CategoricalVector` columns use `R`'s
declared level order filtered to present levels, followed by `X`-only levels
in `X`'s declared order.

## Kernel resolution

`resolve(kernel, R_encoded, rng, v)` as in M1, on the encoded reference:
the `:median` bandwidth is the median pairwise distance of (a weighted draw
of) reference rows. The resolved kernel is stored in `result.method` as now.

## Optimizers and herding with a target

Signatures gain a target that defaults to the data itself:

```julia
support_points(kernel, data, n; target = nothing, target_weights = nothing, kwargs...)
herd(kernel, X, n; target = nothing, target_weights = nothing, n_threads)
```

`target === nothing` keeps M1 behavior exactly (`weights` remains the M1
keyword for the data-as-target case; `weights` and `target` together is an
`ArgumentError` at this level too). With `target = R`:

- Energy MM: the data sums of the update run over the rows of `R` with
  `ŵ = M v̄` (mean one), the repulsion term is unchanged; initial points are
  drawn from rows of `X` (the candidates, jittered inside the bounding box
  of `X`), and points are clamped to the bounding box of `X`, since they are
  rounded to rows of `X` by `select_nearest`. Stochastic mode subsamples
  rows of `R` (`kappa < M`), with the M1 `:uniform` rule. The MM step is
  still the majorizer of the energy distance between the points and `P_R`,
  so full-data descent stays monotone.
- Gaussian: objective `mean k(ξ, ξ) − 2 Σ_l v̄_l mean_m k(ξ_m, r_l)` and the
  matching gradient; Armijo, first step, and convergence rules unchanged.
- Herding: data term `d_i = Σ_l v̄_l k(x_i, r_l)` for every candidate row
  `x_i` of `X` (`O(NM)`), selected-set term over the chosen rows as now.
- `select_nearest(X, points)` is unchanged.

Implementation rule from M1 carries over: `target = nothing` dispatches to
the existing methods; the target versions are new methods, and constant
`target_weights` are turned into `nothing` after validation.

## Public API

```julia
selectrows(s::AbstractSplitter, data, n::Integer;
       weights = nothing, reference = nothing, reference_weights = nothing) -> Vector{Int}
datasplit(s::AbstractSplitter, data;
          weights = nothing, reference = nothing, reference_weights = nothing) -> SplitResult
splitquality(data, result; reference = nothing, reference_weights = nothing, kwargs...)
compare(methods, data; reference = nothing, reference_weights = nothing, kwargs...)
```

- `selectrows` returns the `n` selected row indices of `data` in selection
  order (support-point order, or greedy order for herding), as the roadmap's
  design principle asks (`Vector{Int}`). Convergence diagnostics are
  available through `datasplit`. `1 ≤ n ≤ N`, else `ArgumentError`.
- Internally both share `_select(s, data, n; weights, reference,
  reference_weights) -> (indices, converged, iterations, fitted_splitter)`;
  `datasplit` computes `n_small` from `ratio` as now and adds the complement.
- `SplitResult` gains a sixth field `selected::Symbol` (`:test` or
  `:train`), the side that holds the chosen rows. The existing 5-argument
  constructor stays as an outer constructor that sets `selected` to `:test`
  when `length(test) ≤ length(train)` and `:train` otherwise, so existing
  code and tests keep working. `show` prints it.
- `splitquality(data, result; reference, reference_weights)`: fits the
  preprocessor on the reference, applies it to both, and returns the
  discrepancy (energy distance or MMD² under `kernel`, estimator rules as
  now) between the rows of `data` on the `result.selected` side (uniform)
  and the reference (weighted). Without `reference` the M1 behavior stays.
- `compare` forwards `reference`/`reference_weights` to both `datasplit` and
  `splitquality`, and resolves a `:median` scoring kernel on the encoded
  reference.

Python (splitiq), mirroring the names:

```python
select_rows(data, n, *, method, kernel, bandwidth, kappa, max_iterations, tolerance,
       n_threads, seed, weights=None, reference=None, reference_weights=None) -> np.ndarray   # 0-based
datasplit(..., reference=None, reference_weights=None)
splitquality(data, result, *, reference=None, reference_weights=None, ...)
SplitResult.selected: Literal['test', 'train']
```

`reference` is converted like `data` (`to_julia_data`), `reference_weights`
like `weights` (`to_weights`). Julia `ArgumentError`s surface as
`ValueError`.

## Tests

Properties, appended to the existing files (existing tests untouched):

- Preprocessing: `preprocess(data)` and `preprocess(data, weights)` are `==`
  to their pre-M2 outputs (captured by computing the same formulas inline
  in the test) on numeric matrices and DataFrames with categoricals;
  `apply_preprocessor` uses the reference's `μ`, `σ` (a shifted `Y` does not
  come out centered); categorical union levels encode data-only levels;
  unknown level at apply time errors; column-count and column-name
  mismatches error.
- `reference = data` (the same matrix, no weights) reproduces
  `datasplit(s, data)` exactly (`==` on indices) for both splitters and
  both kernels.
- Sub-population concentration: with `reference = data[data[:, 1] .> 0, :]`
  the fraction of selected rows with positive first coordinate is higher
  than without a reference, for `SupportPointSplitter` (both kernels) and
  `HerdingSplitter`; and `splitquality(...; reference)` of the split beats
  the mean over random subsets of the same size (weighted energy distance
  to the reference).
- Duplication invariance of `reference_weights` for the herding data term
  and for one MM sweep (as in M1, with the reference duplicated).
- Monotone descent of the energy objective toward `P_R` in full-data mode.
- `select(s, data, n)` equals the selected side of `datasplit` for a
  matching `ratio`; `selectrows` returns exactly `n` distinct indices in
  `1:N`; `n` out of range errors; `weights` together with `reference`
  errors.
- `SplitResult.selected` is `:test` for `ratio ≤ 0.5` and `:train` above;
  the 5-argument constructor still works.
- splitiq: parity tests for `selectrows`, `reference`, `reference_weights`, the
  `selected` field, and the error cases.

## Docs

- Methods page: new section "Reference distribution and `selectrows`" (delta
  only: what changes in the five steps when a reference is given, and how
  `select` relates to `datasplit`).
- Roadmap: M2 done, Current-state row updated, changelog line.
- Python page and splitiq docs: the new keywords and `selectrows`.
- AGENTS.md gotcha: with a reference, candidates are rows of `data`, the
  target is the reference; `weights` and `reference` are mutually exclusive;
  preprocessing is fit on the reference.

## Non-goals

- Weighted candidates together with a reference (decision 1).
- A reference for `TwinningSplitter` / k-fold (M3).
- Bypassing preprocessing for embedding inputs (M5 open question).
- Returning diagnostics from `selectrows` (use `datasplit`).

## References

- Mak, S., & Joseph, V. R. (2018). Support points. *Annals of Statistics*,
  46(6A). Support points of an arbitrary target measure.
- Joseph, V. R., & Vakayil, A. (2021). SPlit. *Technometrics*, 63(4).
- Chen, Y., Welling, M., & Smola, A. (2010). Super-samples from kernel
  herding. *UAI*.
