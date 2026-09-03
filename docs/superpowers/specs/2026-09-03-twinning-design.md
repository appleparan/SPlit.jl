# Twinning and multiplets (roadmap M3)

**Status**: Approved design, pre-implementation
**Date**: 2026-09-03
**Branch**: `feat/twinning`
**Roadmap**: M3 on the Roadmap page (`docs/src/85-roadmap.md`)
**Builds on**: `docs/superpowers/specs/2026-09-03-reference-distribution-design.md` (M2)

## TL;DR

Add `TwinningSplitter <: AbstractSplitter`, the sequential nearest-neighbor
partitioning of Vakayil & Joseph (2022, *SADM*, Algorithm 1), and a public
`multiplet(splitter, data, k; strategy)` that returns `k` distribution-balanced
folds using the paper's three strategies (S1 sequential, S2 halving, S3
single run). Twinning plugs into the existing `_select_rows`/`_with_kernel`
protocol, so `datasplit`, `selectrows`, `compare`, and `splitquality` work
unchanged. splitiq mirrors `method='twinning'`, `start`, and `multiplet`. A
benchmark at N up to 10⁶ (p = 10) and a p = 768 case are added to the
Benchmarks page; a kd-tree versus brute-force measurement decides the
nearest-neighbor structure by dimension and is recorded on the Design
experiments page.

Decisions taken with the user on 2026-09-03:

1. Any `ratio` in (0, 1) is accepted. `n` comes from the generic `datasplit`
   rule; the group size is `r = ⌊N/n⌋` with the remainder spread as groups
   of `r + 1`. When `N = r·n` this is exactly the paper's algorithm.
2. All three multiplet strategies of the paper's Section 5 are implemented:
   `:sequential` (S1) and `:halving` (S2) are generic over every
   `AbstractSplitter`; `:single` (S3) is defined for `TwinningSplitter` only.
3. The benchmark follows the roadmap in full: N ∈ {10⁴, 10⁵, 10⁶}, p = 10,
   against the current splitters where they are feasible, plus p = 768.
4. The default starting point is `:farthest` (the row farthest from the
   centroid of the standardized data), which is deterministic; `:random`
   and an explicit row index are also accepted.

## The algorithm (Vakayil & Joseph 2022, Section 3.1, Algorithm 1)

Let `X` be the standardized data (`N` rows) and `n` the number of rows to
select. Twinning forms `n` disjoint groups `S_1, …, S_n` that cover the
rows, and selects one row from each group:

1. `u_1` is the starting row. `S_1 = {u_1} ∪ {r − 1 nearest neighbors of
   u_1}`, ordered by increasing distance to `u_1`; `v_1^{r−1}` denotes the
   farthest member.
2. For `i = 2, …, n`: `u_i` is the row nearest to `v_{i−1}^{r−1}` among the
   rows not yet grouped; `S_i = {u_i} ∪ {r − 1 nearest ungrouped neighbors
   of u_i}`.
3. The selection is `{u_1, …, u_n}`; the complement is the union of the
   `v` members. (Paper, Algorithm 1 lines 4–13; Section 3.1 explains that
   the rule minimizes all three parts of the energy distance between the
   twins, Eq. 10, and Proposition 1 shows that this is the SPlit objective
   up to the factor `(1 − γ)²`.)

Nearest neighbors are Euclidean on the standardized rows. Grouped rows are
masked, not deleted, from the search structure (Section 3.2). Average
complexity `O(dN log N)` (Eq. 12).

### Group sizes (decision 1)

The paper assumes `1/γ = r ∈ ℤ` and `N = rn`; when `n ≠ γN` it sets
`n ← ⌈γN⌉` and lets the last group absorb the remainder (Algorithm 1 line
9). SPlit.jl takes `n` from the generic rule (`n = round(ratio·N)` for the
smaller side in `datasplit`, or the caller's `n` in `selectrows`) and sets

- `r = ⌊N/n⌋`, `extra = N − r·n` (so `0 ≤ extra < n`),
- group `i` has `r + 1` members when `⌊i·extra/n⌋ > ⌊(i−1)·extra/n⌋` and
  `r` members otherwise (the oversized groups are spread evenly along the
  chain rather than concentrated at its end).

Sizes sum to `N`, differ by at most one, and equal `r` everywhere when `N`
is a multiple of `n`, which is the paper's case. `n > N/2` gives `r = 1`
(single-row groups, some of size two); it is allowed by `selectrows` but
`datasplit` never produces it because it always selects the smaller side.
This is recorded under "Differences from the paper" in the docstring.

### Starting row (decision 4)

`start`:

- `:farthest` (default): `argmax_i ‖x_i‖` on the standardized data, whose
  column means are zero, so this is the row farthest from the centroid
  (the paper's choice in Section 4). Ties go to the lowest row index. Does
  not touch `rng`.
- `:random`: `rand(rng, 1:N)` (the paper's choice in Section 5).
- an `Integer` in `1:N`: that row (Algorithm 1 takes `u_1` as input).
  Out of range is an `ArgumentError`.

`TwinningSplitter` is deterministic given the data and a `:farthest` or
integer start; `rng` is consumed only by `:random`.

### Search structure and masking

Implemented with NearestNeighbors.jl:

- `knn(tree, point, k, true, skip)` with `skip(j) = !alive[map[j]]` returns
  the `k` nearest ungrouped rows directly; `nn(tree, point, skip)` returns
  the nearest ungrouped row. No `k` doubling.
- Masked rows still cost traversal time. When more than half of the rows in
  the current tree are masked, the tree is rebuilt on the alive rows only
  (with an index map back to the data). Total rebuild work is geometric,
  `O(N log N)` overall, so per-query cost stays bounded. This is an
  implementation detail, not from the paper, and is stated as such in the
  docstring.
- The tree type is chosen by dimension: `KDTree` below
  `TWINNING_BRUTE_FORCE_DIMENSION`, `BruteTree` at or above it. The
  constant is set by the measurement in `benchmark/twinning_trees.jl`
  (kd-tree versus brute force on standard-normal data at
  `p ∈ {2, 10, 50, 200, 768}`, N = 10⁴), recorded on the Design experiments
  page; if the kd-tree is never slower in the measured range the constant
  is `typemax(Int)` and the page says so. This resolves the roadmap's open
  question "High-dimensional nearest neighbours (M3)".

Duplicate rows need no jitter: ties are resolved by the tree's traversal
order, and the result is still a partition. The docstring says the split is
deterministic given the data, not that ties break by index.

## Public API

```julia
TwinningSplitter(; ratio = 0.2, start = :farthest, rng = Random.default_rng())
multiplet(s::AbstractSplitter, data, k::Integer; strategy = :sequential,
          weights = nothing, reference = nothing, reference_weights = nothing)
  -> Vector{Vector{Int}}
```

### `TwinningSplitter`

```julia
struct TwinningSplitter{R<:AbstractRNG} <: AbstractSplitter
  kernel::EnergyKernel      # fixed: twinning minimizes the energy distance (paper Eq. 9)
  ratio::Float64
  start::Union{Symbol,Int}
  rng::R
end
```

- `kernel` is a field (not a keyword) so `compare`/`DataFrame(::SplitComparison)`
  read it like any other splitter; `_with_kernel(s::TwinningSplitter, k) = s`
  and `_prepare` resolves `EnergyKernel()` to itself.
- Validation: `0 < ratio < 1`; `start ∈ (:farthest, :random)` or a positive
  `Integer` (range-checked against `N` at selection time).
- `_select_rows(s::TwinningSplitter, kernel, X, n; weights, target, target_weights)`
  throws `ArgumentError` when `weights !== nothing` or `target !== nothing`
  ("TwinningSplitter has no weighted or reference form; the paper defines
  it on the data alone"), and otherwise returns
  `(first.(groups), true, n)`: the `u_i` in formation order, `converged =
  true`, `iterations = n` (the number of groups formed, the same convention
  as `HerdingSplitter`).
- `show` prints `TwinningSplitter(ratio=…, start=…)`.
- No `n_threads`: the procedure is serial (paper Section 4).

Internal core, shared by `_select_rows` and `multiplet(:single)`:

```julia
_twin_groups(X::Matrix{Float64}, n::Int, start, rng) -> Vector{Vector{Int}}
```

Groups in formation order; each group lists `u_i` first, then its neighbors
by increasing distance to `u_i`.

### `multiplet`

`multiplet(s, data, k; strategy, weights, reference, reference_weights)`
returns `k` index vectors that partition `1:N`, each sorted ascending, with
sizes differing by at most one. `2 ≤ k ≤ N`, else `ArgumentError`.

- `:sequential` (paper S1, default): for `i = 1, …, k−1`, select
  `n_i = N_rem ÷ (k − i + 1)` rows from the remaining rows with
  `selectrows(s, data[remaining], n_i; weights = weights[remaining],
  reference, reference_weights)`; fold `i` is those rows, and fold `k` is
  what is left. Every run re-fits the preprocessing on the remaining rows
  (Algorithm 1 line 2 per run). The splitter's own `ratio` is ignored.
- `:halving` (paper S2): `k` must be a power of two (else `ArgumentError`).
  Split every part into its selected half (`n = ⌊N_part/2⌋`) and the
  complement, level by level, until `k` parts exist; folds are the parts
  of the last level in order.
- `:single` (paper S3): `TwinningSplitter` only (`ArgumentError` for other
  splitters). One run of `_twin_groups` with `n = ⌊N/k⌋` groups, so every
  group has at least `k` members (`r = k` exactly when `N mod k < n`).
  Fold `j` collects member `j` (by neighbor rank) of every group; the
  `e = N − n·k` members of rank above `k` (`e < k`), taken in group order
  then rank order, go one each to folds `1, …, e`. Fold sizes are `n` or
  `n + 1`.
- `:sequential` and `:halving` work with any `AbstractSplitter` and forward
  `weights`/`reference`/`reference_weights` unchanged (a `TwinningSplitter`
  rejects them as above). Unknown `strategy` is an `ArgumentError`.
- Row subsetting helper `_rows(data, idx)` for matrix (`data[idx, :]`),
  vector (`data[idx]`), and `DataFrame` (`data[idx, :]`).

### Python (splitiq)

```python
datasplit(..., method='twinning', start='farthest')
select_rows(..., method='twinning', start='farthest')
multiplet(data, k, *, strategy='sequential', method='twinning', kernel='energy',
          bandwidth='median', kappa=None, max_iterations=500, tolerance=1e-10,
          n_threads=None, seed=None, start='farthest',
          weights=None, reference=None, reference_weights=None) -> list[np.ndarray]
SplitResult.method: Literal['support_points', 'herding', 'twinning']
```

- `start`: `'farthest'`, `'random'`, or a 0-based `int` (converted to the
  1-based Julia index).
- `method='twinning'` with `kernel='gaussian'`, or with `kappa`,
  `max_iterations`, `tolerance` away from their defaults, or with
  `n_threads` set, raises `ValueError` (the same pattern as herding's
  option check). `start` given with another method raises `ValueError`.
- `multiplet` defaults to `method='twinning'` because the paper defines
  multiplets for twinning and `strategy='single'` exists only there; the
  docstring says so. Folds come back as 0-based `np.ndarray`s.
- `SplitResult.kernel` is `'energy'` and `bandwidth` is `None` for twinning.

## Tests

Properties from the paper, appended to existing files or in new files
(existing tests untouched):

- `test/test_twinning.jl` (new):
  - `_twin_groups` partitions `1:N`; group sizes are `r` or `r + 1`, sum
    to `N`, and are all `r` when `N = r·n`.
  - **Definition equivalence**: a naive `O(N²)` re-implementation of
    Algorithm 1 in the test (brute-force distances, explicit masking, same
    group-size rule) produces the same groups as `_twin_groups` on random
    data with a unique nearest neighbor structure (e.g. `N = 137`, `n = 23`
    and `N = 60`, `n = 30`, sizes that force at least one tree rebuild).
    This is the test that the kd-tree, `skip`, and rebuild machinery
    implement the definition.
  - `start`: `:farthest` selects the largest-norm row first and is
    reproducible without an rng; `:random` is reproducible under the same
    rng; an integer start is the first selected row; out-of-range errors.
  - `datasplit` sizes and partition; `ratio > 0.5` puts the larger side in
    test with `selected = :train`; `selectrows` equals the selected side.
  - `weights`/`reference` are `ArgumentError`s; DataFrame with categorical
    columns works; duplicate rows work; a 1-column vector works.
  - `compare([TwinningSplitter(), HerdingSplitter()], data)` runs and its
    `DataFrame` shows `EnergyKernel`.
  - `multiplet`: for each strategy on `TwinningSplitter`, folds partition
    `1:N`, sizes differ by at most one, folds are sorted; `:sequential` and
    `:halving` also run with `HerdingSplitter`; `:single` with a
    non-twinning splitter errors; `:halving` with `k = 3` errors; `k`
    validation; `N` not a multiple of `k` works for all three.
- `test/test_properties.jl` (append): twinning splits beat random splits
  under energy distance on the 2-D mixture and a 4-D normal; for
  `multiplet(TwinningSplitter(), data, 4)` under every strategy, the
  maximum over folds of the energy distance to the whole data (the paper's
  `ED*`, Section 5) is below the mean of that maximum over random 4-fold
  partitions.
- splitiq `tests/test_twinning.py` (new): `datasplit(method='twinning')`
  partitions and reports `method='twinning'`, `kernel='energy'`,
  `bandwidth=None`; `select_rows` with `start=<int>` returns that index
  first (0-based round trip); `multiplet` for the three strategies
  partitions `range(N)`; the `ValueError` cases above.

## Benchmarks

- `benchmark/twinning.jl` → `docs/src/assets/benchmarks/twinning.md` and
  `twinning.png`. Data: `normal-10d` from `benchmark/datasets.jl` at
  N ∈ {10⁴, 10⁵, 10⁶}. Methods: `twinning`, `herding · energy`,
  `support points · energy` (`kappa = 1_000`, `max_iterations = 100`),
  `random` (mean of 5 seeds). Per-method feasibility caps, stated in the
  table: support points up to N = 10⁵ (its repulsion term is quadratic in
  `n`), herding up to N = 10⁶. Wall time excludes JIT warm-up (throwaway
  splitter copy with its own rng, as in `run.jl`). Score: energy distance
  from `splitquality` with its automatic estimator and a fixed rng, the
  same for every method at a given N. Figure: time versus N (log-log) and
  energy distance relative to random.
- `benchmark/twinning_trees.jl` → `docs/src/assets/benchmarks/twinning_trees.md`:
  twinning wall time with `KDTree` and `BruteTree` on standard-normal data
  at `p ∈ {2, 10, 50, 200, 768}`, N = 10⁴, `ratio = 0.2`; fixes
  `TWINNING_BRUTE_FORCE_DIMENSION`.
- Recorded environment as on the Benchmarks page (Julia version, threads,
  CPU).

## Docs

- Methods page: new section "Twinning and multiplets" (procedure delta:
  step 4 becomes the group chain; group-size rule; start; the three
  multiplet strategies; "Differences from the paper").
- Benchmarks page: new section "Twinning at scale" (one claim, the figure,
  the table link, feasibility caps).
- Design experiments page: new section "Nearest-neighbor structure for
  twinning" with the tree measurement and the resulting constant.
- Roadmap: M3 moved to done, Current-state rows (`TwinningSplitter`,
  k-fold, high-dimensional) updated, open question resolved, changelog.
- index.md: `TwinningSplitter` in "Kernels and splitters", `multiplet` in
  the quick start.
- Python page and splitiq `getting-started.md`/`overview.md`: `method=
  'twinning'`, `start`, `multiplet`.
- AGENTS.md gotchas: twinning's `kernel` is a fixed `EnergyKernel`; it
  rejects `weights`/`reference`; `start = :farthest` uses no rng; group
  sizes `r`/`r + 1`; `multiplet(:single)` is twinning-only; the tree is
  rebuilt when half its rows are masked.

## Non-goals

- Weighted or reference-targeted twinning (no paper definition).
- Approximate nearest neighbors (LSH); the paper uses exact search.
- A twinning-specific `SplitResult` payload (groups are internal; use
  `multiplet(:single)` for the rank structure).
- Changing `datasplit`'s `n` rule to the paper's `⌈γN⌉`.

## References

- Vakayil, A., & Joseph, V. R. (2022). Data Twinning. *Statistical Analysis
  and Data Mining*, 15(5), 598–610. Algorithm 1 (Section 3.1), complexity
  (Section 3.2, Eq. 12), starting point (Sections 4 and 5), multiplets
  (Section 5, strategies S1–S3).
- Joseph, V. R., & Vakayil, A. (2021). SPlit. *Technometrics*, 63(4).
- Mak, S., & Joseph, V. R. (2018). Support points. *Annals of Statistics*,
  46(6A).
