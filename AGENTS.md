# SPlit.jl

Julia package for distribution-preserving subset selection on tabular data
and embeddings, grown from SPlit (Joseph & Vakayil, 2022): train/test splits
(`datasplit`), k-fold multiplets (`multiplet`), and row selection
(`selectrows`) toward the data, a weighted measure, or a reference sample.
The public API is the `export` list in `src/SPlit.jl`; every exported name
has a docstring, and `test/` is the executable spec.

## Source of truth

Correctness is judged against three papers, not against any other
implementation. Do not cite or compare with other implementations in code,
comments, or docs:

- Mak & Joseph (2018), Support points, *Annals of Statistics* 46(6A): the
  MM update and the energy-distance objective.
- Joseph & Vakayil (2022), SPlit, *Technometrics* 64(2): the splitting
  procedure, stochastic MM, nearest-neighbor subsampling.
- Joseph (2022), Optimal Ratio for Data Splitting, *SADM* 15(4):
  γ = 1/(√p + 1).

Tests encode properties those papers guarantee (monotone descent, beating
random splits, reproducibility under a fixed `rng`). Prefer that style over
output-matching tests. The design record is
`docs/superpowers/specs/2026-09-02-paper-aligned-redesign-design.md`.

## Gotchas

- `kappa` is an absolute row count; stochastic mode runs only when it is
  below the number of rows of the target (the data, or the reference when
  one is given). Full-data mode is the pure MM step and must stay monotone;
  the descent test enforces it.
- `tolerance` compares the largest *squared* per-point displacement.
- The MM sweep in `optimizer.jl` is the hot loop and is written as explicit
  coordinate loops on purpose: keep it allocation-free. `n0 = 0.2n` there is
  an implementation constant, not from the papers.
- `energydistance`/`mmd` are exact unless an `estimator` (or the compatibility
  `subsample` keyword) is passed; `Subsample`'s estimate has a positive bias
  of order `1/m` (`Subsample(m, repeats)`), `RandomSlices`/`RandomFeatures`
  are unbiased.
  `splitquality` switches to the fallback `DiscrepancyEstimator` chosen by
  the selection experiment (see Design experiments) automatically above
  `exact_threshold` (20,000 rows).
- estimator/kernel combinations are methods of `_energydistance`/`_mmd`: add
  a method, never an `if`; herding has no estimator mode on purpose (measured
  worse than random, see Design experiments).
- Categorical columns are Helmert-encoded in canonical level order so splits
  do not depend on row order; `Union{Missing,T}` columns without missing values
  are accepted.
- All randomness goes through the caller's `rng`; nothing in `src/` seeds or
  prints on a default path.
- `GaussianKernel`'s `:median` bandwidth is resolved at `datasplit` time and
  the resolved kernel is stored in `result.method.kernel`. Its `kappa` mode
  runs the Gaussian MM sweep (`_mm_sweep!(::GaussianKernel, …)`: mean-shift
  data term, L-smooth majorized repulsion with
  `B = 4(n−1)e^{−3/2}/(nσ²)`, the energy path's running-average blend,
  displacement rule); full data stays the Armijo path — measured, see
  Design experiments.
- `:median` fails with an `ArgumentError` when at least half of all row
  pairs coincide (e.g. a single binary categorical column); pass a numeric
  bandwidth then. Choose `σ` on the scale of the standardized data (`:median`
  does this), since a bandwidth far below the row spacing makes the
  objective flat and the optimizer stops at the initial sample.
- `HerdingSplitter` is deterministic given the data and a numeric kernel;
  `rng` only feeds a `:median` bandwidth. Its data term is exact (`O(N²)`);
  there is no subsampled mode. `SplitResult.iterations` is the number of
  selections.
- Gaussian optimizer (full data): the first trial step is 10% of the standardized data
  scale (median column range) divided by the largest per-point gradient
  norm, not a fixed constant; later iterations warm-start from twice the
  previous accepted step. Convergence needs at least 2 iterations and either
  the displacement rule (`tolerance`, squared displacement) or the
  relative-decrease rule (`rtol`, on the bounded shifted objective).
- `select_nearest` rounds each optimized point to its own starting data row
  whenever the optimizer's displacement is below the row spacing (measured
  on `normal-10d`/`uniform-5d` at N = 10,000, see Benchmarks): the split is
  then exactly the initial random sample. `HerdingSplitter` has no such
  rounding step. It queries a k-d tree below `NEAREST_BRUTE_FORCE_DIMENSION`
  (200) columns and `MatrixSearch` above (see Design experiments);
  `MatrixSearch` compiles once for any width, but its queries must be
  contiguous column views, or the SIMD distance loop degrades (measured 7x).
- `weights` (on `datasplit`, `splitquality`, `compare`) and
  `weights_x`/`weights_y` (on `energydistance`, `mmd`) define weighted
  empirical distributions; the selected subset is always uniform.
  `nothing` dispatches to the unweighted methods, which must stay
  bit-identical; weighted behavior lives in separate methods. In
  particular, a constant weight vector is turned into `nothing` after
  validation (`_uniform_as_nothing`), so uniform weights take the
  unweighted path; inside the optimizers the mean-one factor `ŵ` is
  exactly `1.0` for uniform weights. Tests use "weights as duplication
  counts equals duplicated rows".
- `reference`/`reference_weights` (on `selectrows`, `datasplit`, `splitquality`,
  `compare`) define the target measure; candidates are always rows of
  `data`. Preprocessing is fit on the reference (`fit_preprocessor`) and
  applied to both; `weights` and `reference` are mutually exclusive;
  `reference = nothing` must stay bit-identical to the untargeted path.
  A column constant on the reference but varying on `data` is kept, not
  dropped: it is centered at the reference's value and scaled by the
  data's spread. `SplitResult.selected` names the side holding the chosen
  rows.
- The selection function is named `selectrows`, not `select`, because
  `DataFrames` exports `select`. Docstrings must sit directly above the
  function they document.
- `TwinningSplitter` has a fixed `kernel = EnergyKernel()` (its objective is
  the energy distance) and rejects `weights`/`reference` with an
  `ArgumentError`; `start = :farthest` consumes no rng. Group sizes are
  `r = N ÷ n` or `r + 1`, spread evenly; the paper's case is `N = r·n`.
  The search tree is rebuilt once more than half of its rows are masked,
  and switches to the `MatrixSearch` brute force at
  `TWINNING_BRUTE_FORCE_DIMENSION` (measured, see Design experiments); the
  `search` keyword also accepts `:brute_tree` (the prior `BruteTree`
  structure), kept only so the benchmark that replaced it stays
  reproducible, never chosen by default. `multiplet(:single)` is
  twinning-only; `:sequential`/`:halving` call `selectrows` on any
  splitter and re-fit preprocessing per run.
- `KernelThinningSplitter`: KT-SPLIT runs on the first `n·2^m` rows of a
  shuffle (all rows when `N/n` is a power of two); `weights`/`reference`
  act on the KT-SWAP objective only; `delta` is the papers' δ; swap
  candidates exclude selected rows; `n > N ÷ 2` returns the complement of
  a kernel-thinning selection of `N − n` rows; cost is the herding class
  `O(N²)`; threaded sums use fixed 1,024-row chunks so results do not
  depend on `n_threads`.
- `standardize = false` (on `datasplit`, `selectrows`, `multiplet`,
  `splitquality`, `compare`) skips preprocessing entirely — including
  constant-column removal, not just the scaling — and rejects `DataFrame`
  input. It is the embedding mode; see the LLM data-selection docs page.
- Compress++ (`compress` on `KernelThinningSplitter`) is defined only for
  the data's own measure: `:always` errors with `weights`/`reference` and
  `:auto` falls back to plain kernel thinning there. `:auto` does not fire
  at the default 20% ratio (so `datasplit` with the default splitter is
  unchanged) but can below roughly 10% for N ≥ 10⁴ — up to n = 800 at
  N = 10⁴, 10,100 at N = 10⁵, 64,000 at N = 10⁶; pass `compress = :never`
  to keep the plain path there. `g = max(4, ⌈log₂(2n/√N)⌉)`. The rule
  was checked against wall time (`benchmark/compress.jl`, see Design
  experiments).
- The example under `examples/` is not run in CI, and its table under
  `docs/src/assets/examples/` is committed output — regenerate it only when
  asked, like the benchmark tables.

## Workflow

- Tests: `julia --project=. -e "using Pkg; Pkg.test()"`.
- Formatting runs through pre-commit (JuliaFormatter, markdownlint); it is
  not a package dependency.
- Docs are Documenter.jl (`docs/make.jl`), deployed from `main` to `gh-pages`.
- Julia/Python parity: every capability exposed in SPlit.jl must be exposed
  in `splitiq/` in the same change, with tests under `splitiq/tests` and a
  docs mention; a Julia-only feature PR is incomplete.
