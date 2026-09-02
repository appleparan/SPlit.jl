# SPlit.jl

Julia package for optimal train/test splitting via support points. The
public API is the `export` list in `src/SPlit.jl`; every exported name has a
docstring, and `test/` is the executable spec.

## Source of truth

Correctness is judged against three papers, not against any other
implementation — do not cite or compare with other implementations in code,
comments, or docs:

- Mak & Joseph (2018), Support points, *Annals of Statistics* 46(6A) — the
  MM update and the energy-distance objective.
- Joseph & Vakayil (2021), SPlit, *Technometrics* 63(4) — the splitting
  procedure, stochastic MM, nearest-neighbor subsampling.
- Joseph (2022), Optimal Ratio for Data Splitting, *SADM* 15(4) —
  γ = 1/(√p + 1).

Tests encode properties those papers guarantee (monotone descent, beating
random splits, reproducibility under a fixed `rng`). Prefer that style over
output-matching tests. The design record is
`docs/superpowers/specs/2026-09-02-paper-aligned-redesign-design.md`.

## Gotchas

- `kappa` is an absolute row count; stochastic mode runs only when it is
  below the number of data rows. Full-data mode is the pure MM step and must
  stay monotone — the descent test enforces it.
- `tolerance` compares the largest *squared* per-point displacement.
- The MM sweep in `optimizer.jl` is the hot loop and is written as explicit
  coordinate loops on purpose: keep it allocation-free. `n0 = 0.2n` there is
  an implementation constant, not from the papers.
- `energydistance` is exact unless `subsample` is passed; the subsampled
  estimate has a positive bias of order `1/subsample`. `splitquality`
  switches to it automatically above `exact_threshold`.
- Categorical columns are Helmert-encoded in canonical level order so splits
  do not depend on row order; `Union{Missing,T}` columns without missings
  are accepted.
- All randomness goes through the caller's `rng`; nothing in `src/` seeds or
  prints on a default path.
- `GaussianKernel` has no `kappa` mode; its `:median` bandwidth is resolved
  at `datasplit` time and the resolved kernel is stored in
  `result.method.kernel`.
- `:median` fails with an `ArgumentError` when at least half of all row
  pairs coincide (e.g. a single binary categorical column) — pass a numeric
  bandwidth then. Choose `σ` on the scale of the standardized data (`:median`
  does this), since a bandwidth far below the row spacing makes the
  objective flat and the optimizer stops at the initial sample.

## Workflow

- Tests: `julia --project=. -e "using Pkg; Pkg.test()"`.
- Formatting runs through pre-commit (JuliaFormatter, markdownlint); it is
  not a package dependency.
- Docs are Documenter.jl (`docs/make.jl`), deployed from `main` to `gh-pages`.
