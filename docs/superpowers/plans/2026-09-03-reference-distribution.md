# Reference Distribution and `selectrows` (M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every splitter choose rows of `data` that approximate a separate `reference` sample (optionally weighted), expose a selection-only `selectrows` API, and mirror both in splitiq.

**Architecture:** Preprocessing becomes fit/apply (`Preprocessor`), fit on the reference and applied to both sets. The optimizers and herding gain a `target` (matrix + weights) that defaults to the data itself; with `target = nothing` the code path is the M1 one and results are bit-identical. A shared internal `_select` drives both `selectrows` and `datasplit`; `SplitResult` records which side holds the selection. `splitquality`/`compare` score the selected rows against the reference when one is given.

**Tech Stack:** Julia 1.10+, Documenter.jl, Python 3.13 + juliacall (splitiq), pytest, uv, pre-commit.

**Spec:** `docs/superpowers/specs/2026-09-03-reference-distribution-design.md` (builds on `docs/superpowers/specs/2026-09-03-weighted-samples-design.md`)

## Global Constraints

- Existing public signatures and numerical results are unchanged; every new keyword defaults to `nothing`. `preprocess(data)` and `preprocess(data, weights)` stay bit-identical (`==`) to their current output.
- With `reference = nothing`, `datasplit`, `splitquality`, `compare`, `support_points`, and `herd` produce exactly what they produce today (same `rng` consumption).
- `weights` together with `reference` (or `target`) is an `ArgumentError`; `reference_weights` without `reference` is an `ArgumentError`.
- Candidates are always rows of `data`; the reference only defines the target measure. Initial points, bounding box, and `select_nearest` use `data`; the data term of every objective uses the target.
- Constant `reference_weights` / `target_weights` become `nothing` after validation (`_uniform_as_nothing`), as in M1.
- Estimator/kernel combinations stay methods; the MM sweep stays allocation-free; all randomness through the caller's `rng`; no citation of other implementations.
- Docs: add sections, never rewrite existing ones. Existing tests are append-only.
- Every Julia capability lands in splitiq in this branch, with tests.
- Run one Julia test file with `julia --project=. test/<file>.jl`; the suite with `julia --project=. -e "using Pkg; Pkg.test()"`. Commit messages `<type>: <Capitalized description>` + trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; pre-commit runs on commit, never bypass it.
- Work only in `.claude/worktrees/feat-reference` (branch `feat/reference-distribution`).

---

## File structure

| File | Responsibility |
|---|---|
| `src/preprocessing.jl` | `Preprocessor`, `fit_preprocessor`, `apply_preprocessor`; `preprocess` re-expressed on top of them |
| `src/optimizer.jl` | `target`/`target_weights` on both `support_points` and the two trajectory helpers |
| `src/herding.jl` | `target`/`target_weights` on `herd`; cross data term `_data_term(kernel, X, R, …)` |
| `src/splitter.jl` | `SplitResult.selected`, `_select`, public `selectrows`, generic `datasplit(::AbstractSplitter, …)` |
| `src/quality.jl`, `src/comparison.jl` | `reference`/`reference_weights` on `splitquality` and `compare` |
| `src/SPlit.jl` | export `selectrows` |
| `test/test_preprocessing.jl`, `test_optimizer.jl`, `test_herding.jl`, `test_splitter.jl`, `test_quality.jl`, `test_comparison.jl`, `test_properties.jl` | appended tests |
| `docs/src/10-methods.md`, `85-roadmap.md`, `30-python.md`, `AGENTS.md` | added sections |
| `splitiq/src/splitiq/split.py`, `quality.py`, `__init__.py`, `splitiq/tests/test_reference.py`, `splitiq/docs/getting-started.md` | parity |

---

### Task 1: Preprocessor fit/apply

**Files:**

- Modify: `src/preprocessing.jl`
- Test: `test/test_preprocessing.jl`

**Interfaces:**

- Produces:
  - `struct Preprocessor` with fields `names::Union{Nothing,Vector{String}}` (DataFrame column names, `nothing` for matrix/vector input), `specs::Vector{ColumnSpec}`, `keep::Vector{Bool}` (over encoded columns), `μ::Vector{Float64}`, `σ::Vector{Float64}` (over kept columns).
  - `NumericColumn <: ColumnSpec` (no fields), `CategoricalColumn <: ColumnSpec` with `levels::Vector{String}`.
  - `fit_preprocessor(data; weights = nothing, extra = nothing) -> Preprocessor`
  - `apply_preprocessor(prep::Preprocessor, data) -> Matrix{Float64}`
  - `preprocess(data)`, `preprocess(data, weights)` unchanged in behavior.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_preprocessing.jl`:

```julia
@testset "Preprocessor fit/apply" begin
  @testset "preprocess is unchanged: matrix, weighted, DataFrame" begin
    rng = MersenneTwister(200)
    data = randn(rng, 70, 3) .* [1.0 4.0 0.2] .+ [1.0 -2.0 0.0]
    # reference values computed with the pre-M2 formulas, inline
    expected = copy(data)
    for j = 1:3
      μ = mean(expected[:, j])
      σ = std(expected[:, j])
      expected[:, j] .= (expected[:, j] .- μ) ./ σ
    end
    @test SPlit.preprocess(data) == expected
    w = rand(rng, 70)
    wn = w ./ sum(w)
    correction = 1 - sum(abs2, wn)
    expected_w = copy(data)
    for j = 1:3
      col = expected_w[:, j]
      μ = sum(wn .* col)
      σ = sqrt(sum(wn .* (col .- μ) .^ 2) / correction)
      expected_w[:, j] .= (col .- μ) ./ σ
    end
    @test SPlit.preprocess(data, w) == expected_w
    df = DataFrame(x = randn(MersenneTwister(201), 30), g = repeat(["a", "b", "c"], 10))
    X = SPlit.preprocess(df)
    @test size(X) == (30, 3)
    H = SPlit.helmert_matrix(3)
    idx = Dict("a" => 1, "b" => 2, "c" => 3)
    raw = hcat(df.x, [H[idx[v], 1] for v in df.g], [H[idx[v], 2] for v in df.g])
    for j = 1:3
      raw[:, j] .= (raw[:, j] .- mean(raw[:, j])) ./ std(raw[:, j])
    end
    @test X == raw
    # constant column dropped, all-constant errors
    @test size(SPlit.preprocess(hcat(ones(10), randn(MersenneTwister(202), 10))), 2) == 1
    @test_throws ArgumentError SPlit.preprocess(ones(10, 2))
  end

  @testset "apply uses the fitted μ and σ" begin
    R = randn(MersenneTwister(203), 100, 2)
    prep = SPlit.fit_preprocessor(R)
    Y = randn(MersenneTwister(204), 40, 2) .+ 5.0
    Ya = SPlit.apply_preprocessor(prep, Y)
    @test SPlit.apply_preprocessor(prep, R) == SPlit.preprocess(R)
    @test all(abs.(mean(Ya; dims = 1)) .> 3.0)     # not re-centered
    @test isapprox(Ya, (Y .- mean(R; dims = 1)) ./ std(R; dims = 1); atol = 1e-12)
  end

  @testset "weighted fit uses the weighted moments" begin
    R = randn(MersenneTwister(205), 80, 2)
    w = rand(MersenneTwister(206), 80)
    prep = SPlit.fit_preprocessor(R; weights = w)
    @test SPlit.apply_preprocessor(prep, R) == SPlit.preprocess(R, w)
  end

  @testset "columns constant on the fit set are dropped for both sets" begin
    R = hcat(ones(20), randn(MersenneTwister(207), 20))
    X = randn(MersenneTwister(208), 15, 2)
    prep = SPlit.fit_preprocessor(R; extra = X)
    @test size(SPlit.apply_preprocessor(prep, X), 2) == 1
    @test_throws ArgumentError SPlit.fit_preprocessor(ones(20, 2); extra = X)
  end

  @testset "categorical levels are the union, in canonical order" begin
    R = DataFrame(x = randn(MersenneTwister(209), 12), g = repeat(["a", "b"], 6))
    X = DataFrame(x = randn(MersenneTwister(210), 9), g = repeat(["a", "b", "c"], 3))
    prep = SPlit.fit_preprocessor(R; extra = X)
    spec = prep.specs[2]
    @test spec isa SPlit.CategoricalColumn
    @test spec.levels == ["a", "b", "c"]
    XR = SPlit.apply_preprocessor(prep, R)
    XX = SPlit.apply_preprocessor(prep, X)
    # the (a,b) vs c contrast is constant on R and is dropped: one Helmert column survives
    @test size(XR, 2) == 2
    @test size(XX, 2) == 2
    # level c is unknown when the preprocessor was fit without X
    prep_r = SPlit.fit_preprocessor(R)
    @test_throws ArgumentError SPlit.apply_preprocessor(prep_r, X)
    # CategoricalVector keeps the declared order, then data-only levels
    Rc = DataFrame(g = categorical(repeat(["z", "y"], 5); levels = ["z", "y", "w"]), x = randn(MersenneTwister(211), 10))
    Xc = DataFrame(g = categorical(repeat(["q", "z"], 3); levels = ["q", "z"]), x = randn(MersenneTwister(212), 6))
    prepc = SPlit.fit_preprocessor(Rc; extra = Xc)
    @test prepc.specs[1].levels == ["z", "y", "q"]
  end

  @testset "shape and column mismatches error" begin
    R = randn(MersenneTwister(213), 20, 3)
    prep = SPlit.fit_preprocessor(R)
    @test_throws ArgumentError SPlit.apply_preprocessor(prep, randn(5, 2))
    Rd = DataFrame(x = randn(10), g = repeat(["a", "b"], 5))
    prepd = SPlit.fit_preprocessor(Rd)
    @test_throws ArgumentError SPlit.apply_preprocessor(prepd, DataFrame(g = repeat(["a", "b"], 5), x = randn(10)))
    @test_throws ArgumentError SPlit.apply_preprocessor(prepd, DataFrame(x = randn(10), g = randn(10)))
    @test_throws ArgumentError SPlit.fit_preprocessor(R; extra = randn(5, 2))
    @test_throws ArgumentError SPlit.apply_preprocessor(prep, [1.0, missing, 2.0][:, :] )
  end
end
```

(`test_preprocessing.jl` already uses `DataFrames`; add `using CategoricalArrays` and `using Statistics` at the top of the new testset block via a `begin ... end` if they are not imported at file top — check the file header first and add the `using` lines there only if missing.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_preprocessing.jl`
Expected: `UndefVarError: fit_preprocessor not defined`

- [ ] **Step 3: Implement**

Replace the body of `src/preprocessing.jl` from `_encode(data::AbstractMatrix)` down to the end of the weighted `_standardize!` with the following (keep `helmert_matrix`, `_is_constant`, the unweighted `_standardize!`, `_is_categorical`, `_canonical_levels` as they are; delete the old `_encode` methods and the old `preprocess` definitions):

```julia
# Per-input-column encoding rule learned at fit time.
abstract type ColumnSpec end
struct NumericColumn <: ColumnSpec end
struct CategoricalColumn <: ColumnSpec
  levels::Vector{String}
end

"""
    Preprocessor

Fitted preprocessing: per-column encoding rules, which encoded columns are
kept (those not constant on the fit set), and the mean and scale of every
kept column. Built by [`fit_preprocessor`](@ref), applied by
[`apply_preprocessor`](@ref). Internal.
"""
struct Preprocessor
  names::Union{Nothing,Vector{String}}
  specs::Vector{ColumnSpec}
  keep::Vector{Bool}
  μ::Vector{Float64}
  σ::Vector{Float64}
end

_check_no_missing(data::AbstractMatrix) =
  any(ismissing, data) && throw(ArgumentError("Dataset contains missing value(s)."))
function _check_no_missing(data::DataFrame)
  for col in eachcol(data)
    any(ismissing, col) && throw(ArgumentError("Dataset contains missing value(s)."))
  end
end

# Union of the canonical levels of `col` and, when given, `extra_col`:
# `col`'s canonical order first, then the levels only `extra_col` has, in
# `extra_col`'s canonical order.
function _union_levels(col, extra_col)
  levels_ = string.(_canonical_levels(col))
  extra_col === nothing && return levels_
  for l in string.(_canonical_levels(extra_col))
    l in levels_ || push!(levels_, l)
  end
  return levels_
end

# Column specs from the fit set (and the extra set's levels).
function _column_specs(data::AbstractMatrix, extra)
  _check_no_missing(data)
  all(x -> x isa Number, data) ||
    throw(ArgumentError("Matrix input must contain only numeric values."))
  if extra !== nothing
    extra isa AbstractMatrix ||
      throw(ArgumentError("reference and data must both be matrices or both DataFrames"))
    size(extra, 2) == size(data, 2) ||
      throw(ArgumentError("reference and data must have the same number of columns"))
    _check_no_missing(extra)
  end
  return nothing, ColumnSpec[NumericColumn() for _ in axes(data, 2)]
end

function _column_specs(data::DataFrame, extra)
  _check_no_missing(data)
  if extra !== nothing
    extra isa DataFrame ||
      throw(ArgumentError("reference and data must both be matrices or both DataFrames"))
    names(extra) == names(data) ||
      throw(ArgumentError("reference and data must have the same column names in the same order"))
    _check_no_missing(extra)
  end
  specs = ColumnSpec[]
  for name in names(data)
    col = data[!, name]
    extra_col = extra === nothing ? nothing : extra[!, name]
    if _is_categorical(col)
      extra_col === nothing ||
        _is_categorical(extra_col) ||
        throw(ArgumentError("column $(name) is categorical in one set and numeric in the other"))
      push!(specs, CategoricalColumn(_union_levels(col, extra_col)))
    elseif Base.nonmissingtype(eltype(col)) <: Number
      extra_col === nothing ||
        Base.nonmissingtype(eltype(extra_col)) <: Number ||
        throw(ArgumentError("column $(name) is categorical in one set and numeric in the other"))
      push!(specs, NumericColumn())
    else
      throw(ArgumentError("Unsupported column type in column: $(name)"))
    end
  end
  return String.(names(data)), specs
end

# Encode `data` with fixed specs into the full (unfiltered) column matrix.
function _encode(names_::Nothing, specs::Vector{ColumnSpec}, data::AbstractMatrix)
  _check_no_missing(data)
  size(data, 2) == length(specs) ||
    throw(ArgumentError("expected $(length(specs)) columns, got $(size(data, 2))"))
  all(x -> x isa Number, data) ||
    throw(ArgumentError("Matrix input must contain only numeric values."))
  return Float64.(data)
end
_encode(::Vector{String}, ::Vector{ColumnSpec}, ::AbstractMatrix) =
  throw(ArgumentError("the preprocessor was fit on a DataFrame; pass a DataFrame"))
_encode(::Nothing, ::Vector{ColumnSpec}, ::DataFrame) =
  throw(ArgumentError("the preprocessor was fit on a matrix; pass a matrix"))

function _encode(names_::Vector{String}, specs::Vector{ColumnSpec}, data::DataFrame)
  _check_no_missing(data)
  names(data) == names_ ||
    throw(ArgumentError("expected columns $(names_), got $(names(data))"))
  columns = Vector{Vector{Float64}}()
  for (name, spec) in zip(names_, specs)
    col = data[!, name]
    if spec isa CategoricalColumn
      _is_categorical(col) ||
        throw(ArgumentError("column $(name) must be categorical"))
      index = Dict(l => i for (i, l) in enumerate(spec.levels))
      H = helmert_matrix(length(spec.levels))
      rows = map(col) do v
        get(index, string(v)) do
          throw(ArgumentError("unknown level $(repr(v)) in column $(name)"))
        end
      end
      for j in axes(H, 2)
        push!(columns, [H[r, j] for r in rows])
      end
    else
      Base.nonmissingtype(eltype(col)) <: Number ||
        throw(ArgumentError("column $(name) must be numeric"))
      push!(columns, Float64.(col))
    end
  end
  isempty(columns) && return zeros(nrow(data), 0)
  # hcat(columns...) (not reduce) so a single column still yields an n×1 Matrix
  return hcat(columns...)
end

"""
    fit_preprocessor(data; weights = nothing, extra = nothing) -> Preprocessor

Learn the preprocessing on `data`: categorical columns are Helmert-encoded
over the canonical-order union of their levels in `data` and in `extra`
(the set the preprocessor will also be applied to), encoded columns that
are constant on `data` are dropped, and every kept column gets the mean and
scale of `data` (weighted forms when `weights` is given, as in
[`preprocess`](@ref)). Internal.
"""
_as_matrix(x::AbstractVector) = reshape(collect(x), :, 1)
_as_matrix(x) = x

function fit_preprocessor(data; weights = nothing, extra = nothing)
  data = _as_matrix(data)
  extra = _as_matrix(extra)
  names_, specs = _column_specs(data, extra)
  M = _encode(names_, specs, data)
  keep = [!_is_constant(view(M, :, j)) for j in axes(M, 2)]
  any(keep) || throw(ArgumentError("All columns are constant."))
  K = M[:, keep]
  N = size(K, 1)
  weights === nothing || _check_weights(weights, N)
  w = _uniform_as_nothing(weights)
  μ = Vector{Float64}(undef, size(K, 2))
  σ = Vector{Float64}(undef, size(K, 2))
  if w === nothing
    for j in axes(K, 2)
      μ[j] = mean(view(K, :, j))
      σ[j] = std(view(K, :, j))
    end
  else
    wn = _normalize_weights(w, N)
    correction = 1 - sum(abs2, wn)
    correction > 0 ||
      throw(ArgumentError("weights must be positive on at least two rows"))
    for j in axes(K, 2)
      col = view(K, :, j)
      μ[j] = sum(wn .* col)
      σ[j] = sqrt(sum(wn .* (col .- μ[j]) .^ 2) / correction)
      σ[j] > 0 || throw(
        ArgumentError(
          "column $j is constant on the rows with positive weight; drop it or give those rows weight",
        ),
      )
    end
  end
  return Preprocessor(names_, specs, keep, μ, σ)
end

"""
    apply_preprocessor(prep::Preprocessor, data) -> Matrix{Float64}

Encode `data` with the rules of `prep` and standardize every kept column
with `prep`'s mean and scale. A categorical level `prep` has not seen, a
column-count or column-name mismatch, and a numeric/categorical kind
mismatch are `ArgumentError`s. Internal.
"""
function apply_preprocessor(prep::Preprocessor, data)
  M = _encode(prep.names, prep.specs, _as_matrix(data))[:, prep.keep]
  for j in axes(M, 2)
    @views M[:, j] .= (M[:, j] .- prep.μ[j]) ./ prep.σ[j]
  end
  return M
end

"""
    preprocess(data) -> Matrix{Float64}
    preprocess(data, weights) -> Matrix{Float64}

Validate and transform `data` for splitting: reject missing values, encode
categorical columns with Helmert contrasts, drop constant columns, and
standardize every remaining column. Accepts `AbstractMatrix`, `DataFrame`,
and `AbstractVector` inputs. Equivalent to fitting a [`Preprocessor`](@ref)
on `data` and applying it to `data`.

With `weights` (one non-negative entry per row), standardization uses the
weighted mean `μⱼ = Σ w̄ᵢ xᵢⱼ` and the unbiased weighted variance
`σⱼ² = Σ w̄ᵢ (xᵢⱼ − μⱼ)² / (1 − Σ w̄ᵢ²)` with `w̄` the weights scaled to sum
one, which reduces to the `n − 1` denominator of `std` for uniform weights;
the encoding steps are the same. `weights = nothing` is the unweighted
method. A constant weight vector is treated as `nothing`, so uniform
weights take the unweighted path and reproduce it exactly.
"""
preprocess(data) = apply_preprocessor(fit_preprocessor(data), data)
preprocess(data, ::Nothing) = preprocess(data)
preprocess(data, weights::AbstractVector) =
  apply_preprocessor(fit_preprocessor(data; weights), data)
```

Bit-identity argument (must hold, the tests check `==`): the old unweighted path computed `μ = mean(col)`, `σ = std(col)` on the kept columns and applied `(col .- μ) ./ σ`; the new one computes the same `μ`, `σ` on the same kept columns and applies the same broadcast. The old weighted path used the same weighted formulas. Column order and the kept set are the same (`_is_constant` on the raw numeric column equals `_is_constant` on its `Float64` encoding; a single-level categorical yields no Helmert column in both). If a `==` test fails, do not loosen it: find the arithmetic difference.

Remove the now-unused weighted `_standardize!(M, w)` and the old `_encode` methods; keep the unweighted `_standardize!(M)` only if something else still calls it (`grep -n "_standardize!" src/ test/`), otherwise remove it and adjust the test that calls `SPlit._standardize!(copy(M), fill(1/60, 60))` (from the M1 follow-ups) by replacing it with `SPlit.apply_preprocessor(SPlit.fit_preprocessor(data; weights = fill(1/60, 60)), data)` compared to `SPlit.preprocess(data)` with `isapprox` (that test may be edited for this reason only). Also `grep -n "_encode(" src/ test/` and update any remaining caller (the M1 follow-up test uses `SPlit._encode(data)`; replace with `SPlit._encode(nothing, SPlit.ColumnSpec[SPlit.NumericColumn() for _ in 1:size(data, 2)], data)` or drop that test line if it only fed `_standardize!`).

- [ ] **Step 4: Run the tests**

Run: `julia --project=. test/test_preprocessing.jl`, then the full suite `julia --project=. -e "using Pkg; Pkg.test()"` (every other test file exercises `preprocess`).
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.jl test/test_preprocessing.jl
git commit -m "refactor: Split preprocessing into a fitted Preprocessor and its application

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Target measure in the support-point optimizers

**Files:**

- Modify: `src/optimizer.jl`
- Test: `test/test_optimizer.jl`

**Interfaces:**

- Consumes: `_check_weights`, `_uniform_as_nothing`, `_mean_one_weights`, `_normalize_weights`, `_uniform_weights` (M1); weighted `_exact_energydistance(X, Y, wx, wy)`, `_mmd_objective(k, points, data, w_bar)`, `_mmd_gradient!(G, k, points, data, w_hat, n_threads)`.
- Produces:
  - `support_points(::EnergyKernel, data, n; target = nothing, target_weights = nothing, kwargs...)`
  - `support_points(::GaussianKernel, data, n; target = nothing, target_weights = nothing, kwargs...)`
  - `_objective_trajectory(data, n; …, target = nothing, target_weights = nothing)`, `_mmd_trajectory(k, data, n; …, target = nothing, target_weights = nothing)`
  - `_target_weights(weights, target, target_weights, N, M) -> (w_hat, w_bar)` helper (see below)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_optimizer.jl`:

```julia
@testset "support points toward a target measure" begin
  rng = MersenneTwister(300)
  data = randn(rng, 200, 2)
  R = data[data[:, 1] .> 0, :]          # a sub-population as the target

  @testset "target = data reproduces the untargeted run exactly" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      a, ca, ia = SPlit.support_points(kernel, data, 20; max_iterations = 40, rng = MersenneTwister(1))
      b, cb, ib = SPlit.support_points(kernel, data, 20; max_iterations = 40, rng = MersenneTwister(1), target = data)
      @test a == b
      @test (ca, ia) == (cb, ib)
    end
  end

  @testset "points move toward the target for both kernels" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      plain, _, _ = SPlit.support_points(kernel, data, 30; max_iterations = 100, rng = MersenneTwister(2))
      targeted, _, _ = SPlit.support_points(kernel, data, 30; max_iterations = 100, rng = MersenneTwister(2), target = R)
      @test count(>(0.0), targeted[:, 1]) > count(>(0.0), plain[:, 1])
      @test count(>(0.0), targeted[:, 1]) >= 24
      # points stay inside the candidates' bounding box
      for j = 1:2
        lo, hi = extrema(view(data, :, j))
        @test all(lo .<= targeted[:, j] .<= hi)
      end
    end
  end

  @testset "target weights as duplication counts: one MM sweep on duplicated target rows" begin
    Rsmall = R[1:30, :]
    counts = rand(MersenneTwister(301), 1:3, 30)
    Rdup = vcat([Rsmall[i:i, :] for i = 1:30 for _ = 1:counts[i]]...)
    n = 5
    points = data[1:n, :] .+ 0.05
    bounds = SPlit._data_bounds(data)
    new_w = similar(points)
    new_d = similar(points)
    SPlit._mm_sweep!(new_w, zeros(n), copy(points), Rsmall, SPlit._mean_one_weights(Float64.(counts)), zeros(n), 1.0, bounds, 1)
    SPlit._mm_sweep!(new_d, zeros(n), copy(points), Rdup, ones(size(Rdup, 1)), zeros(n), 1.0, bounds, 1)
    @test isapprox(new_w, new_d; atol = 1e-10)
  end

  @testset "monotone descent toward the target (energy) and non-increase (Gaussian)" begin
    traj = SPlit._objective_trajectory(data, 15; max_iterations = 40, rng = MersenneTwister(3), target = R)
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-8
    end
    v = rand(MersenneTwister(302), size(R, 1))
    trajw = SPlit._objective_trajectory(data, 15; max_iterations = 40, rng = MersenneTwister(3), target = R, target_weights = v)
    for t = 2:length(trajw)
      @test trajw[t] <= trajw[t-1] + 1e-8
    end
    trajg = SPlit._mmd_trajectory(GaussianKernel(1.0), data, 15; max_iterations = 40, rng = MersenneTwister(4), target = R, target_weights = v)
    for t = 2:length(trajg)
      @test trajg[t] <= trajg[t-1] + 1e-12
    end
  end

  @testset "stochastic mode subsamples the target" begin
    big = randn(MersenneTwister(303), 600, 2)
    Rbig = big[big[:, 1] .> 0, :]
    pts, _, _ = SPlit.support_points(EnergyKernel(), big, 40; kappa = 100, max_iterations = 60, rng = MersenneTwister(5), target = Rbig)
    @test count(>(0.0), pts[:, 1]) >= 30
  end

  @testset "validation" begin
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; target = R, weights = ones(200))
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; target_weights = ones(200))
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; target = R, target_weights = ones(3))
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; target = randn(10, 3))
    @test_throws ArgumentError SPlit.support_points(GaussianKernel(1.0), data, 5; target = R, weights = ones(200))
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_optimizer.jl`
Expected: `unsupported keyword argument "target"`

- [ ] **Step 3: Implement**

In `src/optimizer.jl`, add after `_draw_subsample`:

```julia
# Validate the (weights | target, target_weights) combination and return the
# target matrix plus its mean-one and sum-one weight vectors. `weights`
# belongs to the data-as-target case only.
function _resolve_target(
  data::Matrix{Float64},
  weights,
  target,
  target_weights,
)
  N = size(data, 1)
  if target === nothing
    target_weights === nothing ||
      throw(ArgumentError("target_weights needs a target"))
    weights === nothing || _check_weights(weights, N)
    weights = _uniform_as_nothing(weights)
    w_hat = weights === nothing ? ones(N) : _mean_one_weights(weights)
    w_bar = weights === nothing ? nothing : _normalize_weights(weights, N)
    return data, w_hat, w_bar
  end
  weights === nothing ||
    throw(ArgumentError("with a target, weight the target (target_weights), not the data"))
  target isa AbstractMatrix ||
    throw(ArgumentError("target must be a matrix with the same number of columns as data"))
  size(target, 2) == size(data, 2) ||
    throw(ArgumentError("target must have the same number of columns as data"))
  R = Matrix{Float64}(target)
  M = size(R, 1)
  M >= 1 || throw(ArgumentError("target must have at least one row"))
  target_weights === nothing || _check_weights(target_weights, M)
  tw = _uniform_as_nothing(target_weights)
  w_hat = tw === nothing ? ones(M) : _mean_one_weights(tw)
  w_bar = tw === nothing ? nothing : _normalize_weights(tw, M)
  return R, w_hat, w_bar
end
```

Energy `support_points`: add keywords `target = nothing, target_weights = nothing` after `weights`; replace the two lines

```julia
  weights === nothing || _check_weights(weights, N)
  weights = _uniform_as_nothing(weights)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(weights)
```

with

```julia
  R, w_hat, _ = _resolve_target(data, weights, target, target_weights)
  M = size(R, 1)
```

and then: `bounds = _data_bounds(data)` stays; `working = copy(R)` and the duplicate-row jitter uses `_data_bounds(R)` when `target !== nothing` (keep `bounds` when `R === data`), i.e.

```julia
  working = copy(R)
  if length(unique(eachrow(working))) < M
    _jitter!(rng, working, target === nothing ? bounds : _data_bounds(R))
  end
  candidates = target === nothing ? working : copy(data)
  if target !== nothing && length(unique(eachrow(candidates))) < N
    _jitter!(rng, candidates, bounds)
  end
  points = _initial_points(rng, candidates, n, bounds)
```

(`candidates` is `working` itself when there is no target, so the untargeted path draws the initial points from the jittered data exactly as before and consumes `rng` identically.) Replace `stochastic = kappa !== nothing && kappa < N` with `kappa < M`, and `_draw_subsample(rng, N, kappa, w_hat, rule)` with `_draw_subsample(rng, M, kappa, w_hat, rule)`. Everything else is unchanged.

Gaussian `support_points`: add the same keywords; replace the four weight lines with `R, w_hat, w_bar = _resolve_target(data, weights, target, target_weights)`; the working/candidates/initial-points block as above (with `M = size(R, 1)`), and keep `_mmd_objective(k, points, working, w_bar)`, `_mmd_gradient!(G, k, points, working, w_hat, n_threads)`, `_armijo_step!(…, working, bounds, w_bar)` (they now run over the target rows).

`_objective_trajectory` and `_mmd_trajectory`: add `target = nothing, target_weights = nothing`, use `_resolve_target`, run the sweep/gradient over `R` with `w_hat`, take the initial points from `data`, and score against `R`: energy `score(points) = w_bar === nothing ? _exact_energydistance(points, R) : _exact_energydistance(points, R, _uniform_weights(n), w_bar)`; Gaussian `_mmd_objective(k, points, R, w_bar)`.

Docstring paragraph to append to both `support_points` docstrings:

```text
`target` (a matrix with the same columns as `data`) makes the points
approximate the empirical distribution of `target` instead of `data`:
the data term of the objective runs over the rows of `target`, weighted by
`target_weights` (sum-one normalized, `nothing` for uniform; a constant
vector is treated as `nothing`), while the initial points and the bounding
box come from `data`, whose rows the points are later rounded to. In
stochastic mode `kappa` subsamples the rows of `target`. `weights` is only
for the case without a target; giving both is an `ArgumentError`.
```

- [ ] **Step 4: Run the tests**

Run: `julia --project=. test/test_optimizer.jl` and `julia -t 4 --project=. test/test_optimizer.jl`
Expected: all pass, including every pre-existing test (bit-identical untargeted path).

- [ ] **Step 5: Commit**

```bash
git add src/optimizer.jl test/test_optimizer.jl
git commit -m "feat: Let support points target a reference sample

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Target measure in kernel herding

**Files:**

- Modify: `src/herding.jl`
- Test: `test/test_herding.jl`

**Interfaces:**

- Produces:
  - `_data_term(kernel, X, R::Matrix{Float64}, n_threads)` (`dᵢ = meanₗ k(xᵢ, rₗ)`) and `_data_term(kernel, X, R, v_bar::AbstractVector{Float64}, n_threads)` (`dᵢ = Σₗ v̄ₗ k(xᵢ, rₗ)`)
  - `herd(kernel, X, n; weights = nothing, target = nothing, target_weights = nothing, n_threads)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_herding.jl`:

```julia
@testset "herding toward a target" begin
  k = GaussianKernel(1.0)
  data = randn(MersenneTwister(400), 200, 2)
  R = data[data[:, 1] .> 0, :]

  @testset "target = data reproduces the untargeted selection exactly" begin
    @test SPlit.herd(k, data, 25; target = data) == SPlit.herd(k, data, 25)
    @test SPlit.herd(EnergyKernel(), data, 25; target = data) == SPlit.herd(EnergyKernel(), data, 25)
  end

  @testset "cross data term equals the self data term on the same matrix, and the weighted form" begin
    d_self = SPlit._data_term(k, data, 1)
    d_cross = SPlit._data_term(k, data, data, 1)
    @test isapprox(d_self, d_cross; atol = 1e-12)
    Rsmall = R[1:20, :]
    v = rand(MersenneTwister(401), 20)
    d_w = SPlit._data_term(k, data, Rsmall, v ./ sum(v), 1)
    brute = [sum(v[l] / sum(v) * SPlit.kernelvalue(k, data[i, :], Rsmall[l, :]) for l = 1:20) for i = 1:200]
    @test isapprox(d_w, brute; atol = 1e-12)
  end

  @testset "selections concentrate in the target sub-population" begin
    for kernel in (k, EnergyKernel())
      plain = SPlit.herd(kernel, data, 30)
      targeted = SPlit.herd(kernel, data, 30; target = R)
      @test count(i -> data[i, 1] > 0, targeted) > count(i -> data[i, 1] > 0, plain)
      @test count(i -> data[i, 1] > 0, targeted) >= 27
    end
  end

  @testset "validation" begin
    @test_throws ArgumentError SPlit.herd(k, data, 5; target = R, weights = ones(200))
    @test_throws ArgumentError SPlit.herd(k, data, 5; target_weights = ones(200))
    @test_throws ArgumentError SPlit.herd(k, data, 5; target = randn(10, 3))
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_herding.jl`
Expected: `unsupported keyword argument "target"`

- [ ] **Step 3: Implement**

In `src/herding.jl`, after the weighted `_data_term`:

```julia
# Cross data term d_i = mean over the rows r_l of R of k(x_i, r_l), for every
# candidate row x_i of X; threaded over i on the transposes.
function _data_term(kernel::SplitKernel, X::Matrix{Float64}, R::Matrix{Float64}, n_threads::Int)
  N = size(X, 1)
  M = size(R, 1)
  Xt = permutedims(X)
  Rt = permutedims(R)
  d = Vector{Float64}(undef, N)
  chunks = collect(Iterators.partition(1:N, cld(N, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for i in chunk
      s = 0.0
      @views xi = Xt[:, i]
      @views for l = 1:M
        s += kernelvalue(kernel, xi, Rt[:, l])
      end
      d[i] = s / M
    end
  end
  return d
end

# Weighted cross data term d_i = Σ_l v̄_l k(x_i, r_l), v̄ scaled to sum one.
function _data_term(
  kernel::SplitKernel,
  X::Matrix{Float64},
  R::Matrix{Float64},
  v_bar::AbstractVector{Float64},
  n_threads::Int,
)
  N = size(X, 1)
  M = size(R, 1)
  Xt = permutedims(X)
  Rt = permutedims(R)
  d = Vector{Float64}(undef, N)
  chunks = collect(Iterators.partition(1:N, cld(N, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for i in chunk
      s = 0.0
      @views xi = Xt[:, i]
      @views for l = 1:M
        s += v_bar[l] * kernelvalue(kernel, xi, Rt[:, l])
      end
      d[i] = s
    end
  end
  return d
end
```

In `herd`: add keywords `target::Union{Nothing,AbstractMatrix} = nothing, target_weights::Union{Nothing,AbstractVector} = nothing`; replace the weight lines and the `d = ...` expression with

```julia
  R, _, v_bar = _resolve_target(X, weights, target, target_weights)
  d = if target === nothing
    v_bar === nothing ? _data_term(kernel, X, n_threads) :
    _data_term(kernel, X, v_bar, n_threads)
  else
    v_bar === nothing ? _data_term(kernel, X, R, n_threads) :
    _data_term(kernel, X, R, v_bar, n_threads)
  end
```

(`_resolve_target` is defined in `src/optimizer.jl`, which is included before `herding.jl`.) The selected-set term loop is unchanged. Append to the `herd` docstring: "`target` (a matrix with the same columns as `X`) replaces the data term by the mean kernel value to the rows of `target`, weighted by `target_weights` when given; the candidates stay the rows of `X`. Cost `O(NM + nN)`. `weights` and `target` are mutually exclusive."

- [ ] **Step 4: Run the tests**

Run: `julia --project=. test/test_herding.jl` and with `-t 4`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/herding.jl test/test_herding.jl
git commit -m "feat: Let kernel herding target a reference sample

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `SplitResult.selected`, `_select`, `selectrows`, and `datasplit` with a reference

**Files:**

- Modify: `src/splitter.jl`, `src/herding.jl` (remove its `datasplit`), `src/SPlit.jl` (export `selectrows`)
- Test: `test/test_splitter.jl`, `test/test_herding.jl`

**Interfaces:**

- Consumes: Tasks 1–3.
- Produces:
  - `SplitResult` with a sixth field `selected::Symbol`; 5-argument outer constructor kept.
  - `_prepare(s, data, weights, reference, reference_weights) -> (X, kernel, target, target_weights)`: preprocessing + kernel resolution for both cases.
  - `_select_rows(s::SupportPointSplitter, kernel, X, n; weights, target, target_weights) -> (indices, converged, iterations)` and the `HerdingSplitter` method.
  - `_with_kernel(s, kernel) -> s` (the fitted copy).
  - `_select(s, data, n; weights, reference, reference_weights) -> (indices, converged, iterations, fitted)`
  - `selectrows(s::AbstractSplitter, data, n::Integer; weights = nothing, reference = nothing, reference_weights = nothing) -> Vector{Int}` (exported)
  - `datasplit(s::AbstractSplitter, data; weights = nothing, reference = nothing, reference_weights = nothing)` (one generic method replacing the two)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_splitter.jl`:

```julia
@testset "select and datasplit with a reference" begin
  data = randn(MersenneTwister(500), 300, 2)
  R = data[data[:, 1] .> 0, :]

  @testset "SplitResult.selected and the compatibility constructor" begin
    s = SupportPointSplitter(max_iterations = 20, rng = MersenneTwister(1))
    r = datasplit(s, data)
    @test r.selected === :test
    s8 = SupportPointSplitter(ratio = 0.8, max_iterations = 20, rng = MersenneTwister(1))
    @test datasplit(s8, data).selected === :train
    legacy = SPlit.SplitResult(collect(61:300), collect(1:60), true, 0, s)
    @test legacy.selected === :test
    @test occursin("selected=test", sprint(show, r))
  end

  @testset "select returns the selected side of datasplit" begin
    for s in (
      SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(2)),
      SupportPointSplitter(kernel = GaussianKernel(1.0), max_iterations = 40, rng = MersenneTwister(2)),
      HerdingSplitter(kernel = GaussianKernel(1.0)),
    )
      s2 = deepcopy(s)          # copy the rng state before datasplit consumes it
      r = datasplit(s, data)
      idx = selectrows(s2, data, 60)
      @test length(idx) == 60
      @test allunique(idx)
      @test all(1 .<= idx .<= 300)
      @test sort(idx) == sort(r.test_indices)
    end
  end

  @testset "reference = data reproduces the plain split exactly" begin
    for s in (
      SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(3)),
      HerdingSplitter(kernel = EnergyKernel()),
    )
      a = datasplit(deepcopy(s), data)
      b = datasplit(deepcopy(s), data; reference = data)
      @test a.test_indices == b.test_indices
    end
  end

  @testset "a sub-population reference concentrates the selection" begin
    for s in (
      SupportPointSplitter(max_iterations = 100, rng = MersenneTwister(4)),
      SupportPointSplitter(kernel = GaussianKernel(1.0), max_iterations = 100, rng = MersenneTwister(4)),
      HerdingSplitter(kernel = GaussianKernel(1.0)),
    )
      plain = selectrows(deepcopy(s), data, 60)
      targeted = selectrows(deepcopy(s), data, 60; reference = R)
      @test count(i -> data[i, 1] > 0, targeted) > count(i -> data[i, 1] > 0, plain)
      @test count(i -> data[i, 1] > 0, targeted) >= 50
    end
  end

  @testset "reference_weights and DataFrame references" begin
    df = DataFrame(x = data[:, 1], y = data[:, 2], g = repeat(["a", "b", "c"], 100))
    ref = df[df.x .> 0, :]
    s = SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(5))
    idx = selectrows(s, df, 60; reference = ref, reference_weights = rand(MersenneTwister(6), nrow(ref)))
    @test length(idx) == 60
    r = datasplit(SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(7)), df; reference = ref)
    @test length(r.test_indices) == 60
    @test r.selected === :test
  end

  @testset "validation" begin
    s = SupportPointSplitter(max_iterations = 5)
    @test_throws ArgumentError selectrows(s, data, 0)
    @test_throws ArgumentError selectrows(s, data, 301)
    @test_throws ArgumentError selectrows(s, data, 10; reference = R, weights = ones(300))
    @test_throws ArgumentError selectrows(s, data, 10; reference_weights = ones(300))
    @test_throws ArgumentError selectrows(s, data, 10; reference = randn(20, 3))
    @test_throws ArgumentError datasplit(s, data; reference = R, weights = ones(300))
  end
end
```

Append to `test/test_herding.jl` (the herding `datasplit` moves to the generic method; this guards it):

```julia
@testset "herding datasplit through the generic method" begin
  data = randn(MersenneTwister(410), 120, 2)
  s = HerdingSplitter(kernel = GaussianKernel(1.0))
  r = datasplit(s, data)
  @test r.iterations == 24
  @test r.converged
  @test r.selected === :test
  @test sort(r.test_indices) == sort(selectrows(s, data, 24))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. test/test_splitter.jl`
Expected: `type SplitResult has no field selected` / `UndefVarError: selectrows`

- [ ] **Step 3: Implement**

`src/splitter.jl`:

1. `SplitResult` gains `selected::Symbol` as the last field; add the outer constructor

```julia
SplitResult(train, test, converged, iterations, method) = SplitResult(
  train,
  test,
  converged,
  iterations,
  method,
  length(test) <= length(train) ? :test : :train,
)
```

and extend the docstring: "`selected` is the side (`:test` or `:train`) that holds the rows the splitter chose; the other side is the complement." Update `Base.show` to print `selected=$(r.selected)` after `iterations`.

1. Fitted copies:

```julia
_with_kernel(s::SupportPointSplitter, kernel) = SupportPointSplitter(
  kernel, s.ratio, s.kappa, s.max_iterations, s.tolerance, s.n_threads, s.rng, s.verbose,
)
```

(and in `src/herding.jl`: `_with_kernel(s::HerdingSplitter, kernel) = HerdingSplitter(kernel, s.ratio, s.n_threads, s.rng)`.)

1. Preparation shared by both entry points:

```julia
# Preprocess and resolve the kernel for the data-as-target case (weights)
# or the reference case. Returns the encoded data, the resolved kernel, the
# encoded target (or nothing), and the target weights.
function _prepare(s::AbstractSplitter, data, weights, reference, reference_weights)
  if reference === nothing
    reference_weights === nothing ||
      throw(ArgumentError("reference_weights needs a reference"))
    X = preprocess(data, weights)
    return X, resolve(s.kernel, X, s.rng, weights), nothing, nothing
  end
  weights === nothing || throw(
    ArgumentError("with a reference, weight the reference (reference_weights), not the data"),
  )
  prep = fit_preprocessor(reference; weights = reference_weights, extra = data)
  R = apply_preprocessor(prep, reference)
  X = apply_preprocessor(prep, data)
  return X, resolve(s.kernel, R, s.rng, reference_weights), R, reference_weights
end
```

1. Per-splitter row selection:

```julia
function _select_rows(s::SupportPointSplitter, kernel, X, n; weights, target, target_weights)
  points, converged, iterations = support_points(
    kernel, X, n;
    kappa = s.kappa, max_iterations = s.max_iterations, tolerance = s.tolerance,
    n_threads = s.n_threads, rng = s.rng, verbose = s.verbose,
    weights, target, target_weights,
  )
  return select_nearest(X, points), converged, iterations
end
```

(and in `src/herding.jl`:)

```julia
function _select_rows(s::HerdingSplitter, kernel, X, n; weights, target, target_weights)
  return herd(kernel, X, n; weights, target, target_weights, n_threads = s.n_threads), true, n
end
```

1. The shared core and the public functions:

```julia
function _select(
  s::AbstractSplitter,
  data,
  n::Integer;
  weights = nothing,
  reference = nothing,
  reference_weights = nothing,
)
  X, kernel, target, target_weights = _prepare(s, data, weights, reference, reference_weights)
  N = size(X, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  indices, converged, iterations =
    _select_rows(s, kernel, X, Int(n); weights, target, target_weights)
  return indices, converged, iterations, _with_kernel(s, kernel)
end

"""
    selectrows(splitter::AbstractSplitter, data, n; weights = nothing,
           reference = nothing, reference_weights = nothing) -> Vector{Int}

Indices of the `n` rows of `data` the splitter chooses, in selection order
(support-point order for `SupportPointSplitter`, greedy order for
`HerdingSplitter`), without building a train/test partition. The chosen
rows approximate the data's own distribution (weighted by `weights`) or,
when `reference` is given, the distribution of `reference` (weighted by
`reference_weights`): preprocessing is then fit on `reference` and applied
to both, candidates stay the rows of `data`, and `weights` may not be
given. Convergence diagnostics are reported by [`datasplit`](@ref).
"""
function selectrows(
  s::AbstractSplitter,
  data,
  n::Integer;
  weights = nothing,
  reference = nothing,
  reference_weights = nothing,
)
  return _select(s, data, n; weights, reference, reference_weights)[1]
end

_nrows(data::AbstractMatrix) = size(data, 1)
_nrows(data::AbstractVector) = length(data)
_nrows(data::DataFrame) = nrow(data)

function datasplit(
  s::AbstractSplitter,
  data;
  weights = nothing,
  reference = nothing,
  reference_weights = nothing,
)
  n_total = _nrows(data)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))
  small, converged, iterations, fitted =
    _select(s, data, n_small; weights, reference, reference_weights)
  rest = setdiff(1:n_total, small)
  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  selected = s.ratio <= 0.5 ? :test : :train
  return SplitResult(collect(train), collect(test), converged, iterations, fitted, selected)
end
```

Delete the old `datasplit(::SupportPointSplitter, …)` in `splitter.jl` and `datasplit(::HerdingSplitter, …)` in `herding.jl`. Order of operations in the untargeted path must match the old one exactly (preprocess → resolve → support_points/herd → select_nearest), which it does; the `n_small` check now happens before preprocessing, which only changes which error fires first for degenerate input (acceptable).

Extend the `datasplit` docstring with:

```text
`reference` (same kind and columns as `data`; optionally weighted by
`reference_weights`) makes the chosen side approximate the distribution of
`reference` instead of the data: preprocessing is fit on `reference` and
applied to both sets, a `:median` bandwidth is resolved on the encoded
reference, and candidates remain the rows of `data`. `weights` cannot be
combined with `reference`. The train/test labeling rule is unchanged;
`result.selected` names the side that holds the chosen rows. See
[`selectrows`](@ref) for the indices alone.
```

1. `src/SPlit.jl`: add `selectrows` to the `export … datasplit` line.

- [ ] **Step 4: Run the tests**

Run: `julia --project=. test/test_splitter.jl`, `test/test_herding.jl`, then the full suite.
Expected: all pass (the pre-existing `datasplit` tests must be untouched and green).

- [ ] **Step 5: Commit**

```bash
git add src/splitter.jl src/herding.jl src/SPlit.jl test/test_splitter.jl test/test_herding.jl
git commit -m "feat: Add select and a reference distribution to datasplit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `splitquality` and `compare` against a reference

**Files:**

- Modify: `src/quality.jl`, `src/comparison.jl`
- Test: `test/test_quality.jl`, `test/test_comparison.jl`, `test/test_properties.jl`

**Interfaces:**

- Produces: `splitquality(data, result; reference = nothing, reference_weights = nothing, kwargs...)`, `compare(methods, data; reference = nothing, reference_weights = nothing, kwargs...)`, `_selected_indices(result) = result.selected === :test ? result.test_indices : result.train_indices`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_quality.jl`:

```julia
@testset "splitquality against a reference" begin
  data = randn(MersenneTwister(600), 300, 2)
  R = data[data[:, 1] .> 0, :]
  s = SupportPointSplitter(max_iterations = 60, rng = MersenneTwister(1))
  r = datasplit(s, data; reference = R)
  q = splitquality(data, r; reference = R)
  @test q isa Float64
  @test q >= -1e-12
  # equals the discrepancy between the selected rows and the reference under the reference-fit preprocessing
  prep = SPlit.fit_preprocessor(R; extra = data)
  Xs = SPlit.apply_preprocessor(prep, data)[r.test_indices, :]
  Rp = SPlit.apply_preprocessor(prep, R)
  @test isapprox(q, energydistance(Xs, Rp); atol = 1e-12)
  v = rand(MersenneTwister(2), size(R, 1))
  qw = splitquality(data, r; reference = R, reference_weights = v)
  @test isapprox(qw, energydistance(SPlit.apply_preprocessor(SPlit.fit_preprocessor(R; weights = v, extra = data), data)[r.test_indices, :], SPlit.apply_preprocessor(SPlit.fit_preprocessor(R; weights = v, extra = data), R); weights_y = v); atol = 1e-12)
  # Gaussian kernel scoring resolves :median on the reference and runs
  @test splitquality(data, r; reference = R, kernel = GaussianKernel()) >= -1e-12
  @test_throws ArgumentError splitquality(data, r; reference = R, weights = ones(300))
  @test_throws ArgumentError splitquality(data, r; reference_weights = ones(10))
end
```

Append to `test/test_comparison.jl`:

```julia
@testset "compare with a reference" begin
  data = randn(MersenneTwister(610), 150, 2)
  R = data[data[:, 1] .> 0, :]
  methods = [
    SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(1)),
    HerdingSplitter(kernel = GaussianKernel(1.0)),
  ]
  c = compare(methods, data; reference = R)
  @test length(c.qualities) == 2
  @test all(isfinite, c.qualities)
  @test isapprox(c.qualities[2], splitquality(data, c.results[2]; reference = R); atol = 1e-12)
  cg = compare(methods, data; reference = R, kernel = GaussianKernel())
  @test cg.kernel isa GaussianKernel{Float64}
end
```

Append to `test/test_properties.jl` inside the top-level testset:

```julia
  @testset "reference-targeted splits beat random subsets against the reference" begin
    rng = MersenneTwister(620)
    data = randn(rng, 300, 3)
    R = data[data[:, 1] .> 0.3, :]
    for s in (
      SupportPointSplitter(max_iterations = 200, rng = MersenneTwister(621)),
      HerdingSplitter(kernel = GaussianKernel(1.0)),
    )
      r = datasplit(s, data; reference = R)
      q = splitquality(data, r; reference = R)
      n_test = length(test_indices(r))
      random_qs = map(1:25) do i
        perm = randperm(MersenneTwister(6_000 + i), 300)
        fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, r.method)
        splitquality(data, fake; reference = R)
      end
      @test q < mean(random_qs)
    end
  end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. test/test_quality.jl`
Expected: `unsupported keyword argument "reference"`

- [ ] **Step 3: Implement**

`src/quality.jl`: add `reference = nothing, reference_weights = nothing` to `splitquality`; at the top of the body:

```julia
  if reference !== nothing
    weights === nothing || throw(
      ArgumentError("with a reference, weight the reference (reference_weights), not the data"),
    )
    prep = fit_preprocessor(reference; weights = reference_weights, extra = data)
    R = apply_preprocessor(prep, reference)
    Xs = apply_preprocessor(prep, data)[_selected_indices(result), :]
    k = isresolved(kernel) ? kernel : resolve(kernel, R, rng, reference_weights)
    chosen = if subsample !== nothing
      Subsample(subsample, repeats)
    elseif estimator !== nothing
      estimator
    elseif size(Xs, 1) + size(R, 1) <= exact_threshold
      Exact()
    else
      _fallback_estimator(k)
    end
    return mmd(Xs, R, k; estimator = chosen, rng, n_threads, weights_y = reference_weights)
  end
  reference_weights === nothing ||
    throw(ArgumentError("reference_weights needs a reference"))
```

then the existing body unchanged. Add near `splitquality`:

```julia
_selected_indices(r::SplitResult) =
  r.selected === :test ? r.test_indices : r.train_indices
```

Docstring paragraph:

```text
`reference` (with optional `reference_weights`) scores the rows on the
`result.selected` side against `reference` instead of comparing train with
test: preprocessing is fit on `reference` and applied to both, and the
discrepancy is between the selected rows (uniform) and the reference
(weighted). `weights` cannot be combined with `reference`.
```

`src/comparison.jl`: add `reference = nothing, reference_weights = nothing`; forward to `datasplit`; resolve a `:median` scoring kernel on `apply_preprocessor(fit_preprocessor(reference; weights = reference_weights, extra = data), reference)` when `reference !== nothing` (with `reference_weights`), else as now; forward both to `splitquality`. Docstring: "`reference` and `reference_weights` are forwarded to `datasplit` and `splitquality`, and a `:median` scoring kernel is then resolved on the encoded reference."

- [ ] **Step 4: Run the tests**

Run: `julia --project=. test/test_quality.jl`, `test/test_comparison.jl`, `test/test_properties.jl`, then the full suite.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/quality.jl src/comparison.jl test/test_quality.jl test/test_comparison.jl test/test_properties.jl
git commit -m "feat: Score splits against a reference in splitquality and compare

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Docs and AGENTS.md

**Files:**

- Modify: `docs/src/10-methods.md`, `docs/src/85-roadmap.md`, `docs/src/index.md` (Quick start: one `selectrows` line), `AGENTS.md`

- [ ] **Step 1: Methods page**

Append to `docs/src/10-methods.md`:

```markdown
## [Reference distribution and `selectrows`](@id reference-distribution)

By default the chosen rows approximate the distribution of the data they
are drawn from. Passing `reference` (a second sample with the same columns,
optionally weighted by `reference_weights`) makes them approximate the
reference instead, while the candidates stay the rows of `data`. Only two
steps of the procedure change:

1. **Preprocess.** The transform is fit on the reference and applied to
   both sets: categorical levels are the union of both sets' levels, columns
   that are constant on the reference are dropped, and the reference's mean
   and scale (weighted forms with `reference_weights`) standardize every
   row.
3. **Resolve the kernel.** A `:median` bandwidth is resolved on the encoded
   reference.

Step 4 keeps its rules with the reference as the target measure: the data
term of the energy distance or MMD² runs over the reference rows
``r_1, \dots, r_M`` with weights ``\bar v_l``, the support points start at
rows of `data` and are rounded to rows of `data`, and kernel herding's data
term becomes ``\sum_l \bar v_l\, k(x, r_l)`` for every candidate ``x``.
`weights` for the data cannot be combined with `reference`.

`selectrows(splitter, data, n; ...)` runs the same steps and returns the `n`
chosen row indices without forming a partition; `datasplit` is `selectrows`
with `n` set by `ratio` plus the complement, and `result.selected` names the
side that holds the chosen rows. `splitquality(data, result; reference)`
then measures the chosen rows against the reference, which is the quantity
the splitter minimized.
```

- [ ] **Step 2: Roadmap**

In `docs/src/85-roadmap.md`: Current-state row `| Reference (target) distribution | not supported | The reference is always the data itself. |` → `| Reference (target) distribution | done |`reference`/`reference_weights` on `selectrows`,`datasplit`,`splitquality`,`compare`; see [Methods](@ref reference-distribution). |`; add a row `|`selectrows` (selection without a partition) | done | Returns the chosen row indices; `datasplit`builds on it. |` after it. M2 section: first word `Planned, depends on M1.` → `Done (2026-09-03).`; append a bullet "`selectrows(splitter, data, n; reference, reference_weights)` ships alongside, as the selection-only entry point the embedding workflow (M5) needs."; changelog line `- 2026-09-03: M2 (reference distribution) and`selectrows`done.`

- [ ] **Step 3: index.md and AGENTS.md**

`docs/src/index.md`, Quick start: after the existing `datasplit` example line(s), add one line `idx = selectrows(SupportPointSplitter(), data, 100; reference = target_sample)` with a one-sentence comment, matching the block's style (read the block first).

`AGENTS.md` Gotchas: add

```markdown
- `reference`/`reference_weights` (on `selectrows`, `datasplit`, `splitquality`,
  `compare`) define the target measure; candidates are always rows of
  `data`. Preprocessing is fit on the reference (`fit_preprocessor`) and
  applied to both; `weights` and `reference` are mutually exclusive;
  `reference = nothing` must stay bit-identical to the untargeted path.
  `SplitResult.selected` names the side holding the chosen rows.
```

- [ ] **Step 4: Build docs and commit**

`julia --project=docs docs/make.jl 2>&1 | grep -iE "error|warning: (invalid|unresolved|missing)"` prints nothing beyond the repository-URL warning; `ls docs/build/10-methods/index.html`.

```bash
git add docs/src/10-methods.md docs/src/85-roadmap.md docs/src/index.md AGENTS.md
git commit -m "docs: Describe the reference distribution and select

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: splitiq parity

**Files:**

- Modify: `splitiq/src/splitiq/split.py`, `quality.py`, `__init__.py`
- Create: `splitiq/tests/test_reference.py`
- Modify: `splitiq/docs/getting-started.md`, `docs/src/30-python.md`

**Interfaces:**

- Produces: `select_rows(data, n, *, method='support_points', kernel='energy', bandwidth='median', kappa=None, max_iterations=500, tolerance=1e-10, n_threads=None, seed=None, weights=None, reference=None, reference_weights=None) -> np.ndarray` (0-based, selection order); `datasplit(..., reference=None, reference_weights=None)`; `splitquality(..., reference=None, reference_weights=None)`; `SplitResult.selected: Literal['test', 'train']`.

- [ ] **Step 1: Write the failing tests**

`splitiq/tests/test_reference.py`:

```python
"""Parity tests for select and the reference distribution in splitiq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import datasplit, select_rows, splitquality


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    data = np.random.default_rng(seed).standard_normal((300, 2))
    return data, data[data[:, 0] > 0]


def test_select_returns_n_distinct_zero_based_indices() -> None:
    data, _ = _data(1)
    idx = select_rows(data, 60, seed=3, max_iterations=40)
    assert idx.shape == (60,)
    assert len(set(idx.tolist())) == 60
    assert idx.min() >= 0 and idx.max() < 300


def test_select_matches_the_selected_side_of_datasplit() -> None:
    data, _ = _data(2)
    result = datasplit(data, ratio=0.2, seed=4, max_iterations=40)
    idx = select_rows(data, 60, seed=4, max_iterations=40)
    assert result.selected == 'test'
    assert sorted(idx.tolist()) == sorted(result.test_indices.tolist())


def test_selected_side_follows_ratio() -> None:
    data, _ = _data(3)
    assert datasplit(data, ratio=0.8, seed=1, max_iterations=10).selected == 'train'


def test_reference_concentrates_the_selection() -> None:
    data, ref = _data(5)
    plain = select_rows(data, 60, seed=6, max_iterations=100)
    targeted = select_rows(data, 60, seed=6, max_iterations=100, reference=ref)
    assert np.sum(data[targeted, 0] > 0) > np.sum(data[plain, 0] > 0)


def test_herding_and_gaussian_accept_reference() -> None:
    data, ref = _data(7)
    idx = select_rows(data, 40, method='herding', kernel='gaussian', bandwidth=1.0, reference=ref)
    assert np.sum(data[idx, 0] > 0) >= 35
    result = datasplit(
        data, ratio=0.2, kernel='gaussian', seed=8, max_iterations=60, reference=ref
    )
    assert result.bandwidth is not None


def test_reference_weights_and_dataframes() -> None:
    data, _ = _data(9)
    df = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1], 'g': ['a', 'b', 'c'] * 100})
    ref = df[df.x > 0]
    weights = np.random.default_rng(10).random(len(ref))
    idx = select_rows(df, 60, seed=11, max_iterations=40, reference=ref, reference_weights=weights)
    assert idx.shape == (60,)


def test_splitquality_against_reference_is_lower_for_the_targeted_split() -> None:
    data, ref = _data(12)
    targeted = datasplit(data, ratio=0.2, seed=13, max_iterations=150, reference=ref)
    plain = datasplit(data, ratio=0.2, seed=13, max_iterations=150)
    assert splitquality(data, targeted, reference=ref) < splitquality(data, plain, reference=ref)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'reference': np.ones((10, 3))},
        {'reference_weights': np.ones(10)},
        {'reference': np.ones((10, 2)), 'weights': np.ones(300)},
    ],
    ids=['column-mismatch', 'weights-without-reference', 'weights-and-reference'],
)
def test_bad_reference_arguments_raise_value_error(kwargs: dict) -> None:
    data, _ = _data(14)
    with pytest.raises(ValueError):
        select_rows(data, 10, seed=1, max_iterations=5, **kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

From `splitiq/`: `make test`
Expected: `ImportError: cannot import name 'select_rows'`

- [ ] **Step 3: Implement**

`split.py`:

- `SplitResult` gains `selected: Literal['test', 'train']` (documented in Attributes) placed after `ratio`, before `_julia_result`.
- `_to_split_result` sets `selected=str(result.selected)` (Julia `Symbol` → `str` gives `'test'`/`'train'`; verify with `str(jl.Symbol('test'))`, and use `str(result.selected).lstrip(':')` if the colon is included).
- Factor the splitter construction into `_build_splitter(jl, method, kernel_obj, ratio, kappa, max_iterations, tolerance, n_threads, rng)` (the current `if method == 'herding' … else …` block, including the herding-options validation), returning the Julia splitter.
- `datasplit(..., reference=None, reference_weights=None)`: convert with `to_julia_data(reference)` when not `None` and `to_weights(reference_weights)`; pass as keywords via a `_reference_kwargs(reference_julia, reference_weights_julia)` helper in `_convert.py` (returns `{}` / `{'reference': ...}` / plus `'reference_weights'`), alongside `_weights_kwarg`.
- New `select_rows(data, n, *, method, kernel, bandwidth, kappa, max_iterations, tolerance, n_threads, seed, weights, reference, reference_weights) -> np.ndarray`: builds the splitter the same way (ratio is irrelevant; pass `0.5`), calls `jl.selectrows(splitter, julia_data, int(n), **kwargs)` inside `_translate_error()`, returns `to_python_indices(...)`. Google docstring with Args/Returns/Raises (ValueError for unknown method/kernel, herding options, and Julia `ArgumentError`s: `n` out of range, mismatched reference, weights with reference).
- `quality.py` `splitquality(..., reference=None, reference_weights=None)` forwards the same way; docstring Args updated.
- `__init__.py`: import and export `selectrows`.

- [ ] **Step 4: Run gates**

From `splitiq/`: `make test && make lint && make format && make typecheck`.

- [ ] **Step 5: Docs**

Append to `splitiq/docs/getting-started.md`:

````markdown
## Selecting rows toward a reference

`select_rows` returns the indices of `n` rows without forming a partition, and
`reference` makes the chosen rows follow a target sample instead of the
data itself:

```python
import numpy as np
from splitiq import select_rows, splitquality, datasplit

data = np.random.default_rng(0).standard_normal((5000, 16))
target = np.random.default_rng(1).standard_normal((800, 16)) + 0.5

idx = select_rows(data, 500, seed=42, reference=target)          # 0-based row indices
result = datasplit(data, ratio=0.1, seed=42, reference=target)
print(result.selected, splitquality(data, result, reference=target))
```

`reference_weights` weights the reference rows; `weights` cannot be
combined with `reference`.
````

Append a matching "## Selecting rows toward a reference" section to `docs/src/30-python.md` with the same example and a sentence pointing at `[Methods](@ref reference-distribution)`.

- [ ] **Step 6: Commit**

```bash
git add splitiq/src/splitiq/split.py splitiq/src/splitiq/quality.py splitiq/src/splitiq/__init__.py splitiq/src/splitiq/_convert.py splitiq/tests/test_reference.py splitiq/docs/getting-started.md docs/src/30-python.md
git commit -m "feat(splitiq): Expose select and the reference distribution

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Quality gate

- [ ] `julia -t 4 --project=. -e "using Pkg; Pkg.test()"` and `julia -t 1 --project=. -e "using Pkg; Pkg.test()"`: all pass.
- [ ] `julia --project=docs docs/make.jl 2>&1 | grep -iE "error|warning: (invalid|unresolved|missing|no doc)"`: nothing beyond the repository-URL warning.
- [ ] From `splitiq/`: `make test && make lint && make typecheck && make docs`.
- [ ] `pre-commit run --all-files` clean (CI pins Julia 1.10 + JuliaFormatter 1.0.62, the same as pre-commit; do not use a Julia 1.12 formatter).
- [ ] `git diff origin/main...HEAD -- test/ | grep "^-" | grep -v "^---"` shows only the lines this plan allowed to change (the M1 follow-up `_standardize!`/`_encode` test lines in Task 1); every new keyword defaults to `nothing`.
- [ ] Report; the PR is opened only after the user confirms.
