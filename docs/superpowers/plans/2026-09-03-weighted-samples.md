# Weighted Samples (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-row sample weights to every discrepancy, optimizer, splitter, and diagnostic in SPlit.jl, and expose the same keywords in the splitiq Python package.

**Architecture:** Weights enter as a `weights` keyword beside the data (`datasplit`, `splitquality`, `compare`) or as `weights_x`/`weights_y` on the two-sample discrepancies. `nothing` (the default) dispatches to the existing methods untouched, so unweighted results stay bit-identical; weighted behavior is added as new methods that take normalized (`w̄`, sum one) or mean-one (`ŵ`) weight vectors. Inside the hot loops a single code path multiplies by `ŵᵢ`, which is exactly `1.0` for uniform weights.

**Tech Stack:** Julia 1.10+ (package), Documenter.jl (docs), Python 3.13 + juliacall (splitiq), pytest, uv, pre-commit (JuliaFormatter, markdownlint).

**Spec:** `docs/superpowers/specs/2026-09-03-weighted-samples-design.md`

## Global Constraints

- Existing public signatures and numerical results are unchanged; every new keyword defaults to `nothing`.
- Estimator/kernel combinations are methods of `_energydistance`/`_mmd`: add a method, never an `if` (`AGENTS.md`).
- All randomness flows through the caller's `rng`; nothing in `src/` seeds or prints on a default path.
- The MM sweep in `optimizer.jl` stays allocation-free inside the per-point loop.
- Correctness is judged against Mak & Joseph (2018), Joseph & Vakayil (2021), Chen, Welling & Smola (2010); do not cite or compare with other implementations anywhere.
- Docs: do not rewrite existing sections; add one "Weighted samples" section per page.
- Every capability added to Julia is added to splitiq in this same branch, with tests.
- Run a single Julia test file with `julia --project=. test/<file>.jl`; the full suite with `julia --project=. -e "using Pkg; Pkg.test()"`.
- Commit messages: `<type>: <Description>` (capitalized), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Pre-commit runs on commit; never bypass it.
- Work only inside the worktree `.claude/worktrees/feat-weighted-samples` (branch `feat/weighted-samples`).

---

## File structure

| File | Responsibility |
|---|---|
| `src/weights.jl` (new) | Validation and normalization helpers shared by every weighted method |
| `src/preprocessing.jl` | `preprocess(data, weights)`: weighted standardization (new method) |
| `src/kernels.jl` | `resolve(kernel, data, rng, weights)`: weighted median heuristic (new method) |
| `src/quality.jl` | Weighted exact energy distance / MMD, `weights_x`/`weights_y` keywords, `splitquality(...; weights)` |
| `src/estimators.jl` | Weighted `RandomSlices` (1-D prefix sums) and `RandomFeatures` (weighted feature means) |
| `src/optimizer.jl` | `ŵ` in the MM sweep and the MMD gradient; `weights` and `_subsampling` keywords on `support_points` |
| `src/herding.jl` | Weighted data term; `weights` on `herd` and `datasplit(::HerdingSplitter)` |
| `src/splitter.jl` | `datasplit(::SupportPointSplitter, data; weights)` |
| `src/comparison.jl` | `compare(...; weights)` forwarding |
| `benchmark/weighted_kappa.jl` (new) | The `kappa` rule experiment; writes `docs/src/assets/benchmarks/weighted_kappa.md` |
| `docs/src/10-methods.md`, `25-design-experiments.md`, `85-roadmap.md`, `30-python.md`, `AGENTS.md` | One added section each |
| `splitiq/src/splitiq/_convert.py`, `split.py`, `quality.py` | `to_weights`, `weights` / `weights_x` / `weights_y` keywords |
| `splitiq/tests/test_weights.py` (new), `splitiq/docs/getting-started.md` | Parity tests and docs |

---

### Task 1: Weight helpers

**Files:**

- Create: `src/weights.jl`
- Modify: `src/SPlit.jl` (add `include("weights.jl")` right after `include("kernels.jl")`)
- Create: `test/test_weights.jl`
- Modify: `test/runtests.jl` (add `include("test_weights.jl")` after `test_kernels.jl`)

**Interfaces:**

- Produces:
  - `_check_weights(weights::AbstractVector, N::Int) -> weights` (throws `ArgumentError`)
  - `_normalize_weights(weights::AbstractVector, N::Int) -> Vector{Float64}` (sum one)
  - `_mean_one_weights(weights::AbstractVector) -> Vector{Float64}` (mean one; `ones` stay exactly `1.0`)
  - `_uniform_weights(n::Int) -> Vector{Float64}` (`fill(1/n, n)`)
  - `_side_weights(::Nothing, n) -> _uniform_weights(n)`, `_side_weights(w::AbstractVector, n) -> _normalize_weights(w, n)`

- [ ] **Step 1: Write the failing tests**

`test/test_weights.jl`:

```julia
using Test
using SPlit

@testset "weight helpers" begin
  @testset "validation" begin
    @test_throws ArgumentError SPlit._check_weights([1.0, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([1.0, -1.0, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([1.0, NaN, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([1.0, Inf, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([0.0, 0.0, 0.0], 3)
    @test SPlit._check_weights([0.0, 1.0, 2.0], 3) == [0.0, 1.0, 2.0]
  end

  @testset "normalization" begin
    w = SPlit._normalize_weights([1, 3], 2)
    @test w isa Vector{Float64}
    @test w == [0.25, 0.75]
    @test SPlit._normalize_weights([2.0, 2.0, 2.0], 3) == fill(1 / 3, 3)
    @test SPlit._uniform_weights(4) == fill(0.25, 4)
    @test SPlit._side_weights(nothing, 4) == fill(0.25, 4)
    @test SPlit._side_weights([1, 1, 2], 3) == [0.25, 0.25, 0.5]
  end

  @testset "mean-one scaling keeps uniform weights exactly 1.0" begin
    @test SPlit._mean_one_weights(ones(7)) == ones(7)
    @test all(SPlit._mean_one_weights(fill(0.3, 5)) .== 1.0)
    w = SPlit._mean_one_weights([1.0, 3.0])
    @test w == [0.5, 1.5]
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_weights.jl`
Expected: `UndefVarError: _check_weights not defined`

- [ ] **Step 3: Implement `src/weights.jl`**

```julia
"""
Sample-weight helpers shared by every weighted method. Weights are
per-row, non-negative, finite, with a positive sum. Two scalings are used:
`w̄` (sum one) for discrepancies and data terms, and `ŵ` (mean one) inside
the optimizers, so that uniform weights become exactly `1.0` and the
weighted arithmetic reproduces the unweighted arithmetic bit for bit.

Not part of the public API.
"""

# Validate `weights` for `N` rows: length N, all finite, none negative,
# positive sum. Returns `weights` unchanged.
function _check_weights(weights::AbstractVector, N::Int)
  length(weights) == N || throw(
    ArgumentError("weights must have one entry per row ($N), got $(length(weights))"),
  )
  all(w -> isfinite(w) && w >= 0, weights) ||
    throw(ArgumentError("weights must be finite and non-negative"))
  sum(weights) > 0 || throw(ArgumentError("weights must not all be zero"))
  return weights
end

# `w̄`: weights scaled to sum one, as a fresh Float64 vector.
function _normalize_weights(weights::AbstractVector, N::Int)
  _check_weights(weights, N)
  w = Vector{Float64}(weights)
  w ./= sum(w)
  return w
end

# `ŵ`: weights scaled to mean one, as a fresh Float64 vector. For uniform
# weights the scale factor is exactly 1.0, so the result is exactly ones.
function _mean_one_weights(weights::AbstractVector)
  w = Vector{Float64}(weights)
  w .*= length(w) / sum(w)
  return w
end

_uniform_weights(n::Int) = fill(1.0 / n, n)

# Normalized weights for one side of a two-sample discrepancy: `nothing`
# means uniform.
_side_weights(::Nothing, n::Int) = _uniform_weights(n)
_side_weights(weights::AbstractVector, n::Int) = _normalize_weights(weights, n)
```

Add to `src/SPlit.jl` after `include("kernels.jl")`:

```julia
include("weights.jl")
```

Add to `test/runtests.jl` after `include("test_kernels.jl")`:

```julia
  include("test_weights.jl")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_weights.jl`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/weights.jl src/SPlit.jl test/test_weights.jl test/runtests.jl
git commit -m "feat: Add sample-weight validation and normalization helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Weighted preprocessing and weighted median bandwidth

**Files:**

- Modify: `src/preprocessing.jl` (extract the encoding step; add weighted standardization and `preprocess(data, weights)`)
- Modify: `src/kernels.jl` (add `resolve(kernel, data, rng, weights)` methods)
- Test: `test/test_preprocessing.jl`, `test/test_kernels.jl`

**Interfaces:**

- Consumes: `_normalize_weights`, `_check_weights` (Task 1)
- Produces:
  - `preprocess(data, ::Nothing) -> Matrix{Float64}` (identical to `preprocess(data)`)
  - `preprocess(data, weights::AbstractVector) -> Matrix{Float64}` (weighted standardization)
  - `resolve(kernel, data, rng, ::Nothing)` (identical to `resolve(kernel, data, rng)`)
  - `resolve(::GaussianKernel{Symbol}, data, rng, weights::AbstractVector)` (weight-proportional row draw)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_preprocessing.jl` (inside a new top-level `@testset`):

```julia
@testset "weighted preprocess" begin
  @testset "nothing dispatches to the unweighted method" begin
    data = randn(MersenneTwister(30), 50, 3)
    @test SPlit.preprocess(data, nothing) == SPlit.preprocess(data)
  end

  @testset "weighted mean 0 and weighted variance 1 per column" begin
    rng = MersenneTwister(31)
    data = randn(rng, 80, 3) .* [1.0 5.0 0.1] .+ [2.0 -1.0 0.0]
    w = rand(rng, 80)
    X = SPlit.preprocess(data, w)
    wn = w ./ sum(w)
    for j = 1:3
      μ = sum(wn .* X[:, j])
      σ2 = sum(wn .* (X[:, j] .- μ) .^ 2) / (1 - sum(abs2, wn))
      @test isapprox(μ, 0.0; atol = 1e-12)
      @test isapprox(σ2, 1.0; atol = 1e-12)
    end
  end

  @testset "uniform weights match the unweighted result up to rounding" begin
    data = randn(MersenneTwister(32), 60, 2)
    X_unweighted = SPlit.preprocess(data)
    X_weighted = SPlit.preprocess(data, ones(60))
    @test isapprox(X_weighted, X_unweighted; atol = 1e-12)
  end

  @testset "all weight on one row errors" begin
    w = zeros(10)
    w[3] = 1.0
    @test_throws ArgumentError SPlit.preprocess(randn(10, 2), w)
  end

  @testset "DataFrame with categoricals accepts weights" begin
    df = DataFrame(x = randn(MersenneTwister(33), 30), g = repeat(["a", "b", "c"], 10))
    X = SPlit.preprocess(df, ones(30))
    @test size(X) == (30, 3)
  end

  @testset "wrong length errors" begin
    @test_throws ArgumentError SPlit.preprocess(randn(10, 2), ones(9))
  end
end
```

Append to `test/test_kernels.jl`:

```julia
@testset "weighted median bandwidth" begin
  X = randn(MersenneTwister(40), 300, 2)
  @test SPlit.resolve(GaussianKernel(), X, MersenneTwister(1), nothing) ==
        SPlit.resolve(GaussianKernel(), X, MersenneTwister(1))
  @test SPlit.resolve(EnergyKernel(), X, MersenneTwister(1), ones(300)) == EnergyKernel()
  @test SPlit.resolve(GaussianKernel(2.0), X, MersenneTwister(1), ones(300)) ==
        GaussianKernel(2.0)

  # Two clusters far apart, more rows than the 1_000 the heuristic draws:
  # weight concentrated on one cluster makes most drawn pairs intra-cluster,
  # so the median distance drops.
  Y = vcat(randn(MersenneTwister(41), 750, 2), randn(MersenneTwister(42), 750, 2) .+ 20.0)
  w = vcat(fill(100.0, 750), fill(1e-3, 750))
  σ_uniform = SPlit.resolve(GaussianKernel(), Y, MersenneTwister(3)).bandwidth
  σ_weighted = SPlit.resolve(GaussianKernel(), Y, MersenneTwister(3), w).bandwidth
  @test σ_weighted < σ_uniform
  @test_throws ArgumentError SPlit.resolve(GaussianKernel(), Y, MersenneTwister(3), ones(10))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. test/test_preprocessing.jl` and `julia --project=. test/test_kernels.jl`
Expected: `MethodError: no method matching preprocess(::Matrix{Float64}, ::Nothing)` and `no method matching resolve(..., ::Nothing)`

- [ ] **Step 3: Implement**

In `src/preprocessing.jl`, replace the two `preprocess` bodies so encoding is one function and standardization another, keeping behavior identical:

```julia
# Encoded, unstandardized column matrix: constant columns dropped, categorical
# columns Helmert-encoded. Shared by the unweighted and weighted `preprocess`.
function _encode(data::AbstractMatrix)
  any(ismissing, data) && throw(ArgumentError("Dataset contains missing value(s)."))
  all(x -> x isa Number, data) ||
    throw(ArgumentError("Matrix input must contain only numeric values."))
  keep = [!_is_constant(view(data, :, j)) for j in axes(data, 2)]
  any(keep) || throw(ArgumentError("All columns are constant."))
  return Float64.(data[:, keep])
end

_encode(data::AbstractVector) = _encode(reshape(collect(data), :, 1))

function _encode(data::DataFrame)
  for col in eachcol(data)
    any(ismissing, col) && throw(ArgumentError("Dataset contains missing value(s)."))
  end

  columns = Vector{Vector{Float64}}()
  for name in names(data)
    col = data[!, name]
    if _is_categorical(col)
      levels_ = _canonical_levels(col)
      length(levels_) <= 1 && continue
      H = helmert_matrix(length(levels_))
      index = Dict(l => i for (i, l) in enumerate(levels_))
      for j in axes(H, 2)
        push!(columns, [H[index[v], j] for v in col])
      end
    elseif Base.nonmissingtype(eltype(col)) <: Number
      _is_constant(col) && continue
      push!(columns, Float64.(col))
    else
      throw(ArgumentError("Unsupported column type in column: $(name)"))
    end
  end

  isempty(columns) && throw(ArgumentError("All columns are constant."))
  # hcat(columns...) (not reduce) so a single column still yields an n×1 Matrix
  return hcat(columns...)
end

"""
    preprocess(data) -> Matrix{Float64}
    preprocess(data, weights) -> Matrix{Float64}

Validate and transform `data` for splitting: reject missing values, encode
categorical columns with Helmert contrasts, drop constant columns, and
standardize every remaining column. Accepts `AbstractMatrix`, `DataFrame`,
and `AbstractVector` inputs.

With `weights` (one non-negative entry per row), standardization uses the
weighted mean `μⱼ = Σ w̄ᵢ xᵢⱼ` and the unbiased weighted variance
`σⱼ² = Σ w̄ᵢ (xᵢⱼ − μⱼ)² / (1 − Σ w̄ᵢ²)` with `w̄` the weights scaled to sum
one, which reduces to the `n − 1` denominator of `std` for uniform weights;
the encoding steps are the same. `weights = nothing` is the unweighted
method.
"""
preprocess(data) = _standardize!(_encode(data))
preprocess(data, ::Nothing) = preprocess(data)
function preprocess(data, weights::AbstractVector)
  M = _encode(data)
  return _standardize!(M, _normalize_weights(weights, size(M, 1)))
end

# Weighted standardization in place, `w` scaled to sum one. The variance
# denominator 1 − Σ w² is the unbiased correction for normalized weights;
# for uniform weights it equals (n − 1)/n, matching `std`.
function _standardize!(M::Matrix{Float64}, w::Vector{Float64})
  correction = 1 - sum(abs2, w)
  correction > 0 ||
    throw(ArgumentError("weights must be positive on at least two rows"))
  for j in axes(M, 2)
    col = view(M, :, j)
    μ = sum(w .* col)
    σ = sqrt(sum(w .* (col .- μ) .^ 2) / correction)
    col .= (col .- μ) ./ σ
  end
  return M
end
```

Delete the old `preprocess(data::AbstractMatrix)`, `preprocess(data::AbstractVector)`, and `preprocess(data::DataFrame)` definitions (their bodies moved into `_encode`). Keep `_standardize!(M::Matrix{Float64})` as is.

In `src/kernels.jl`, after the existing `resolve` methods:

```julia
"""
    resolve(kernel, data, rng, weights)

Weighted form of [`resolve`](@ref): for `GaussianKernel(:median)` the rows
behind the median heuristic are drawn with probability proportional to
`weights` (without replacement), so the bandwidth reflects the weighted
distribution. `weights = nothing` is the unweighted method; numeric kernels
and `EnergyKernel` are returned unchanged.
"""
resolve(k::SplitKernel, data::AbstractMatrix, rng::AbstractRNG, ::Nothing) =
  resolve(k, data, rng)
resolve(k::EnergyKernel, ::AbstractMatrix, ::AbstractRNG, ::AbstractVector) = k
resolve(k::GaussianKernel{Float64}, ::AbstractMatrix, ::AbstractRNG, ::AbstractVector) = k
function resolve(
  ::GaussianKernel{Symbol},
  data::AbstractMatrix,
  rng::AbstractRNG,
  weights::AbstractVector,
)
  N = size(data, 1)
  _check_weights(weights, N)
  m = min(N, MEDIAN_HEURISTIC_ROWS)
  rows = m == N ? (1:N) : sample(rng, 1:N, Weights(Vector{Float64}(weights)), m; replace = false)
  D = pairwise(Euclidean(), view(data, rows, :); dims = 1)
  dists = [D[i, j] for i = 1:m for j = (i+1):m]
  σ = median(dists)
  σ > 0 || throw(
    ArgumentError(
      "median pairwise distance is zero; pass a numeric bandwidth to GaussianKernel",
    ),
  )
  return GaussianKernel(σ)
end
```

and change the `using StatsBase: sample` line in `src/kernels.jl` to `using StatsBase: sample, Weights`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. test/test_preprocessing.jl` and `julia --project=. test/test_kernels.jl`
Expected: all pass, including the pre-existing preprocessing tests.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.jl src/kernels.jl test/test_preprocessing.jl test/test_kernels.jl
git commit -m "feat: Add weighted standardization and a weighted median bandwidth

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Weighted exact energy distance and MMD (Exact and Subsample)

**Files:**

- Modify: `src/quality.jl`
- Test: `test/test_quality.jl`

**Interfaces:**

- Consumes: `_side_weights`, `_normalize_weights` (Task 1)
- Produces:
  - `_mean_pairwise(X, Y, wx::Vector{Float64}, wy::Vector{Float64}; block, n_threads)`
  - `_mean_kernel(k, X, Y, wx, wy; block, n_threads)`
  - `_exact_energydistance(X, Y, wx, wy; block, n_threads)`, `_exact_mmd(k, X, Y, wx, wy; block, n_threads)`
  - `energydistance(X, Y; weights_x = nothing, weights_y = nothing, ...)`, `mmd(X, Y, kernel; weights_x = nothing, weights_y = nothing, ...)`
  - Weighted methods `_energydistance(e, X, Y, wx, wy, rng, n_threads)` and `_mmd(e, k, X, Y, wx, wy, rng, n_threads)` for `Exact` and `Subsample`, plus the `_undefined` fallbacks.
  - `_renormalized(w::AbstractVector) -> Vector{Float64}`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_quality.jl`:

```julia
@testset "weighted energydistance and mmd" begin
  rng = MersenneTwister(50)
  X = randn(rng, 30, 2)
  Y = randn(rng, 25, 2) .+ 0.5

  @testset "uniform weights give exactly the unweighted value" begin
    @test energydistance(X, Y; weights_x = ones(30), weights_y = ones(25)) ==
          energydistance(X, Y)
    @test energydistance(X, Y; weights_x = fill(0.2, 30)) == energydistance(X, Y)
    k = GaussianKernel(1.0)
    @test mmd(X, Y, k; weights_x = ones(30), weights_y = ones(25)) == mmd(X, Y, k)
    @test mmd(X, Y, EnergyKernel(); weights_x = ones(30)) == energydistance(X, Y)
  end

  @testset "hand-computed weighted 1-D values" begin
    # X = {0, 1} with weights (3, 1), Y = {1}:
    # 2·E|X−Y| = 2·(0.75·1 + 0.25·0) = 1.5; E|X−X'| = 2·0.75·0.25·1 = 0.375; E|Y−Y'| = 0
    @test isapprox(
      energydistance(reshape([0.0, 1.0], :, 1), reshape([1.0], :, 1); weights_x = [3.0, 1.0]),
      1.5 - 0.375;
      atol = 1e-12,
    )
  end

  @testset "duplication invariance: weights as counts equal duplicated rows" begin
    Xdup = vcat(X[1:1, :], X)               # row 1 twice
    wx = vcat([2.0], ones(29))
    @test isapprox(energydistance(X, Y; weights_x = wx), energydistance(Xdup, Y); atol = 1e-12)
    k = GaussianKernel(0.8)
    @test isapprox(mmd(X, Y, k; weights_x = wx), mmd(Xdup, Y, k); atol = 1e-12)
    # both sides weighted
    Ydup = vcat(Y, Y[end:end, :])
    wy = vcat(ones(24), [2.0])
    @test isapprox(
      energydistance(X, Y; weights_x = wx, weights_y = wy),
      energydistance(Xdup, Ydup);
      atol = 1e-12,
    )
  end

  @testset "block accumulation matches the unblocked weighted value" begin
    wx = rand(MersenneTwister(51), 30)
    wy = rand(MersenneTwister(52), 25)
    a = energydistance(X, Y; weights_x = wx, weights_y = wy)
    b = SPlit._exact_energydistance(X, Y, wx ./ sum(wx), wy ./ sum(wy); block = 7)
    @test isapprox(a, b; atol = 1e-10)
  end

  @testset "Subsample with weights runs, and is exact below m" begin
    wx = rand(MersenneTwister(53), 30)
    exact = energydistance(X, Y; weights_x = wx)
    @test energydistance(X, Y; weights_x = wx, estimator = Subsample(100)) == exact
    big = randn(MersenneTwister(54), 400, 2)
    wbig = rand(MersenneTwister(55), 400)
    est = energydistance(
      big,
      Y;
      weights_x = wbig,
      estimator = Subsample(150, 20),
      rng = MersenneTwister(1),
    )
    @test isapprox(est, energydistance(big, Y; weights_x = wbig); rtol = 0.3)
    k = GaussianKernel(1.0)
    @test isapprox(
      mmd(big, Y, k; weights_x = wbig, estimator = Subsample(150, 20), rng = MersenneTwister(1)),
      mmd(big, Y, k; weights_x = wbig);
      rtol = 0.3,
    )
  end

  @testset "validation" begin
    @test_throws ArgumentError energydistance(X, Y; weights_x = ones(29))
    @test_throws ArgumentError energydistance(X, Y; weights_y = -ones(25))
    @test_throws ArgumentError mmd(X, Y, GaussianKernel(1.0); weights_x = zeros(30))
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_quality.jl`
Expected: `MethodError` / `unsupported keyword argument "weights_x"`

- [ ] **Step 3: Implement**

In `src/quality.jl`, add `using LinearAlgebra: dot` next to the other `using` lines. After the unweighted `_mean_pairwise`, add:

```julia
# Weighted mean pairwise distance Σᵢⱼ wxᵢ wyⱼ ‖xᵢ − yⱼ‖ with both weight
# vectors scaled to sum one (so no 1/(nm) division), block-wise.
function _mean_pairwise(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  pairs = _block_pairs(size(X, 1), size(Y, 1), block)
  return _threaded_block_sum(pairs, n_threads) do (xblock, yblock)
    i0, i1 = xblock
    j0, j1 = yblock
    @views D = pairwise(Euclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1)
    @views dot(wx[i0:i1], D, wy[j0:j1])
  end
end

function _exact_energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  return 2 * _mean_pairwise(X, Y, wx, wy; block, n_threads) -
         _mean_pairwise(X, X, wx, wx; block, n_threads) -
         _mean_pairwise(Y, Y, wy, wy; block, n_threads)
end
```

After the unweighted `_mean_kernel`/`_exact_mmd`, add:

```julia
# Weighted mean kernel value Σᵢⱼ wxᵢ wyⱼ k(xᵢ, yⱼ), weights scaled to sum one.
function _mean_kernel(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  scale = -1 / (2 * k.bandwidth^2)
  pairs = _block_pairs(size(X, 1), size(Y, 1), block)
  return _threaded_block_sum(pairs, n_threads) do (xblock, yblock)
    i0, i1 = xblock
    j0, j1 = yblock
    @views D = pairwise(SqEuclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1)
    D .= exp.(scale .* D)
    @views dot(wx[i0:i1], D, wy[j0:j1])
  end
end

function _exact_mmd(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  return _mean_kernel(k, X, X, wx, wx; block, n_threads) +
         _mean_kernel(k, Y, Y, wy, wy; block, n_threads) -
         2 * _mean_kernel(k, X, Y, wx, wy; block, n_threads)
end

# Weights of a drawn subsample rescaled to sum one.
function _renormalized(w::AbstractVector)
  s = sum(w)
  s > 0 || throw(ArgumentError("a subsample drew only zero-weight rows; use a larger m"))
  return w ./ s
end
```

Change the `mmd` signature and body to:

```julia
function mmd(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  kernel::SplitKernel;
  estimator::DiscrepancyEstimator = Exact(),
  rng::AbstractRNG = Random.default_rng(),
  n_threads::Int = Threads.nthreads(),
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  weights_x::Union{Nothing,AbstractVector} = nothing,
  weights_y::Union{Nothing,AbstractVector} = nothing,
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))
  subsample === nothing || (estimator = Subsample(subsample, repeats))
  kernel isa EnergyKernel &&
    return energydistance(X, Y; estimator, rng, n_threads, weights_x, weights_y)
  k = isresolved(kernel) ? kernel : resolve(kernel, vcat(X, Y), rng)
  weights_x === nothing && weights_y === nothing && return _mmd(estimator, k, X, Y, rng, n_threads)
  wx = _side_weights(weights_x, size(X, 1))
  wy = _side_weights(weights_y, size(Y, 1))
  return _mmd(estimator, k, X, Y, wx, wy, rng, n_threads)
end
```

and append to its docstring, as a new final paragraph:

```text
`weights_x` and `weights_y` (one non-negative entry per row of `X` and
`Y`; `nothing` means uniform) turn each sample into a weighted empirical
distribution, `Σᵢ w̄ᵢ δ(xᵢ)` with `w̄` scaled to sum one, and the statistic
becomes `Σ w̄ᵢ w̄ₖ k(xᵢ, xₖ) + Σ v̄ⱼ v̄ₗ k(yⱼ, yₗ) − 2 Σ w̄ᵢ v̄ⱼ k(xᵢ, yⱼ)`.
`Subsample` draws rows uniformly and rescales the drawn weights to sum
one. Weights proportional to duplication counts are equivalent to
duplicating rows.
```

Add the weighted `_mmd` methods after the unweighted ones:

```julia
_mmd(::Exact, k, X, Y, wx, wy, rng, n_threads) = _exact_mmd(k, X, Y, wx, wy; n_threads)
function _mmd(e::Subsample, k, X, Y, wx, wy, rng, n_threads)
  (size(X, 1) <= e.m && size(Y, 1) <= e.m) && return _exact_mmd(k, X, Y, wx, wy; n_threads)
  estimates = Vector{Float64}(undef, e.repeats)
  for r = 1:e.repeats
    xs = sample(rng, 1:size(X, 1), min(e.m, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(e.m, size(Y, 1)); replace = false)
    estimates[r] = _exact_mmd(
      k,
      X[xs, :],
      Y[ys, :],
      _renormalized(wx[xs]),
      _renormalized(wy[ys]);
      n_threads,
    )
  end
  return mean(estimates)
end
_mmd(e::DiscrepancyEstimator, k, X, Y, wx, wy, rng, n_threads) =
  _undefined(e, "mmd under $(nameof(typeof(k))) with weights")
```

Change the `energydistance` signature and body to:

```julia
function energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix;
  estimator::DiscrepancyEstimator = Exact(),
  rng::AbstractRNG = Random.default_rng(),
  n_threads::Int = Threads.nthreads(),
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  weights_x::Union{Nothing,AbstractVector} = nothing,
  weights_y::Union{Nothing,AbstractVector} = nothing,
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))
  subsample === nothing || (estimator = Subsample(subsample, repeats))
  weights_x === nothing &&
    weights_y === nothing &&
    return _energydistance(estimator, X, Y, rng, n_threads)
  wx = _side_weights(weights_x, size(X, 1))
  wy = _side_weights(weights_y, size(Y, 1))
  return _energydistance(estimator, X, Y, wx, wy, rng, n_threads)
end
```

append to its docstring:

```text
`weights_x` and `weights_y` (one non-negative entry per row; `nothing`
means uniform) give the energy distance between the weighted empirical
distributions `Σᵢ w̄ᵢ δ(xᵢ)` and `Σⱼ v̄ⱼ δ(yⱼ)`, with the weights scaled to
sum one: `2 Σ w̄ᵢ v̄ⱼ ‖xᵢ − yⱼ‖ − Σ w̄ᵢ w̄ₖ ‖xᵢ − xₖ‖ − Σ v̄ⱼ v̄ₗ ‖yⱼ − yₗ‖`.
Weights proportional to duplication counts are equivalent to duplicating
rows.
```

and add the weighted `_energydistance` methods after the unweighted ones:

```julia
_energydistance(::Exact, X, Y, wx, wy, rng, n_threads) =
  _exact_energydistance(X, Y, wx, wy; n_threads)
function _energydistance(e::Subsample, X, Y, wx, wy, rng, n_threads)
  (size(X, 1) <= e.m && size(Y, 1) <= e.m) &&
    return _exact_energydistance(X, Y, wx, wy; n_threads)
  estimates = Vector{Float64}(undef, e.repeats)
  for r = 1:e.repeats
    xs = sample(rng, 1:size(X, 1), min(e.m, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(e.m, size(Y, 1)); replace = false)
    estimates[r] = _exact_energydistance(
      X[xs, :],
      Y[ys, :],
      _renormalized(wx[xs]),
      _renormalized(wy[ys]);
      n_threads,
    )
  end
  return mean(estimates)
end
_energydistance(e::DiscrepancyEstimator, X, Y, wx, wy, rng, n_threads) =
  _undefined(e, "the energy distance with weights")
```

Also extend the vector convenience method so weights pass through:

```julia
energydistance(X::AbstractVector, Y::AbstractVector; kwargs...) = energydistance(
  reshape(collect(Float64, X), :, 1),
  reshape(collect(Float64, Y), :, 1);
  kwargs...,
)
```

(already forwards `kwargs...`; no change needed, just confirm.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_quality.jl`
Expected: all pass (`RandomSlices`/`RandomFeatures` with weights are still undefined; they are Task 4 and not exercised here).

- [ ] **Step 5: Commit**

```bash
git add src/quality.jl test/test_quality.jl
git commit -m "feat: Add weights_x and weights_y to energydistance and mmd

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Weighted RandomSlices and RandomFeatures

**Files:**

- Modify: `src/estimators.jl`
- Modify: `src/quality.jl` (two dispatch lines)
- Test: `test/test_estimators.jl`, `test/test_quality.jl`

**Interfaces:**

- Consumes: weighted `_energydistance`/`_mmd` dispatch (Task 3)
- Produces:
  - `_weighted_within_abs(sorted, w) -> Float64` (`Σᵢₖ wᵢ wₖ |aᵢ − aₖ|`)
  - `_weighted_cross_abs(a_sorted, w, b_sorted, v) -> Float64`
  - `_ed1d(a, w, b, v) -> Float64`
  - `_sliced_energydistance(X, Y, wx, wy, k, rng)`
  - `_feature_mean(φ, X, w; block)`, `_rff_mmd(k, X, Y, wx, wy, D, rng)`
  - `_energydistance(e::RandomSlices, X, Y, wx, wy, rng, n_threads)`, `_mmd(e::RandomFeatures, k::GaussianKernel{Float64}, X, Y, wx, wy, rng, n_threads)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_estimators.jl`:

```julia
@testset "weighted one-dimensional energy distance" begin
  rng = MersenneTwister(60)
  a = randn(rng, 40)
  b = randn(rng, 35) .+ 0.3
  w = rand(rng, 40)
  v = rand(rng, 35)
  w ./= sum(w)
  v ./= sum(v)
  brute = 2 * sum(w[i] * v[j] * abs(a[i] - b[j]) for i = 1:40, j = 1:35) -
          sum(w[i] * w[k] * abs(a[i] - a[k]) for i = 1:40, k = 1:40) -
          sum(v[j] * v[l] * abs(b[j] - b[l]) for j = 1:35, l = 1:35)
  @test isapprox(SPlit._ed1d(a, w, b, v), brute; atol = 1e-12)
  # uniform weights reduce to the unweighted routine
  @test isapprox(SPlit._ed1d(a, fill(1 / 40, 40), b, fill(1 / 35, 35)), SPlit._ed1d(a, b); atol = 1e-12)
end
```

Append to the `"weighted energydistance and mmd"` testset in `test/test_quality.jl` (or as a new testset after it):

```julia
@testset "weighted RandomSlices and RandomFeatures" begin
  rng = MersenneTwister(61)
  X = randn(rng, 200, 3)
  Y = randn(rng, 180, 3) .+ 0.4
  Xdup = vcat(X[1:1, :], X)
  wx = vcat([2.0], ones(199))

  # same rng ⇒ same directions/features, so duplication invariance is exact
  @test isapprox(
    energydistance(X, Y; weights_x = wx, estimator = RandomSlices(32), rng = MersenneTwister(5)),
    energydistance(Xdup, Y; estimator = RandomSlices(32), rng = MersenneTwister(5));
    atol = 1e-10,
  )
  @test energydistance(X, Y; weights_x = ones(200), estimator = RandomSlices(32), rng = MersenneTwister(5)) ==
        energydistance(X, Y; estimator = RandomSlices(32), rng = MersenneTwister(5))

  k = GaussianKernel(1.2)
  @test isapprox(
    mmd(X, Y, k; weights_x = wx, estimator = RandomFeatures(256), rng = MersenneTwister(6)),
    mmd(Xdup, Y, k; estimator = RandomFeatures(256), rng = MersenneTwister(6));
    atol = 1e-10,
  )

  # weighted sliced estimate agrees with the weighted exact value
  wr = rand(MersenneTwister(62), 200)
  exact = energydistance(X, Y; weights_x = wr)
  est = energydistance(X, Y; weights_x = wr, estimator = RandomSlices(512), rng = MersenneTwister(7))
  @test isapprox(est, exact; rtol = 0.2)

  @test_throws ArgumentError energydistance(X, Y; weights_x = wr, estimator = RandomFeatures(8))
  @test_throws ArgumentError mmd(X, Y, k; weights_x = wr, estimator = RandomSlices(8))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. test/test_estimators.jl` and `julia --project=. test/test_quality.jl`
Expected: `MethodError: no method matching _ed1d(::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64})`; `ArgumentError: RandomSlices is not defined for the energy distance with weights`

- [ ] **Step 3: Implement**

In `src/estimators.jl`, after `_ed1d(a, b)`:

```julia
# Σ_{i,k} w_i w_k |a_i − a_k| for a sorted sample `sorted` with weights `w`
# aligned to it: Σ_{i<k} w_i w_k (a_k − a_i) = Σ_k w_k (a_k W_{k−1} − A_{k−1})
# with W, A the prefix sums of w and w·a; doubled for ordered pairs.
function _weighted_within_abs(sorted::AbstractVector{<:Real}, w::AbstractVector{<:Real})
  W = 0.0
  A = 0.0
  s = 0.0
  @inbounds for k in eachindex(sorted, w)
    s += w[k] * (sorted[k] * W - A)
    W += w[k]
    A += w[k] * sorted[k]
  end
  return 2s
end

# Σ_{i,j} w_i v_j |a_i − b_j| with `a` sorted and `w` aligned to it, via
# prefix sums of w and w·a; `b` need not be sorted.
function _weighted_cross_abs(
  a::AbstractVector{<:Real},
  w::AbstractVector{<:Real},
  b::AbstractVector{<:Real},
  v::AbstractVector{<:Real},
)
  n = length(a)
  W = cumsum(w)
  A = cumsum(w .* a)
  Wn = W[n]
  An = A[n]
  total = 0.0
  @inbounds for j in eachindex(b, v)
    y = b[j]
    r = searchsortedlast(a, y)
    Wr = r == 0 ? 0.0 : W[r]
    Ar = r == 0 ? 0.0 : A[r]
    total += v[j] * ((y * Wr - Ar) + ((An - Ar) - y * (Wn - Wr)))
  end
  return total
end

# Weighted one-dimensional energy distance, weights scaled to sum one.
function _ed1d(
  a::AbstractVector{<:Real},
  w::AbstractVector{<:Real},
  b::AbstractVector{<:Real},
  v::AbstractVector{<:Real},
)
  pa = sortperm(a)
  pb = sortperm(b)
  sa = Float64.(a[pa])
  sb = Float64.(b[pb])
  wa = w[pa]
  vb = v[pb]
  return 2 * _weighted_cross_abs(sa, wa, sb, vb) - _weighted_within_abs(sa, wa) -
         _weighted_within_abs(sb, vb)
end
```

After `_sliced_energydistance(X, Y, k, rng)`:

```julia
function _sliced_energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64},
  k::Int,
  rng::AbstractRNG,
)
  p = size(X, 2)
  Θ = _project_directions(rng, p, k)
  total = 0.0
  for j = 1:k
    θ = view(Θ, :, j)
    total += _ed1d(X * θ, wx, Y * θ, wy)
  end
  return total / (k * sphere_constant(p))
end
```

After `_feature_mean(φ, X; block)`:

```julia
# Weighted feature mean Σᵢ wᵢ z(xᵢ), weights scaled to sum one, block-wise.
function _feature_mean(
  φ::FourierFeatureMap,
  X::AbstractMatrix,
  w::AbstractVector{Float64};
  block::Int = 4_096,
)
  D = length(φ.b)
  n = size(X, 1)
  acc = zeros(D)
  for i0 = 1:block:n
    i1 = min(i0 + block - 1, n)
    @views Z = cos.(X[i0:i1, :] * φ.W' .+ φ.b')      # (rows × D)
    @views acc .+= Z' * w[i0:i1]
  end
  return φ.scale .* acc
end
```

After `_rff_mmd(k, X, Y, D, rng)`:

```julia
function _rff_mmd(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64},
  D::Int,
  rng::AbstractRNG,
)
  φ = FourierFeatureMap(k, size(X, 2), D, rng)
  return sum(abs2, _feature_mean(φ, X, wx) .- _feature_mean(φ, Y, wy))
end
```

In `src/quality.jl`, immediately before the weighted `_undefined` fallbacks added in Task 3:

```julia
_energydistance(e::RandomSlices, X, Y, wx, wy, rng, n_threads) =
  _sliced_energydistance(X, Y, wx, wy, e.k, rng)
```

```julia
_mmd(e::RandomFeatures, k::GaussianKernel{Float64}, X, Y, wx, wy, rng, n_threads) =
  _rff_mmd(k, X, Y, wx, wy, e.D, rng)
```

Update the `RandomSlices` and `RandomFeatures` docstrings in `src/estimators.jl` with one added sentence each: "With sample weights the per-direction one-dimensional energy distance is weighted (prefix sums of the sorted weights)." and "With sample weights the feature means are weighted means."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. test/test_estimators.jl` and `julia --project=. test/test_quality.jl`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/estimators.jl src/quality.jl test/test_estimators.jl test/test_quality.jl
git commit -m "feat: Add weighted RandomSlices and RandomFeatures estimators

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Weighted MM support points (energy kernel) with both `kappa` rules

**Files:**

- Modify: `src/optimizer.jl` (`_mm_sweep!`, `support_points(::EnergyKernel, ...)`, `_objective_trajectory`)
- Test: `test/test_optimizer.jl`

**Interfaces:**

- Consumes: `_mean_one_weights`, `_check_weights`, `_normalize_weights`, `_uniform_weights` (Task 1); weighted `_exact_energydistance` (Task 3)
- Produces:
  - `_mm_sweep!(new_points, current_const, points, subsample_data, subsample_weights::AbstractVector{Float64}, running_const, alpha, bounds, n_threads)` (one extra positional argument, `ŵ` for the subsample rows, mean one)
  - `support_points(::EnergyKernel, data, n; weights = nothing, _subsampling = :uniform, ...)`
  - `_objective_trajectory(data, n; max_iterations, rng, weights = nothing)`
  - `_draw_subsample(rng, N, kappa, w_hat, ::Val{:uniform})` and `(…, ::Val{:proportional})` returning `(indices, subsample_weights)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_optimizer.jl` (top-level testset):

```julia
@testset "weighted support points (energy kernel)" begin
  @testset "nothing and uniform weights give identical points" begin
    data = randn(MersenneTwister(70), 150, 2)
    a, ca, ia = SPlit.support_points(EnergyKernel(), data, 15; max_iterations = 40, rng = MersenneTwister(1))
    b, cb, ib = SPlit.support_points(EnergyKernel(), data, 15; max_iterations = 40, rng = MersenneTwister(1), weights = ones(150))
    c, cc, ic = SPlit.support_points(EnergyKernel(), data, 15; max_iterations = 40, rng = MersenneTwister(1), weights = fill(0.37, 150))
    @test a == b == c
    @test (ca, ia) == (cb, ib) == (cc, ic)
    # stochastic mode too, both rules, for uniform weights
    d, _, _ = SPlit.support_points(EnergyKernel(), data, 15; kappa = 60, max_iterations = 30, rng = MersenneTwister(2))
    e, _, _ = SPlit.support_points(EnergyKernel(), data, 15; kappa = 60, max_iterations = 30, rng = MersenneTwister(2), weights = ones(150))
    @test d == e
  end

  @testset "one weighted sweep equals one sweep on duplicated rows" begin
    rng = MersenneTwister(71)
    data = randn(rng, 40, 2)
    counts = rand(rng, 1:3, 40)
    dup = vcat([data[i:i, :] for i = 1:40 for _ = 1:counts[i]]...)
    n = 6
    points = data[1:n, :] .+ 0.05
    bounds_w = SPlit._data_bounds(data)
    bounds_d = SPlit._data_bounds(dup)
    new_w = similar(points)
    new_d = similar(points)
    cw = zeros(n)
    cd = zeros(n)
    SPlit._mm_sweep!(new_w, cw, copy(points), data, SPlit._mean_one_weights(Float64.(counts)), zeros(n), 1.0, bounds_w, 1)
    SPlit._mm_sweep!(new_d, cd, copy(points), dup, ones(size(dup, 1)), zeros(n), 1.0, bounds_d, 1)
    @test isapprox(new_w, new_d; atol = 1e-10)
  end

  @testset "weighted full-data MM monotonically decreases the weighted objective" begin
    rng = MersenneTwister(72)
    data = randn(rng, 150, 2)
    w = rand(rng, 150) .^ 3
    traj = SPlit._objective_trajectory(data, 15; max_iterations = 40, rng = MersenneTwister(3), weights = w)
    @test length(traj) >= 2
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-8
    end
  end

  @testset "concentrated weights pull support points toward the heavy cluster" begin
    rng = MersenneTwister(73)
    A = randn(rng, 200, 2) .- 4.0
    B = randn(rng, 200, 2) .+ 4.0
    data = vcat(A, B)
    w = vcat(fill(9.0, 200), fill(1.0, 200))
    in_A(pts) = count(<(0.0), pts[:, 1])
    unweighted, _, _ = SPlit.support_points(EnergyKernel(), data, 40; max_iterations = 100, rng = MersenneTwister(4))
    weighted, _, _ = SPlit.support_points(EnergyKernel(), data, 40; max_iterations = 100, rng = MersenneTwister(4), weights = w)
    @test in_A(weighted) > in_A(unweighted)
    @test in_A(weighted) >= 30
    for rule in (:uniform, :proportional)
      stoch, _, _ = SPlit.support_points(EnergyKernel(), data, 40; kappa = 120, max_iterations = 100, rng = MersenneTwister(5), weights = w, _subsampling = rule)
      @test in_A(stoch) >= 28
    end
  end

  @testset "validation" begin
    data = randn(MersenneTwister(74), 50, 2)
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; weights = ones(49))
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; weights = ones(50), _subsampling = :other)
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_optimizer.jl`
Expected: `unsupported keyword argument "weights"` / `MethodError` on `_mm_sweep!` with 9 arguments

- [ ] **Step 3: Implement**

In `src/optimizer.jl`, change `using StatsBase: sample` to `using StatsBase: sample, Weights` and replace `_mm_sweep!` with:

```julia
# One MM sweep over all support points. Reads `points`, writes `new_points`
# and `current_const`; each m is independent, so chunks run in parallel.
# `subsample_weights` are ŵ (mean one) for the rows of `subsample_data`:
# with normalized weights w̄ the update is
#   ξ_m ← [ (1/n) Σ_{o≠m} (ξ_m − ξ_o)/‖ξ_m − ξ_o‖ + Σ_i w̄_i x_i/‖x_i − ξ_m‖ ]
#         / Σ_i w̄_i/‖x_i − ξ_m‖
# (Mak & Joseph 2018, Theorem 3, with the empirical measure replaced by
# Σ w̄_i δ(x_i); the majorizer is the same bound term by term). Multiplying
# numerator and denominator by n_sub gives the form below with ŵ = n_sub w̄
# and the (n_sub/n) factor on the repulsion term. Uniform weights make
# ŵ ≡ 1.0 exactly, so the arithmetic is the unweighted one bit for bit.
function _mm_sweep!(
  new_points::Matrix{Float64},
  current_const::Vector{Float64},
  points::Matrix{Float64},
  subsample_data::AbstractMatrix{Float64},
  subsample_weights::AbstractVector{Float64},
  running_const::Vector{Float64},
  alpha::Float64,
  bounds::Matrix{Float64},
  n_threads::Int,
)
  n, p = size(points)
  n_sub = size(subsample_data, 1)
  chunks = collect(Iterators.partition(1:n, cld(n, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for m in chunk
      xprime = zeros(p)
      for o = 1:n
        o == m && continue
        s = 0.0
        for j = 1:p
          s += (points[m, j] - points[o, j])^2
        end
        d = sqrt(s) + eps(Float64)
        for j = 1:p
          xprime[j] += (points[m, j] - points[o, j]) / d
        end
      end
      xprime .*= n_sub / n
      c = 0.0
      for i = 1:n_sub
        s = 0.0
        for j = 1:p
          s += (subsample_data[i, j] - points[m, j])^2
        end
        d = sqrt(s) + eps(Float64)
        wi = subsample_weights[i]
        c += wi / d
        for j = 1:p
          xprime[j] += wi * subsample_data[i, j] / d
        end
      end
      current_const[m] = c
      denom = (1 - alpha) * running_const[m] + alpha * c
      if denom > 0
        for j = 1:p
          xprime[j] =
            ((1 - alpha) * running_const[m] * points[m, j] + alpha * xprime[j]) / denom
        end
      else
        for j = 1:p
          xprime[j] = points[m, j]
        end
      end
      for j = 1:p
        new_points[m, j] = clamp(xprime[j], bounds[j, 1], bounds[j, 2])
      end
    end
  end
  return nothing
end

# Stochastic-mode subsample: row indices and their ŵ (mean one within the
# subsample). `:uniform` draws rows uniformly and rescales their weights;
# `:proportional` draws rows with probability ∝ w and treats them as uniform.
function _draw_subsample(rng::AbstractRNG, N::Int, kappa::Int, w_hat::Vector{Float64}, ::Val{:uniform})
  idx = sample(rng, 1:N, kappa; replace = false)
  return idx, _mean_one_weights(w_hat[idx])
end
function _draw_subsample(rng::AbstractRNG, N::Int, kappa::Int, w_hat::Vector{Float64}, ::Val{:proportional})
  idx = sample(rng, 1:N, Weights(w_hat), kappa; replace = false)
  return idx, ones(kappa)
end
```

Replace the `support_points(::EnergyKernel, ...)` function with:

```julia
function support_points(
  ::EnergyKernel,
  data::Matrix{Float64},
  n::Int;
  kappa::Union{Nothing,Int} = nothing,
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
  verbose::Bool = false,
  weights::Union{Nothing,AbstractVector} = nothing,
  _n0_factor::Float64 = 0.2,
  _subsampling::Symbol = :uniform,
)
  N = size(data, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  kappa === nothing ||
    kappa > 0 ||
    throw(ArgumentError("kappa must be positive, got $kappa"))
  max_iterations > 0 ||
    throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
  _subsampling in (:uniform, :proportional) ||
    throw(ArgumentError("_subsampling must be :uniform or :proportional, got :$_subsampling"))
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(_check_weights(weights, N))

  bounds = _data_bounds(data)
  working = copy(data)
  if length(unique(eachrow(working))) < N
    _jitter!(rng, working, bounds)
  end

  points = _initial_points(rng, working, n, bounds)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  stochastic = kappa !== nothing && kappa < N
  rule = Val(_subsampling)
  # Implementation constant (not from the papers): running-average weight
  # n0 = 0.2n, chosen by a small convergence experiment; see docstring.
  n0 = _n0_factor * n

  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    verbose && print("\rIteration $iteration/$max_iterations")

    if stochastic
      idx, sub_w = _draw_subsample(rng, N, kappa, w_hat, rule)
      sub = working[idx, :]
    else
      sub, sub_w = working, w_hat
    end
    alpha = stochastic ? n0 / (iteration + n0) : 1.0

    _mm_sweep!(
      new_points,
      current_const,
      points,
      sub,
      sub_w,
      running_const,
      alpha,
      bounds,
      n_threads,
    )

    max_move = 0.0
    @views for m = 1:n
      max_move = max(max_move, sum(abs2, new_points[m, :] .- points[m, :]))
    end
    points, new_points = new_points, points
    if stochastic
      running_const .= (1 - alpha) .* running_const .+ alpha .* current_const
    end
    converged = max_move < tolerance
  end
  verbose && println()

  return points, converged, iteration
end
```

Note: for the unweighted stochastic path the `:uniform` rule draws `idx` with the same `sample` call as before and returns `_mean_one_weights(ones[idx])`, which is exactly `ones(kappa)`, so points are unchanged.

Append to the `support_points` docstring (the shared one above the energy method):

```text
`weights` (one non-negative entry per row, `nothing` for uniform) makes the
points approximate the weighted empirical distribution `Σ w̄ᵢ δ(xᵢ)`: the
data sums in the MM update carry `ŵᵢ = N w̄ᵢ`, which is exactly `1.0` for
uniform weights. In stochastic mode `_subsampling` (internal) selects how
the `kappa` rows are drawn: `:uniform` draws them uniformly and rescales
their weights to mean one within the subsample; `:proportional` draws them
with probability proportional to the weights and treats the subsample as
uniform (this needs at least `kappa` rows with positive weight). The
default was chosen by the weighted-`kappa` experiment on the Design
experiments page.
```

Replace `_objective_trajectory` with:

```julia
# Test helper: energy objective E(points, data) after each full-data MM sweep,
# weighted when `weights` is given.
function _objective_trajectory(
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
  weights::Union{Nothing,AbstractVector} = nothing,
)
  N = size(data, 1)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(_check_weights(weights, N))
  w_bar = weights === nothing ? _uniform_weights(N) : _normalize_weights(weights, N)
  u = _uniform_weights(n)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  traj = Float64[_exact_energydistance(points, data, u, w_bar)]
  for _ = 1:max_iterations
    _mm_sweep!(new_points, current_const, points, data, w_hat, running_const, 1.0, bounds, 1)
    points, new_points = new_points, points
    push!(traj, _exact_energydistance(points, data, u, w_bar))
  end
  return traj
end
```

(The existing unweighted trajectory test still passes: with uniform weights the weighted exact energy distance equals the unweighted one up to rounding, and the test only checks monotonicity.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_optimizer.jl`
Expected: all pass, including the pre-existing `support_points` tests (the Gaussian tests in this file still call the unchanged Gaussian method).

- [ ] **Step 5: Commit**

```bash
git add src/optimizer.jl test/test_optimizer.jl
git commit -m "feat: Add sample weights to the energy-kernel MM optimizer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Weighted Gaussian-kernel support points

**Files:**

- Modify: `src/optimizer.jl` (`_mmd_objective`, `_mmd_gradient!`, `_armijo_step!`, `support_points(::GaussianKernel, ...)`, `_mmd_trajectory`)
- Test: `test/test_optimizer.jl`

**Interfaces:**

- Consumes: weighted `_mean_kernel` (Task 3), `_mean_one_weights`, `_normalize_weights`, `_uniform_weights` (Task 1)
- Produces:
  - `_mmd_objective(k, points, data, ::Nothing)` (= existing) and `_mmd_objective(k, points, data, w_bar::Vector{Float64})`
  - `_mmd_gradient!(G, k, points, data, w_hat::AbstractVector{Float64}, n_threads)` (one extra positional argument)
  - `_armijo_step!(new_points, points, G, f0, t0, k, data, bounds, w_bar)` (one extra positional argument, `nothing` or `w̄`)
  - `support_points(::GaussianKernel, data, n; weights = nothing, ...)`
  - `_mmd_trajectory(k, data, n; max_iterations, rng, weights = nothing)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_optimizer.jl`:

```julia
@testset "weighted support points (Gaussian kernel)" begin
  k = GaussianKernel(1.0)

  @testset "nothing and uniform weights give identical points" begin
    data = randn(MersenneTwister(80), 120, 2)
    a, ca, ia = SPlit.support_points(k, data, 12; max_iterations = 40, rng = MersenneTwister(1))
    b, cb, ib = SPlit.support_points(k, data, 12; max_iterations = 40, rng = MersenneTwister(1), weights = ones(120))
    @test a == b
    @test (ca, ia) == (cb, ib)
  end

  @testset "weighted gradient matches finite differences of the weighted objective" begin
    rng = MersenneTwister(81)
    data = randn(rng, 30, 2)
    w = rand(rng, 30)
    w_bar = w ./ sum(w)
    w_hat = w .* (30 / sum(w))
    points = randn(rng, 5, 2)
    G = similar(points)
    SPlit._mmd_gradient!(G, k, points, data, w_hat, 1)
    h = 1e-6
    for m = 1:5, j = 1:2
      plus = copy(points); plus[m, j] += h
      minus = copy(points); minus[m, j] -= h
      fd = (SPlit._mmd_objective(k, plus, data, w_bar) - SPlit._mmd_objective(k, minus, data, w_bar)) / (2h)
      @test isapprox(G[m, j], fd; atol = 1e-6)
    end
  end

  @testset "weighted objective never increases across accepted steps" begin
    rng = MersenneTwister(82)
    data = randn(rng, 120, 2)
    w = rand(rng, 120) .^ 2
    traj = SPlit._mmd_trajectory(k, data, 12; max_iterations = 40, rng = MersenneTwister(2), weights = w)
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-12
    end
  end

  @testset "concentrated weights pull support points toward the heavy cluster" begin
    rng = MersenneTwister(83)
    A = randn(rng, 200, 2) .- 4.0
    B = randn(rng, 200, 2) .+ 4.0
    data = vcat(A, B)
    w = vcat(fill(9.0, 200), fill(1.0, 200))
    in_A(pts) = count(<(0.0), pts[:, 1])
    unweighted, _, _ = SPlit.support_points(k, data, 40; max_iterations = 100, rng = MersenneTwister(4))
    weighted, _, _ = SPlit.support_points(k, data, 40; max_iterations = 100, rng = MersenneTwister(4), weights = w)
    @test in_A(weighted) > in_A(unweighted)
  end

  @testset "validation" begin
    data = randn(MersenneTwister(84), 50, 2)
    @test_throws ArgumentError SPlit.support_points(k, data, 5; weights = ones(49))
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_optimizer.jl`
Expected: `unsupported keyword argument "weights"` on the Gaussian method; `MethodError` on `_mmd_gradient!` with 6 arguments

- [ ] **Step 3: Implement**

In `src/optimizer.jl`:

Add after the existing `_mmd_objective`:

```julia
_mmd_objective(k::GaussianKernel{Float64}, points, data, ::Nothing) =
  _mmd_objective(k, points, data)

# Weighted MMD² objective up to the constant data self-term:
# mean k(ξ, ξ) − 2 Σ_l w̄_l mean_m k(ξ_m, x_l), with w̄ scaled to sum one.
function _mmd_objective(
  k::GaussianKernel{Float64},
  points::AbstractMatrix{Float64},
  data::AbstractMatrix{Float64},
  w_bar::AbstractVector{Float64},
)
  n = size(points, 1)
  return _mean_kernel(k, points, points) -
         2 * _mean_kernel(k, points, data, _uniform_weights(n), w_bar)
end
```

Replace `_mmd_gradient!` with (the only change is the `w_hat` argument and the `w_hat[l] *` factor):

```julia
# Full gradient of _mmd_objective with respect to every support point.
# Row m of G is (2/n²) Σ_{j≠m} ∇k(ξ_m, ξ_j) − (2/n) Σ_l w̄_l ∇k(ξ_m, x_l);
# with ŵ = N w̄ (mean one, exactly 1.0 for uniform weights) the data term is
# (2/(nN)) Σ_l ŵ_l ∇k(ξ_m, x_l). Chunks write disjoint rows of G; `points`
# and `data` are read-only.
function _mmd_gradient!(
  G::Matrix{Float64},
  k::GaussianKernel{Float64},
  points::Matrix{Float64},
  data::Matrix{Float64},
  w_hat::AbstractVector{Float64},
  n_threads::Int,
)
  n, p = size(points)
  N = size(data, 1)
  chunks = collect(Iterators.partition(1:n, cld(n, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn begin
      g = zeros(p)
      for m in chunk
        @views ξ = points[m, :]
        for j = 1:p
          G[m, j] = 0.0
        end
        for o = 1:n
          o == m && continue
          @views kernelgrad!(g, k, ξ, points[o, :])
          for j = 1:p
            G[m, j] += (2 / n^2) * g[j]
          end
        end
        for l = 1:N
          @views kernelgrad!(g, k, ξ, data[l, :])
          wl = w_hat[l]
          for j = 1:p
            G[m, j] -= (2 / (n * N)) * wl * g[j]
          end
        end
      end
    end
  end
  return G
end
```

Note: `(2 / (n * N)) * wl * g[j]` evaluates left to right as `((2/(nN)) * wl) * g[j]`; with `wl == 1.0` this is exactly `(2/(nN)) * g[j]`, the previous expression.

In `_armijo_step!`, add a trailing positional parameter `w_bar::Union{Nothing,AbstractVector{Float64}}` after `bounds`, and change the objective call to `f_new = _mmd_objective(k, new_points, data, w_bar)`.

In `support_points(k::GaussianKernel, ...)`:

- add keyword `weights::Union{Nothing,AbstractVector} = nothing` after `verbose`;
- after the argument checks add

```julia
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(_check_weights(weights, N))
  w_bar = weights === nothing ? nothing : _normalize_weights(weights, N)
```

- change `f = _mmd_objective(k, points, working)` to `f = _mmd_objective(k, points, working, w_bar)`;
- change `_mmd_gradient!(G, k, points, working, n_threads)` to `_mmd_gradient!(G, k, points, working, w_hat, n_threads)`;
- change `_armijo_step!(new_points, points, G, f, t0, k, working, bounds)` to `_armijo_step!(new_points, points, G, f, t0, k, working, bounds, w_bar)`.

Append to the Gaussian `support_points` docstring:

```text
`weights` (one non-negative entry per row, `nothing` for uniform) makes the
points minimize the MMD² to the weighted empirical distribution
`Σ w̄ᵢ δ(xᵢ)`: the data term of the objective and of the gradient carries
`w̄`, with `ŵ = N w̄` (exactly `1.0` for uniform weights) inside the
gradient loop, so unweighted results are unchanged.
```

Replace `_mmd_trajectory` with:

```julia
# Test helper: objective after each accepted step (full-data Gaussian path),
# weighted when `weights` is given. Mirrors the scale-aware first step of
# `support_points(::GaussianKernel, …)`.
function _mmd_trajectory(
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
  weights::Union{Nothing,AbstractVector} = nothing,
)
  N = size(data, 1)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(_check_weights(weights, N))
  w_bar = weights === nothing ? nothing : _normalize_weights(weights, N)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  G = similar(points)
  f = _mmd_objective(k, points, data, w_bar)
  traj = Float64[f]
  t = 1.0
  for iteration = 1:max_iterations
    _mmd_gradient!(G, k, points, data, w_hat, 1)
    t0 = iteration == 1 ? _first_step(G, bounds) : 2t
    t, f = _armijo_step!(new_points, points, G, f, t0, k, data, bounds, w_bar)
    t == 0.0 && break
    points, new_points = new_points, points
    push!(traj, f)
  end
  return traj
end
```

Search the test suite for other callers of `_mmd_gradient!`, `_armijo_step!`, `_mmd_trajectory` (`grep -n "_mmd_gradient!\|_armijo_step!\|_mmd_trajectory" test/*.jl`) and add the new argument (`ones(N)` / `nothing`) where they are called directly.

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_optimizer.jl`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/optimizer.jl test/test_optimizer.jl
git commit -m "feat: Add sample weights to the Gaussian-kernel support-point optimizer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Weighted kernel herding

**Files:**

- Modify: `src/herding.jl`
- Test: `test/test_herding.jl`

**Interfaces:**

- Consumes: `_normalize_weights` (Task 1), `preprocess(data, weights)`, `resolve(kernel, X, rng, weights)` (Task 2)
- Produces:
  - `_data_term(kernel, X, w_bar::Vector{Float64}, n_threads) -> Vector{Float64}` (`dᵢ = Σₗ w̄ₗ k(xᵢ, xₗ)`)
  - `herd(kernel, X, n; weights = nothing, n_threads)`
  - `datasplit(s::HerdingSplitter, data; weights = nothing)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_herding.jl`:

```julia
@testset "weighted herding" begin
  k = GaussianKernel(1.0)

  @testset "uniform weights reproduce the unweighted selection exactly" begin
    X = randn(MersenneTwister(90), 120, 2)
    @test SPlit.herd(k, X, 20; weights = ones(120)) == SPlit.herd(k, X, 20)
    @test SPlit.herd(EnergyKernel(), X, 20; weights = fill(2.5, 120)) == SPlit.herd(EnergyKernel(), X, 20)
  end

  @testset "weighted data term equals the data term on duplicated rows" begin
    X = randn(MersenneTwister(91), 25, 2)
    Xdup = vcat(X[1:1, :], X)
    d_w = SPlit._data_term(k, X, vcat([2.0], ones(24)) ./ 26, 1)
    d_dup = SPlit._data_term(k, Xdup, 1)
    @test isapprox(d_w, d_dup[2:end]; atol = 1e-12)
  end

  @testset "concentrated weights pull selections toward the heavy cluster" begin
    rng = MersenneTwister(92)
    A = randn(rng, 150, 2) .- 4.0
    B = randn(rng, 150, 2) .+ 4.0
    X = vcat(A, B)
    w = vcat(fill(9.0, 150), fill(1.0, 150))
    for kernel in (k, EnergyKernel())
      unweighted = SPlit.herd(kernel, X, 30)
      weighted = SPlit.herd(kernel, X, 30; weights = w)
      @test count(<=(150), weighted) > count(<=(150), unweighted)
    end
  end

  @testset "datasplit forwards weights" begin
    data = randn(MersenneTwister(93), 100, 2)
    s = HerdingSplitter(kernel = k)
    @test datasplit(s, data; weights = ones(100)).test_indices == datasplit(s, data).test_indices
    @test_throws ArgumentError datasplit(s, data; weights = ones(99))
  end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_herding.jl`
Expected: `unsupported keyword argument "weights"`

- [ ] **Step 3: Implement**

In `src/herding.jl`, after the unweighted `_data_term`:

```julia
# Weighted data term d_i = Σ_l w̄_l k(x_i, x_l), w̄ scaled to sum one.
function _data_term(
  kernel::SplitKernel,
  X::Matrix{Float64},
  w_bar::AbstractVector{Float64},
  n_threads::Int,
)
  N = size(X, 1)
  Xt = permutedims(X)
  d = Vector{Float64}(undef, N)
  chunks = collect(Iterators.partition(1:N, cld(N, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for i in chunk
      s = 0.0
      @views xi = Xt[:, i]
      @views for l = 1:N
        s += w_bar[l] * kernelvalue(kernel, xi, Xt[:, l])
      end
      d[i] = s
    end
  end
  return d
end
```

Change `herd`'s signature to `herd(kernel, X, n; weights::Union{Nothing,AbstractVector} = nothing, n_threads::Int = Threads.nthreads())` and its data-term line to:

```julia
  d = weights === nothing ? _data_term(kernel, X, n_threads) :
      _data_term(kernel, X, _normalize_weights(weights, N), n_threads)
```

Append to the `herd` docstring: "`weights` (one non-negative entry per row, `nothing` for uniform) replaces the data term by `Σₗ w̄ₗ k(x, xₗ)` with `w̄` scaled to sum one, so the selection targets the weighted empirical distribution; the selected-set term is unchanged."

Change `datasplit(s::HerdingSplitter, data)` to:

```julia
function datasplit(s::HerdingSplitter, data; weights::Union{Nothing,AbstractVector} = nothing)
  X = preprocess(data, weights)
  n_total = size(X, 1)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))
  kernel = resolve(s.kernel, X, s.rng, weights)
  fitted = HerdingSplitter(kernel, s.ratio, s.n_threads, s.rng)
  small = herd(kernel, X, n_small; weights, n_threads = s.n_threads)
  rest = setdiff(1:n_total, small)
  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  return SplitResult(collect(train), collect(test), true, n_small, fitted)
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_herding.jl`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/herding.jl test/test_herding.jl
git commit -m "feat: Add sample weights to kernel herding

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `weights` on `datasplit`, `splitquality`, and `compare`

**Files:**

- Modify: `src/splitter.jl`, `src/quality.jl` (`splitquality`), `src/comparison.jl`
- Test: `test/test_splitter.jl`, `test/test_comparison.jl`, `test/test_properties.jl`

**Interfaces:**

- Consumes: everything above
- Produces:
  - `datasplit(s::SupportPointSplitter, data; weights = nothing)`
  - `splitquality(data, result; weights = nothing, kwargs...)`
  - `compare(methods, data; weights = nothing, kwargs...)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_splitter.jl`:

```julia
@testset "datasplit with weights" begin
  data = randn(MersenneTwister(100), 200, 3)

  @testset "uniform weights reproduce the unweighted split exactly" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      s = SupportPointSplitter(kernel = kernel, max_iterations = 60, rng = MersenneTwister(1))
      r0 = datasplit(s, data)
      s = SupportPointSplitter(kernel = kernel, max_iterations = 60, rng = MersenneTwister(1))
      r1 = datasplit(s, data; weights = ones(200))
      # weighted standardization matches the unweighted one up to rounding,
      # so the selected rows agree; the optimizer trajectory is compared
      # bit for bit at the `support_points` level in test_optimizer.jl
      @test r1.test_indices == r0.test_indices
      @test r1.train_indices == r0.train_indices
    end
  end

  @testset "a :median bandwidth is resolved with the weights" begin
    s = SupportPointSplitter(kernel = GaussianKernel(), max_iterations = 5, rng = MersenneTwister(2))
    r = datasplit(s, data; weights = rand(MersenneTwister(3), 200))
    @test r.method.kernel isa GaussianKernel{Float64}
  end

  @testset "heavy cluster gets more test rows" begin
    rng = MersenneTwister(101)
    A = randn(rng, 200, 2) .- 4.0
    B = randn(rng, 200, 2) .+ 4.0
    X = vcat(A, B)
    w = vcat(fill(9.0, 200), fill(1.0, 200))
    s = SupportPointSplitter(ratio = 0.2, max_iterations = 100, rng = MersenneTwister(4))
    r_u = datasplit(s, X)
    s = SupportPointSplitter(ratio = 0.2, max_iterations = 100, rng = MersenneTwister(4))
    r_w = datasplit(s, X; weights = w)
    @test count(<=(200), r_w.test_indices) > count(<=(200), r_u.test_indices)
  end

  @testset "DataFrame input with weights" begin
    df = DataFrame(x = randn(MersenneTwister(102), 90), g = repeat(["a", "b", "c"], 30))
    s = SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(5))
    r = datasplit(s, df; weights = rand(MersenneTwister(6), 90))
    @test length(test_indices(r)) == 18
  end

  @testset "validation" begin
    s = SupportPointSplitter(max_iterations = 5)
    @test_throws ArgumentError datasplit(s, data; weights = ones(199))
    @test_throws ArgumentError datasplit(s, data; weights = -ones(200))
  end
end

@testset "splitquality with weights" begin
  data = randn(MersenneTwister(103), 150, 2)
  s = SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(7))
  r = datasplit(s, data)
  @test isapprox(splitquality(data, r; weights = ones(150)), splitquality(data, r); rtol = 1e-8)
  w = rand(MersenneTwister(8), 150)
  q = splitquality(data, r; weights = w)
  @test q isa Float64
  @test q >= -1e-12
  # equals the weighted discrepancy between the weighted train and test rows
  X = SPlit.preprocess(data, w)
  wn = w ./ sum(w)
  expected = energydistance(
    X[r.train_indices, :],
    X[r.test_indices, :];
    weights_x = wn[r.train_indices],
    weights_y = wn[r.test_indices],
  )
  @test isapprox(q, expected; atol = 1e-12)
  @test_throws ArgumentError splitquality(data, r; weights = ones(10))
end
```

Append to `test/test_comparison.jl`:

```julia
@testset "compare forwards weights" begin
  data = randn(MersenneTwister(110), 120, 2)
  w = rand(MersenneTwister(111), 120)
  methods = [
    SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(1)),
    HerdingSplitter(kernel = GaussianKernel(1.0)),
  ]
  c = compare(methods, data; weights = w)
  @test length(c.qualities) == 2
  @test all(isfinite, c.qualities)
  @test isapprox(c.qualities[1], splitquality(data, c.results[1]; weights = w); atol = 1e-12)
end
```

Append to `test/test_properties.jl` inside the top-level testset:

```julia
  @testset "weighted support-point splits beat random splits under the weighted energy distance" begin
    rng = MersenneTwister(120)
    data = randn(rng, 300, 3)
    w = exp.(randn(rng, 300))        # log-normal, heavy-tailed weights
    s = SupportPointSplitter(max_iterations = 200, rng = MersenneTwister(121))
    r = datasplit(s, data; weights = w)
    q_sp = splitquality(data, r; weights = w)
    n_test = length(test_indices(r))
    random_qs = map(1:25) do i
      perm = randperm(MersenneTwister(5_000 + i), 300)
      fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
      splitquality(data, fake; weights = w)
    end
    @test q_sp < mean(random_qs)
  end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. test/test_splitter.jl`
Expected: `unsupported keyword argument "weights"` on `datasplit(::SupportPointSplitter, ...)`

- [ ] **Step 3: Implement**

In `src/splitter.jl`, change `datasplit(s::SupportPointSplitter, data)` to take `; weights::Union{Nothing,AbstractVector} = nothing`, replace `X = preprocess(data)` with `X = preprocess(data, weights)`, `kernel = resolve(s.kernel, X, s.rng)` with `kernel = resolve(s.kernel, X, s.rng, weights)`, and add `weights,` to the `support_points(...)` keyword list. Append to the `datasplit` docstring:

```text
`weights` (one non-negative entry per row; `nothing` for uniform) makes the
split target the weighted empirical distribution `Σ w̄ᵢ δ(xᵢ)`: the smaller
subset is chosen to approximate it, preprocessing standardizes with the
weighted mean and variance, and a `:median` bandwidth is resolved from rows
drawn in proportion to the weights. The train/test labeling rule is
unchanged. Weights proportional to duplication counts are equivalent to
duplicating rows.
```

In `src/quality.jl`, change `splitquality` to:

```julia
function splitquality(
  data,
  result::SplitResult;
  kernel::SplitKernel = EnergyKernel(),
  estimator::Union{Nothing,DiscrepancyEstimator} = nothing,
  exact_threshold::Int = 20_000,
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  rng::AbstractRNG = Random.default_rng(),
  n_threads::Int = Threads.nthreads(),
  weights::Union{Nothing,AbstractVector} = nothing,
)
  X = preprocess(data, weights)
  train = X[result.train_indices, :]
  test = X[result.test_indices, :]
  k = isresolved(kernel) ? kernel : resolve(kernel, X, rng, weights)
  chosen = if subsample !== nothing
    Subsample(subsample, repeats)
  elseif estimator !== nothing
    estimator
  elseif size(train, 1) + size(test, 1) <= exact_threshold
    Exact()
  else
    _fallback_estimator(k)
  end
  weights === nothing && return mmd(train, test, k; estimator = chosen, rng, n_threads)
  w = _normalize_weights(weights, size(X, 1))
  return mmd(
    train,
    test,
    k;
    estimator = chosen,
    rng,
    n_threads,
    weights_x = w[result.train_indices],
    weights_y = w[result.test_indices],
  )
end
```

Append to its docstring:

```text
`weights` (one non-negative entry per row of `data`) applies the weighted
preprocessing `datasplit` used and compares the weighted train rows with
the weighted test rows, each side's weights rescaled to sum one.
```

In `src/comparison.jl`, change `compare` to:

```julia
function compare(
  methods::Vector{<:AbstractSplitter},
  data;
  kernel::SplitKernel = EnergyKernel(),
  rng::AbstractRNG = Random.default_rng(),
  weights::Union{Nothing,AbstractVector} = nothing,
  kwargs...,
)
  results = [datasplit(m, data; weights) for m in methods]
  k = isresolved(kernel) ? kernel : resolve(kernel, preprocess(data, weights), rng, weights)
  qualities = [splitquality(data, r; kernel = k, rng, weights, kwargs...) for r in results]
  return SplitComparison([r.method for r in results], results, qualities, k)
end
```

and add to its docstring: "`weights` is forwarded to both `datasplit` and `splitquality`."

- [ ] **Step 4: Run the full suite**

Run: `julia --project=. -e "using Pkg; Pkg.test()"`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/splitter.jl src/quality.jl src/comparison.jl test/test_splitter.jl test/test_comparison.jl test/test_properties.jl
git commit -m "feat: Accept sample weights in datasplit, splitquality, and compare

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Weighted `kappa` benchmark and the default rule

**Files:**

- Create: `benchmark/weighted_kappa.jl`
- Create (generated): `docs/src/assets/benchmarks/weighted_kappa.md`
- Modify: `docs/src/25-design-experiments.md` (new section at the end), `src/optimizer.jl` (default of `_subsampling` if `:proportional` wins)

**Interfaces:**

- Consumes: `support_points(::EnergyKernel, ...; weights, kappa, _subsampling)`, `select_nearest`, `preprocess(data, weights)`, `energydistance(...; weights_y)`

- [ ] **Step 1: Write the benchmark script**

`benchmark/weighted_kappa.jl`:

```julia
# Weighted-kappa rule experiment (roadmap M1 open question): with sample
# weights, should the stochastic MM draw its `kappa` rows uniformly and
# rescale their weights (`:uniform`), or draw them in proportion to the
# weights and treat the subsample as uniform (`:proportional`)? For each
# rule, dataset, weight profile, kappa, and rng seed, run the energy-kernel
# support points, round them to rows with `select_nearest`, and score the
# selected rows (uniform) against the full data under the weights with the
# weighted energy distance; record wall time too. Writes
# `docs/src/assets/benchmarks/weighted_kappa.md` and prints the decision.
# Run: `julia -t auto --project=benchmark benchmark/weighted_kappa.jl`.

using SPlit, DataFrames, Random, Statistics

include(joinpath(@__DIR__, "datasets.jl"))

const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const N = 10_000
const RATIO = 0.2
const KAPPAS = (500, 2_000)
const SEEDS = 1:5
const RULES = (:uniform, :proportional)

# Weight profiles: log-normal (heavy-tailed quality scores) and a 10:1
# two-cluster profile keyed on the sign of the first coordinate.
profiles(data, rng) = [
  ("lognormal", exp.(randn(rng, size(data, 1)))),
  ("cluster-10:1", [x > 0 ? 10.0 : 1.0 for x in view(data, :, 1)]),
]

rows = DataFrame(
  dataset = String[],
  profile = String[],
  kappa = Int[],
  rule = String[],
  mean_discrepancy = Float64[],
  se_discrepancy = Float64[],
  mean_seconds = Float64[],
)

for (dname, data) in datasets(N, MersenneTwister(2026))
  dname in ("normal-10d", "uniform-5d") || continue
  for (pname, w) in profiles(data, MersenneTwister(99))
    X = SPlit.preprocess(data, w)
    n_small = round(Int, RATIO * N)
    for kappa in KAPPAS, rule in RULES
      ds = Float64[]
      ts = Float64[]
      for seed in SEEDS
        t = @elapsed begin
          pts, _, _ = SPlit.support_points(
            EnergyKernel(),
            X,
            n_small;
            kappa,
            weights = w,
            rng = MersenneTwister(seed),
            _subsampling = rule,
          )
          small = SPlit.select_nearest(X, pts)
        end
        push!(ds, energydistance(X[small, :], X; weights_y = w))
        push!(ts, t)
      end
      push!(
        rows,
        (dname, pname, kappa, string(rule), mean(ds), std(ds) / sqrt(length(ds)), mean(ts)),
      )
    end
  end
end

open(joinpath(OUT, "weighted_kappa.md"), "w") do io
  println(io, "| dataset | profile | kappa | rule | weighted ED (mean ± se, 5 seeds) | mean seconds |")
  println(io, "|---|---|---:|---|---:|---:|")
  for r in eachrow(rows)
    println(
      io,
      "| $(r.dataset) | $(r.profile) | $(r.kappa) | `$(r.rule)` | ",
      "$(round(r.mean_discrepancy; sigdigits = 3)) ± $(round(r.se_discrepancy; sigdigits = 2)) | ",
      "$(round(r.mean_seconds; sigdigits = 3)) |",
    )
  end
end

# Decision: mean discrepancy over all datasets and profiles at kappa = 500.
at500 = rows[rows.kappa .== 500, :]
score(rule) = mean(at500[at500.rule .== rule, :mean_discrepancy])
se(rule) = sqrt(sum(abs2, at500[at500.rule .== rule, :se_discrepancy])) / count(==(rule), at500.rule)
for rule in RULES
  println("$(rule): mean weighted ED at kappa = 500 = $(score(string(rule))) (se $(se(string(rule))))")
end
gap = score("uniform") - score("proportional")
if gap > se("uniform") + se("proportional")
  println("decision: :proportional (lower by $(gap), beyond one standard error)")
else
  println("decision: :uniform (difference $(gap) within one standard error, simpler rule wins)")
end
```

- [ ] **Step 2: Run it**

Run: `julia -t auto --project=benchmark benchmark/weighted_kappa.jl` (from the worktree root; if `benchmark/Manifest.toml` is missing, first run `julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`).
Expected: prints two score lines and a `decision:` line; writes `docs/src/assets/benchmarks/weighted_kappa.md`. Copy the printed numbers into the next step.

- [ ] **Step 3: Apply the decision**

If the decision is `:proportional`, change the default in `src/optimizer.jl` from `_subsampling::Symbol = :uniform` to `_subsampling::Symbol = :proportional`, then rerun `julia --project=. test/test_optimizer.jl` (the "nothing and uniform weights give identical points" stochastic assertion compares two runs that both use the default rule, so it still holds). If the decision is `:uniform`, leave the code as is.

- [ ] **Step 4: Record it on the Design experiments page**

Append to `docs/src/25-design-experiments.md`:

````markdown
## [Weighted `kappa` subsampling](@id weighted-kappa)

With sample weights, the stochastic MM can draw its `kappa` rows in two
ways: uniformly, rescaling the drawn weights to mean one within the
subsample (`:uniform`), or in proportion to the weights, treating the
subsample as uniform (`:proportional`). Both are implemented behind the
internal `_subsampling` keyword of `support_points`; the default is
`:<winner>`. Measured on `normal-10d` and `uniform-5d` at N = 10,000 with
log-normal weights and a 10:1 two-cluster profile, `kappa` ∈ {500, 2000},
five rng seeds each; the score is the weighted energy distance between the
selected rows and the full data under the weights. At `kappa` = 500 the
mean score was <uniform mean> for `:uniform` and <proportional mean> for
`:proportional`; <one sentence: which won and by how much, or that they
were within one standard error so the simpler rule was kept>. Full table:
[`assets/benchmarks/weighted_kappa.md`](assets/benchmarks/weighted_kappa.md).
Reproduce with:

```sh
julia -t auto --project=benchmark benchmark/weighted_kappa.jl
```

````

Replace every `<...>` placeholder with the numbers printed in Step 2 (this is the only step of the plan where the values come from a run, not from the plan).

- [ ] **Step 5: Commit**

```bash
git add benchmark/weighted_kappa.jl docs/src/assets/benchmarks/weighted_kappa.md docs/src/25-design-experiments.md src/optimizer.jl
git commit -m "bench: Choose the weighted kappa subsampling rule by experiment

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Julia docs and AGENTS.md

**Files:**

- Modify: `docs/src/10-methods.md` (new section before "## Estimators" or at the end of the page, whichever keeps the existing order intact; append at the end), `docs/src/85-roadmap.md`, `AGENTS.md`

- [ ] **Step 1: Methods page**

Append to `docs/src/10-methods.md`:

````markdown
## [Weighted samples](@id weighted-samples)

Every quantity above is defined for the empirical distribution of the
data, in which each row has mass ``1/N``. Passing `weights` (one
non-negative entry per row, positive sum) replaces it by the weighted
empirical distribution ``P_w = \sum_i \bar w_i\, \delta_{x_i}`` with
``\bar w_i = w_i / \sum_l w_l``; the selected subset itself stays
uniformly weighted. Nothing else changes: the procedure runs the same five
steps, and `weights = nothing` (the default) is the unweighted case.

Which steps read the weights:

1. **Preprocess.** `preprocess(data, weights)` standardizes with the
   weighted mean ``\mu_j = \sum_i \bar w_i x_{ij}`` and the unbiased
   weighted variance
   ``\sigma_j^2 = \sum_i \bar w_i (x_{ij} - \mu_j)^2 / (1 - \sum_i \bar w_i^2)``,
   which is the usual ``n - 1`` denominator for uniform weights.
3. **Resolve the kernel.** A `:median` bandwidth is the median pairwise
   distance over rows drawn in proportion to the weights.
4. **Choose ``n`` rows.** The data terms carry the weights:
   - The energy distance between the support points and ``P_w`` is
     ``\frac{2}{n} \sum_{m,i} \bar w_i \|\xi_m - x_i\| - \frac{1}{n^2}\sum_{m,o} \|\xi_m - \xi_o\| - \sum_{i,k} \bar w_i \bar w_k \|x_i - x_k\|``,
     and MMD² is the same expression with kernel values in place of
     negated distances. The MM update of Mak & Joseph (2018) and the
     projected gradient keep their form with ``\bar w_i`` multiplying every
     data term, so the MM step is still monotone.
   - Kernel herding's data term becomes ``\sum_l \bar w_l\, k(x, x_l)``.

`splitquality(data, result; weights)` compares the weighted train rows with
the weighted test rows, each side's weights rescaled to sum one, and
`energydistance`/`mmd` accept `weights_x` and `weights_y` for the two
samples; every [`DiscrepancyEstimator`](@ref) has a weighted form
(`Subsample` rescales the weights of the rows it draws, `RandomSlices` uses
the weighted one-dimensional energy distance on each projection, and
`RandomFeatures` uses weighted feature means). Weights proportional to
duplication counts are equivalent to duplicating rows, which is what the
tests check. How stochastic `kappa` subsampling combines with weights was
decided by the experiment on the
[Design experiments](@ref weighted-kappa) page.
````

- [ ] **Step 2: Roadmap page**

In `docs/src/85-roadmap.md`:

- In the Current state table change `| Weighted samples | not supported | |` to `| Weighted samples | done |`weights` on `datasplit`,`splitquality`,`compare`;`weights_x`/`weights_y` on `energydistance` and `mmd`; see [Methods](@ref weighted-samples). |`.
- Change the M1 heading paragraph's first word from `Planned.` to `Done (2026-09-03).` and replace the bullet "Decide and document how `kappa` subsampling interacts with weights (weight-proportional vs. uniform sampling): open question below." with "How `kappa` subsampling interacts with weights was decided by experiment: see [Design experiments](@ref weighted-kappa)."
- In Open questions, replace the "Weighted `kappa` subsampling (M1)" bullet with: "Weighted `kappa` subsampling (M1). Resolved 2026-09-03 by the experiment on the [Design experiments](@ref weighted-kappa) page."
- Append to the Changelog: `- 2026-09-03: M1 (weighted samples) done; kappa question resolved.`

- [ ] **Step 3: AGENTS.md**

Under `## Gotchas` add:

````markdown
- `weights` (on `datasplit`, `splitquality`, `compare`) and
  `weights_x`/`weights_y` (on `energydistance`, `mmd`) define weighted
  empirical distributions; the selected subset is always uniform.
  `nothing` dispatches to the unweighted methods, which must stay
  bit-identical; weighted behavior lives in separate methods (or, in the
  hot loops, in a `ŵ` factor that is exactly `1.0` for uniform weights).
  Tests use "weights as duplication counts equals duplicated rows".
````

Under `## Workflow` add:

````markdown
- Julia/Python parity: every capability exposed in SPlit.jl must be exposed
  in `splitiq/` in the same change, with tests under `splitiq/tests` and a
  docs mention; a Julia-only feature PR is incomplete.
````

- [ ] **Step 4: Build the docs**

Run: `julia --project=docs docs/make.jl 2>&1 | grep -iE "error|warning: (invalid|unresolved|missing|no doc)"`
Expected: no output other than the pre-existing repository-URL warning. Also `ls docs/build/10-methods/index.html` exists.

- [ ] **Step 5: Commit**

```bash
git add docs/src/10-methods.md docs/src/85-roadmap.md AGENTS.md
git commit -m "docs: Describe weighted samples and record the Julia/Python parity rule

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: splitiq parity

**Files:**

- Modify: `splitiq/src/splitiq/_convert.py` (add `to_weights`), `splitiq/src/splitiq/split.py`, `splitiq/src/splitiq/quality.py`
- Create: `splitiq/tests/test_weights.py`
- Modify: `splitiq/docs/getting-started.md` (append a section), `docs/src/30-python.md` (append a section)

**Interfaces:**

- Consumes: Julia keywords from Tasks 3 and 8
- Produces:
  - `to_weights(weights) -> np.ndarray | None` (1-D contiguous `float64`; `None` passes through)
  - `datasplit(..., weights=None)`, `splitquality(..., weights=None)`, `energydistance(x, y, *, weights_x=None, weights_y=None, ...)`, `mmd(x, y, kernel, *, weights_x=None, weights_y=None, ...)`

- [ ] **Step 0: Prepare the dev Julia project**

From `splitiq/`: `make julia-dev` (builds `.julia_dev/` that develops the Julia package from this worktree) then `make setup`. Run tests with `make test`.

- [ ] **Step 1: Write the failing tests**

`splitiq/tests/test_weights.py`:

```python
"""Parity tests for sample weights in splitiq."""

from __future__ import annotations

import numpy as np
import pytest

from splitiq import datasplit, energydistance, mmd, splitquality


def _clusters(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((150, 2)) - 4.0
    b = rng.standard_normal((150, 2)) + 4.0
    weights = np.concatenate([np.full(150, 9.0), np.full(150, 1.0)])
    return np.vstack([a, b]), weights


def test_uniform_weights_reproduce_the_unweighted_split() -> None:
    data = np.random.default_rng(1).standard_normal((120, 2))
    plain = datasplit(data, ratio=0.2, seed=3, max_iterations=40)
    weighted = datasplit(data, ratio=0.2, seed=3, max_iterations=40, weights=np.ones(120))
    np.testing.assert_array_equal(plain.test_indices, weighted.test_indices)


def test_heavy_cluster_gets_more_test_rows() -> None:
    data, weights = _clusters()
    plain = datasplit(data, ratio=0.2, seed=4, max_iterations=100)
    weighted = datasplit(data, ratio=0.2, seed=4, max_iterations=100, weights=weights)
    assert np.sum(weighted.test_indices < 150) > np.sum(plain.test_indices < 150)


def test_herding_accepts_weights() -> None:
    data, weights = _clusters(seed=2)
    plain = datasplit(data, ratio=0.2, method='herding', kernel='gaussian', bandwidth=1.0)
    weighted = datasplit(
        data, ratio=0.2, method='herding', kernel='gaussian', bandwidth=1.0, weights=weights
    )
    assert np.sum(weighted.test_indices < 150) > np.sum(plain.test_indices < 150)


def test_energydistance_duplication_invariance() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((30, 2))
    y = rng.standard_normal((25, 2)) + 0.5
    x_dup = np.vstack([x[:1], x])
    weights_x = np.concatenate([[2.0], np.ones(29)])
    assert energydistance(x, y, weights_x=weights_x) == pytest.approx(
        energydistance(x_dup, y), abs=1e-10
    )
    assert mmd(x, y, bandwidth=0.8, weights_x=weights_x) == pytest.approx(
        mmd(x_dup, y, bandwidth=0.8), abs=1e-10
    )


def test_splitquality_accepts_weights() -> None:
    data, weights = _clusters(seed=6)
    result = datasplit(data, ratio=0.2, seed=7, max_iterations=40, weights=weights)
    assert splitquality(data, result, weights=weights) >= -1e-12


def test_weights_accept_lists_and_series_like_inputs() -> None:
    data = np.random.default_rng(8).standard_normal((40, 2))
    result = datasplit(data, ratio=0.25, seed=1, max_iterations=10, weights=[1.0] * 40)
    assert len(result.test_indices) == 10


@pytest.mark.parametrize(
    'bad',
    [np.ones(39), -np.ones(40), np.zeros(40), np.ones((40, 1))],
    ids=['wrong-length', 'negative', 'all-zero', 'two-dimensional'],
)
def test_bad_weights_raise_value_error(bad: np.ndarray) -> None:
    data = np.random.default_rng(9).standard_normal((40, 2))
    with pytest.raises(ValueError):
        datasplit(data, ratio=0.25, seed=1, max_iterations=10, weights=bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `splitiq/`): `make test`
Expected: `TypeError: datasplit() got an unexpected keyword argument 'weights'`

- [ ] **Step 3: Implement**

In `splitiq/src/splitiq/_convert.py`, after `to_matrix`:

```python
def to_weights(weights: DataLike | None) -> np.ndarray | None:
    """Convert a sample-weights argument to a 1-D ``float64`` vector.

    Args:
        weights: A 1-D array-like with one entry per row, or ``None``.

    Returns:
        A contiguous ``numpy.ndarray`` of ``float64``, or ``None`` when
        `weights` is ``None`` (callers omit the keyword in that case).
        Validation of the values themselves (length, sign, finiteness)
        happens in Julia and surfaces as ``ValueError``.

    Raises:
        ValueError: If `weights` is not one-dimensional.
    """
    if weights is None:
        return None
    array = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    if array.ndim != 1:
        msg = f'weights must be 1-D, got {array.ndim}-D'
        raise ValueError(msg)
    return array
```

In `splitiq/src/splitiq/split.py`:

- import `to_weights` from `splitiq._convert`;
- add `weights: DataLike | None = None,` to `datasplit`'s keyword-only parameters (after `seed`);
- add to its docstring Args: `weights: One non-negative entry per row, or ``None`` for uniform weights. Makes the split target the weighted empirical distribution of the rows; the selected subset itself is uniform. Weights proportional to duplication counts are equivalent to duplicating rows.` and to Raises: `or if`weights`has the wrong length, a negative or non-finite entry, or sums to zero.`;
- build `julia_weights = to_weights(weights)` next to `julia_data`, and pass it to both `jl.datasplit(...)` calls as `**_weights_kwarg(julia_weights)` where

```python
def _weights_kwarg(weights: np.ndarray | None) -> dict[str, np.ndarray]:
    """Keyword arguments carrying `weights`, empty when it is ``None``.

    Args:
        weights: A converted weights vector, or ``None``.

    Returns:
        ``{'weights': weights}`` or ``{}``.
    """
    return {} if weights is None else {'weights': weights}
```

In `splitiq/src/splitiq/quality.py`:

- import `to_weights` and add the same `_weights_kwarg` helper (or import it from `split.py`; prefer moving `_weights_kwarg` to `_convert.py` and importing it in both modules);
- `energydistance` and `mmd` gain keyword-only `weights_x: DataLike | None = None, weights_y: DataLike | None = None` with docstring Args `weights_x: One non-negative entry per row of`x`, or ``None`` for uniform.` / `weights_y: Same for`y`.`; after `kwargs = _estimator_kwargs(...)` add

```python
    if weights_x is not None:
        kwargs['weights_x'] = to_weights(weights_x)
    if weights_y is not None:
        kwargs['weights_y'] = to_weights(weights_y)
```

- `splitquality` gains `weights: DataLike | None = None` with docstring Arg `weights: One non-negative entry per row of`data`, or ``None``; compares the weighted train rows with the weighted test rows.` and passes `**_weights_kwarg(to_weights(weights))` into `jl.splitquality(...)`.

- [ ] **Step 4: Run tests, lint, types**

From `splitiq/`: `make test`, `make lint`, `make format`, `make typecheck`
Expected: tests pass; ruff and ty report nothing.

- [ ] **Step 5: Docs**

Append to `splitiq/docs/getting-started.md`:

````markdown
## Weighted samples

Pass `weights` (one non-negative entry per row) to make the split target
the weighted distribution of the rows, for example a quality score per
sample:

```python
import numpy as np
from splitiq import datasplit, splitquality

data = np.random.default_rng(0).standard_normal((1000, 8))
weights = np.exp(np.random.default_rng(1).standard_normal(1000))

result = datasplit(data, ratio=0.2, seed=42, weights=weights)
print(splitquality(data, result, weights=weights))
```

`energydistance` and `mmd` take `weights_x` and `weights_y` for their two
samples. Weights proportional to duplication counts are equivalent to
duplicating rows.

````

Append to `docs/src/30-python.md` a short section "## Weighted samples" with the same example, and one sentence pointing at the Julia keywords (`weights`, `weights_x`, `weights_y`) and the [Methods](@ref weighted-samples) section.

- [ ] **Step 6: Commit**

```bash
git add splitiq/src/splitiq/_convert.py splitiq/src/splitiq/split.py splitiq/src/splitiq/quality.py splitiq/tests/test_weights.py splitiq/docs/getting-started.md docs/src/30-python.md
git commit -m "feat(splitiq): Expose sample weights in datasplit, splitquality, energydistance, and mmd

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Quality gate

**Files:** none new

- [ ] **Step 1: Julia suite, threaded and single-threaded**

Run: `julia -t 4 --project=. -e "using Pkg; Pkg.test()"` and `julia -t 1 --project=. -e "using Pkg; Pkg.test()"`
Expected: all pass both ways.

- [ ] **Step 2: Docs build**

Run: `julia --project=docs docs/make.jl 2>&1 | grep -iE "error|warning: (invalid|unresolved|missing|no doc)"`
Expected: nothing beyond the repository-URL warning.

- [ ] **Step 3: splitiq**

From `splitiq/`: `make test && make lint && make typecheck && make docs`
Expected: all green.

- [ ] **Step 4: Formatting under the CI Julia version**

Run: `pre-commit run --all-files`, and if `julia +1.12` is available also `julia +1.12 -e 'using JuliaFormatter; format(".")'` followed by `git status --short` (CI lint uses Julia 1.12's formatter; see the repo memory note). Commit any formatter changes as `chore: Apply formatter`.

- [ ] **Step 5: Final review before the PR**

Run `git diff origin/main...HEAD --stat` and confirm: no existing test was modified (only appended), no existing docs section was rewritten, every new keyword defaults to `nothing`. Then report to the user; the PR is opened only after explicit confirmation.
