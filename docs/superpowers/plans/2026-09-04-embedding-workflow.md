# Embedding Workflow, Compress++, and Data-Selection Guide (M5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `standardize = false` to the public API, Compress++ to `KernelThinningSplitter` (`compress = :auto | :always | :never`), an example that selects LLM training data from a real embedding matrix, and the docs page that tells users which method to use.

**Architecture:** `standardize` is a keyword threaded from the five public functions into `_prepare` (a raw-matrix branch beside the existing preprocessing branch). Compress++ lives in `src/kernel_thinning.jl` as `_compress` (recursive four-way split + symmetrized kernel-thinning halving) and `_compress_plus_plus` (Compress then THIN by `kernel_thinning`), selected inside `kernel_thinning` by the `compress` keyword. The example is a standalone script with its own project under `examples/`; its table is committed and quoted by the new docs page.

**Tech Stack:** Julia 1.10+, DuckDB.jl (example only, reads the parquet's list column), Documenter.jl, Python 3.13 + juliacall (splitiq), pytest, uv, pre-commit.

**Spec:** `docs/superpowers/specs/2026-09-04-embedding-workflow-design.md`

## Global Constraints

- Every new keyword defaults to today's behavior (`standardize = true`, `compress = :auto` on the splitter but `:never` inside `kernel_thinning`); with the defaults every existing result is bit-identical. `:auto` never triggers at split ratios (see the spec), so `datasplit` with `KernelThinningSplitter()` is unchanged.
- `standardize = false`: numeric `AbstractMatrix`/`AbstractVector` only; `X = Matrix{Float64}(data)` (vector → `N × 1`); no centering, scaling, or column removal; a `DataFrame` raises `ArgumentError("standardize = false needs a numeric matrix or vector; encode DataFrames yourself or keep standardize = true")`; a reference must have the same number of columns.
- Compress: base case `length(seq) ≤ 4^g` returns `seq`; four consecutive parts of sizes `⌊ℓ/4⌋`/`⌈ℓ/4⌉`; HALVE = `kernel_thinning` of the block's own rows to `ℓ ÷ 2` with `compress = :never`, then with one `rand(rng)` keep the selected half or its complement (trimmed to `ℓ ÷ 2` by dropping its last row when `ℓ` is odd); rows returned in the block's original order.
- Compress++: `seq = randperm(rng, N)`; `g = max(4, ⌈log₂(2n/√N)⌉)`; `δ_halve = delta / (2K)` with `K` the number of HALVE calls (`delta / 2` when `K = 0`); THIN = `kernel_thinning(kernel, X[S_C, :], n; delta = delta / 2, compress = :never)`; if `|S_C| ≤ n`, increment `g` and rerun Compress; `iterations` = THIN's swaps.
- `:auto` uses Compress++ iff `weights === nothing && target === nothing && _compress_pays_off(N, n)` with `_compress_pays_off(N, n) = 4.0^g * (4 * log(4, N) + 1) < 1.5 * N`; `:always` with `weights`/`target` raises `ArgumentError("Compress++ is defined for the data's own distribution; pass compress = :never with weights or a reference")`; `:never` is today's path.
- All randomness through the caller's `rng`; nothing in `src/` seeds or prints. Never cite or compare with other implementations. Docstrings sit directly above what they document. Existing tests are append-only.
- Every Julia capability lands in splitiq in this branch (tests + docs mention).
- Test one file with `julia --project=<worktree> <worktree>/test/<file>.jl`; the suite with `Pkg.test()`; new files also on `julia +1.12`. Commit messages `<type>: <Capitalized description>` + trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; pre-commit runs on commit, never bypass it.
- Work only in `/home/appleparan/src/SPlit.jl/.claude/worktrees/feat-embedding` (branch `feat/embedding-workflow`). Use absolute paths in shell commands.

---

## File structure

| File | Responsibility |
|---|---|
| `src/splitter.jl` | `_raw_matrix`, `_prepare_raw`, `standardize` keyword on `_prepare`, `_select`, `selectrows`, `datasplit` |
| `src/multiplet.jl` | forward `standardize` |
| `src/quality.jl` | `splitquality(...; standardize)` |
| `src/comparison.jl` | `compare(...; standardize)` |
| `src/kernel_thinning.jl` | `_COMPRESS_G_MIN`, `_compress_g`, `_compress_pays_off`, `_four_parts`, `_compress_halvings`, `_symmetrized_halve`, `_compress`, `_compress_plus_plus`, `compress` keyword on `kernel_thinning`, `compress` field on `KernelThinningSplitter` |
| `test/test_standardize.jl` (new), `test/test_kernel_thinning.jl` (append), `test/test_properties.jl` (append), `test/runtests.jl` | tests |
| `examples/Project.toml`, `examples/llm_data_selection.jl`, `examples/README.md` (new); `docs/src/assets/examples/llm_selection.md` (output) | example |
| `docs/src/40-llm-data-selection.md` (new), `10-methods.md`, `85-roadmap.md`, `index.md`, `30-python.md`, `README.md`, `AGENTS.md`, `.gitignore` (`examples/data/`) | docs |
| `splitiq/src/splitiq/split.py`, `multiplet.py`, `quality.py`, `splitiq/tests/test_standardize.py` (new), `test_kernel_thinning.py`, `splitiq/docs/*`, `splitiq/README.md` | parity |

---

### Task 1: `standardize = false`

**Files:**

- Modify: `src/splitter.jl`, `src/multiplet.jl`, `src/quality.jl`, `src/comparison.jl`
- Create: `test/test_standardize.jl`; modify `test/runtests.jl`

**Interfaces:**

- Produces: `_raw_matrix(data) -> Matrix{Float64}`; `_prepare(s, data, weights, reference, reference_weights; standardize::Bool = true)`; `_select(...; standardize)`; public `standardize::Bool = true` on `selectrows`, `datasplit`, `multiplet`, `splitquality`, `compare`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_standardize.jl`:

```julia
using Test
using SPlit
using Random
using Statistics
using DataFrames
using LinearAlgebra

@testset "standardize = false" begin
  E = let M = randn(MersenneTwister(1), 300, 8)
    M ./ norm.(eachrow(M))            # cosine-normalized rows
  end
  E[:, 8] .= 0.25                      # a constant column: kept when standardize = false

  @testset "raw rows reach the splitter unchanged" begin
    rows = selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 60; standardize = false)
    direct = SPlit.herd(EnergyKernel(), Matrix{Float64}(E), 60)
    @test rows == direct
    @test rows != selectrows(HerdingSplitter(kernel = EnergyKernel()), E[:, 1:7], 60)  # standardized path differs
    r = datasplit(TwinningSplitter(), E; standardize = false)
    @test sort(vcat(train_indices(r), test_indices(r))) == 1:300
    v = randn(MersenneTwister(2), 100)
    @test selectrows(HerdingSplitter(kernel = EnergyKernel()), v, 20; standardize = false) ==
          SPlit.herd(EnergyKernel(), reshape(Matrix{Float64}(reshape(v, :, 1)), :, 1), 20)
  end

  @testset "weights and a reference on raw rows" begin
    w = rand(MersenneTwister(3), 300)
    @test selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 60; weights = w, standardize = false) ==
          SPlit.herd(EnergyKernel(), Matrix{Float64}(E), 60; weights = w)
    R = E[1:40, :]
    @test selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 60; reference = R, standardize = false) ==
          SPlit.herd(EnergyKernel(), Matrix{Float64}(E), 60; target = Matrix{Float64}(R))
    @test_throws ArgumentError selectrows(HerdingSplitter(), E, 60; reference = E[1:40, 1:7], standardize = false)
    @test_throws ArgumentError selectrows(HerdingSplitter(), E, 60; weights = w, reference = R, standardize = false)
    @test_throws ArgumentError selectrows(HerdingSplitter(), E, 60; reference_weights = rand(40), standardize = false)
  end

  @testset "DataFrames are rejected; the default is unchanged" begin
    df = DataFrame(x = randn(MersenneTwister(4), 50), g = repeat(["a", "b"], 25))
    @test_throws ArgumentError datasplit(HerdingSplitter(), df; standardize = false)
    @test_throws ArgumentError selectrows(HerdingSplitter(), E, 10; reference = df[1:10, [:x]], standardize = false)
    data = randn(MersenneTwister(5), 200, 3)
    @test test_indices(datasplit(HerdingSplitter(kernel = EnergyKernel()), data)) ==
          test_indices(datasplit(HerdingSplitter(kernel = EnergyKernel()), data; standardize = true))
  end

  @testset "splitquality, compare, and multiplet on raw rows" begin
    r = datasplit(HerdingSplitter(kernel = EnergyKernel()), E; standardize = false)
    q = splitquality(E, r; standardize = false)
    @test q ≈ energydistance(E[train_indices(r), :], E[test_indices(r), :])
    @test q != splitquality(E[:, 1:7], r)
    qr = splitquality(E, r; reference = E[1:40, :], standardize = false)
    @test qr ≈ energydistance(E[test_indices(r), :], E[1:40, :])
    c = compare([HerdingSplitter(kernel = EnergyKernel()), TwinningSplitter()], E; standardize = false)
    @test c.qualities[1] ≈ q
    folds = multiplet(TwinningSplitter(), E, 4; standardize = false)
    @test sort(reduce(vcat, folds)) == 1:300
    @test_throws ArgumentError multiplet(TwinningSplitter(), DataFrame(x = randn(40)), 4; standardize = false)
  end
end
```

Add `include("test_standardize.jl")` to `test/runtests.jl` after `include("test_kernel_thinning.jl")`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=<worktree> <worktree>/test/test_standardize.jl`
Expected: `MethodError` / `UndefKeywordError` for `standardize`.

- [ ] **Step 3: Implement**

`src/splitter.jl`, above `_prepare`:

```julia
# Raw-matrix input for `standardize = false`: no centering, scaling, or
# column removal; a vector is one column. Anything else (a DataFrame) is
# rejected, since categorical encoding needs the preprocessing path.
_raw_matrix(data::AbstractMatrix{<:Real}) = Matrix{Float64}(data)
_raw_matrix(data::AbstractVector{<:Real}) = Matrix{Float64}(reshape(data, :, 1))
_raw_matrix(::Any) = throw(
  ArgumentError(
    "standardize = false needs a numeric matrix or vector; encode DataFrames yourself or keep standardize = true",
  ),
)

# `_prepare` without preprocessing: the rows are used as they are.
function _prepare_raw(s::AbstractSplitter, data, weights, reference, reference_weights)
  X = _raw_matrix(data)
  if reference === nothing
    reference_weights === nothing ||
      throw(ArgumentError("reference_weights needs a reference"))
    weights === nothing || _check_weights(weights, size(X, 1))
    return X, resolve(s.kernel, X, s.rng, weights), nothing, nothing
  end
  weights === nothing || throw(
    ArgumentError(
      "with a reference, weight the reference (reference_weights), not the data",
    ),
  )
  R = _raw_matrix(reference)
  size(R, 1) >= 1 || throw(ArgumentError("reference must have at least one row"))
  size(R, 2) == size(X, 2) ||
    throw(ArgumentError("reference must have the same number of columns as data"))
  reference_weights === nothing || _check_weights(reference_weights, size(R, 1))
  return X, resolve(s.kernel, R, s.rng, reference_weights), R, reference_weights
end
```

Change `_prepare`'s signature to `function _prepare(s::AbstractSplitter, data, weights, reference, reference_weights; standardize::Bool = true)` and make its first line `standardize || return _prepare_raw(s, data, weights, reference, reference_weights)`. Thread `standardize::Bool = true` through `_select` (pass to `_prepare`), `selectrows`, and `datasplit` (pass to `_select`), and document it in both docstrings with one sentence: "`standardize = false` uses a numeric matrix or vector as it is (no centering, scaling, or constant-column removal) — for cosine-normalized embeddings; a `DataFrame` then raises an `ArgumentError`." Update the `_prepare` comment.

`src/multiplet.jl`: `multiplet` gains `standardize::Bool = true`, passed to `_multiplet_sequential`/`_multiplet_halving` (which pass it to every `selectrows`) and to `_multiplet_single` (which passes it to `_prepare`); one docstring sentence.

`src/quality.jl`: `splitquality` gains `standardize::Bool = true`; in the reference branch replace the `prep`/`R`/`Xs` lines by

```julia
    R, Xall = if standardize
      prep = fit_preprocessor(reference; weights = reference_weights, extra = data)
      apply_preprocessor(prep, reference), apply_preprocessor(prep, data)
    else
      Rr = _raw_matrix(reference)
      Xr = _raw_matrix(data)
      size(Rr, 2) == size(Xr, 2) ||
        throw(ArgumentError("reference must have the same number of columns as data"))
      Rr, Xr
    end
    Xs = Xall[_selected_indices(result), :]
```

and in the plain branch `X = standardize ? preprocess(data, weights) : (weights === nothing || _check_weights(weights, _nrows(data)); _raw_matrix(data))`. One docstring sentence.

`src/comparison.jl`: `compare` gains `standardize::Bool = true`, forwarded to `datasplit` and `splitquality`; the `:median` scoring-kernel resolution uses `_raw_matrix(reference)` / `_raw_matrix(data)` when `standardize` is false (same structure as today's `if`). One docstring sentence.

- [ ] **Step 4: Run the tests**

Run: `julia --project=<worktree> <worktree>/test/test_standardize.jl`, then the full suite.
Expected: all pass; existing tests unchanged in count.

- [ ] **Step 5: Commit**

```bash
git add src/splitter.jl src/multiplet.jl src/quality.jl src/comparison.jl test/test_standardize.jl test/runtests.jl
git commit -m "feat: Add standardize = false to use numeric rows as they are

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Compress core

**Files:**

- Modify: `src/kernel_thinning.jl` (append before the splitter section), `test/test_kernel_thinning.jl` (append)

**Interfaces:**

- Consumes: `kernel_thinning(kernel, X, n; delta, compress = :never, n_threads, rng)` — note Task 3 adds the `compress` keyword; in this task `kernel_thinning` has no `compress` keyword yet, so the helpers below call it without `compress` and Task 3 adds the argument.
- Produces: `_COMPRESS_G_MIN = 4`; `_compress_g(N, n)`; `_compress_pays_off(N, n)`; `_four_parts(seq)`; `_compress_halvings(ℓ, g)`; `_symmetrized_halve(kernel, X, S, δ, rng; n_threads)`; `_compress(kernel, X, seq, g, δ_halve, rng; n_threads)`; `_compress_plus_plus(kernel, X, n; delta, rng, n_threads) -> (rows, swaps)`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_kernel_thinning.jl`:

```julia
@testset "Compress and Compress++" begin
  @testset "g and the cost rule" begin
    @test SPlit._compress_g(10_000, 500) == 4
    @test SPlit._compress_g(10_000, 2_000) == 6
    @test SPlit._compress_g(1_000_000, 10_000) == 5
    @test SPlit._compress_pays_off(10_000, 500)
    @test !SPlit._compress_pays_off(10_000, 2_000)
    @test !SPlit._compress_pays_off(1_000, 50)
    @test !SPlit._compress_pays_off(10_000, 2_000) && !SPlit._compress_pays_off(100_000, 20_000)  # split ratios never
  end

  @testset "four parts and the halving count" begin
    @test SPlit._four_parts(collect(1:10)) == [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10]]
    @test SPlit._compress_halvings(256, 4) == 0
    @test SPlit._compress_halvings(1024, 4) == 1
    @test SPlit._compress_halvings(4096, 4) == 5
  end

  @testset "symmetrized halving returns half of the block in its order" begin
    X = SPlit.preprocess(randn(MersenneTwister(80), 400, 2))
    S = randperm(MersenneTwister(81), 400)[1:201]
    outs = [SPlit._symmetrized_halve(EnergyKernel(), X, S, 0.1, MersenneTwister(s); n_threads = 2) for s = 1:8]
    @test all(o -> length(o) == 100 && allunique(o) && all(in(S), o), outs)
    @test all(o -> issorted(indexin(o, S)), outs)                    # block order preserved
    @test length(unique(outs)) > 1                                    # both halves occur across seeds
  end

  @testset "Compress returns about 2^g √N rows of the input, deterministically" begin
    X = SPlit.preprocess(randn(MersenneTwister(82), 4096, 3))
    seq = randperm(MersenneTwister(83), 4096)
    S = SPlit._compress(EnergyKernel(), X, seq, 4, 1e-3, MersenneTwister(84); n_threads = 2)
    @test allunique(S) && all(in(seq), S)
    @test 512 <= length(S) <= 2048                                    # 2^4 √4096 = 1024
    @test S == SPlit._compress(EnergyKernel(), X, seq, 4, 1e-3, MersenneTwister(84); n_threads = 1)
    @test SPlit._compress(EnergyKernel(), X, seq[1:200], 4, 1e-3, MersenneTwister(0)) == seq[1:200]   # base case
  end

  @testset "Compress++ selects n distinct rows and beats random" begin
    mixture = let rng = MersenneTwister(85), N = 8_000
      c = rand(rng, 1:4, N)
      centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
      SPlit.preprocess(centers[c, :] .+ randn(rng, N, 2))
    end
    rows, swaps = SPlit._compress_plus_plus(EnergyKernel(), mixture, 200; delta = 0.5, rng = MersenneTwister(86), n_threads = 2)
    @test length(rows) == 200 && allunique(rows) && swaps >= 0
    q = energydistance(mixture[rows, :], mixture)
    random_q = mean(energydistance(mixture[randperm(MersenneTwister(300 + i), 8_000)[1:200], :], mixture) for i = 1:10)
    @test q < random_q
    @test rows == SPlit._compress_plus_plus(EnergyKernel(), mixture, 200; delta = 0.5, rng = MersenneTwister(86), n_threads = 1)[1]
  end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: `UndefVarError: _compress_g`.

- [ ] **Step 3: Implement**

Insert into `src/kernel_thinning.jl` after `kernel_thinning` and before the `KernelThinningSplitter` docstring:

```julia
# ---- Compress++ (Shetty, Dwivedi & Mackey 2022) -----------------------------

# The paper's experiments use g = 4 throughout; `_compress_g` raises it so
# the compressed set has about 2n rows or more.
const _COMPRESS_G_MIN = 4

_compress_g(N::Int, n::Int) = max(_COMPRESS_G_MIN, ceil(Int, log2(2n / sqrt(N))))

# Estimated kernel evaluations: plain kernel thinning ≈ 1.5N² (halvings,
# data term, swap pass) against Compress++ ≈ 4^g N (4 log₄ N + 1) (paper
# Remark 1 with a quadratic HALVE, plus THIN on 2^g √N rows).
_compress_pays_off(N::Int, n::Int) = 4.0^_compress_g(N, n) * (4 * log(4, N) + 1) < 1.5 * N

# Four consecutive parts of sizes ⌊ℓ/4⌋ or ⌈ℓ/4⌉ (the paper's "arbitrary
# subsequences"; the input is already a random permutation).
function _four_parts(seq::Vector{Int})
  ℓ = length(seq)
  bounds = round.(Int, (0:4) .* (ℓ / 4))
  return [seq[(bounds[i]+1):bounds[i+1]] for i = 1:4]
end

# Number of HALVE calls Compress makes on an input of length ℓ.
function _compress_halvings(ℓ::Int, g::Int)
  ℓ <= 4^g && return 0
  bounds = round.(Int, (0:4) .* (ℓ / 4))
  return 1 + sum(_compress_halvings(bounds[i+1] - bounds[i], g) for i = 1:4)
end

# HALVE (paper Ex. 2 and Remark 3): kernel thinning of the block's own rows
# to ⌊ℓ/2⌋, then the selected half or its complement with equal
# probability, so each halving is unbiased. Rows keep the block's order.
function _symmetrized_halve(
  kernel::SplitKernel,
  X::Matrix{Float64},
  S::Vector{Int},
  δ::Float64,
  rng::AbstractRNG;
  n_threads::Int = Threads.nthreads(),
)
  ℓ = length(S)
  ℓ >= 2 || return S
  half = ℓ ÷ 2
  local_rows, _ = kernel_thinning(kernel, X[S, :], half; delta = δ, n_threads, rng)
  keep = if rand(rng) < 0.5
    local_rows
  else
    other = setdiff(1:ℓ, local_rows)
    length(other) > half ? other[1:half] : other
  end
  return S[sort(keep)]
end

"""
    _compress(kernel, X, seq, g, δ_halve, rng; n_threads) -> Vector{Int}

Compress (Shetty, Dwivedi & Mackey 2022, Alg. 1) of the row sequence `seq`
with oversampling `g`: sequences of at most `4^g` rows are returned as they
are; longer ones are split into four consecutive parts, compressed
recursively, concatenated, and halved by `_symmetrized_halve`. Returns about
`2^g √(length(seq))` rows.
"""
function _compress(
  kernel::SplitKernel,
  X::Matrix{Float64},
  seq::Vector{Int},
  g::Int,
  δ_halve::Float64,
  rng::AbstractRNG;
  n_threads::Int = Threads.nthreads(),
)
  length(seq) <= 4^g && return seq
  merged = reduce(vcat, (_compress(kernel, X, part, g, δ_halve, rng; n_threads) for part in _four_parts(seq)))
  return _symmetrized_halve(kernel, X, merged, δ_halve, rng; n_threads)
end

"""
    _compress_plus_plus(kernel, X, n; delta, rng, n_threads) -> (rows, swaps)

Compress++ (Shetty, Dwivedi & Mackey 2022, Alg. 2): Compress a random
permutation of the rows with `g = max(4, ⌈log₂(2n/√N)⌉)`, then THIN the
compressed set to `n` rows with [`kernel_thinning`](@ref) (whose data term
and swap candidates are the compressed rows, as in the paper). `delta` is
split evenly between the halvings (each gets `delta / 2K`, `K` the number
of HALVE calls) and THIN (`delta / 2`), a union bound in place of the
paper's per-call schedule. If a compressed set has no more than `n` rows,
`g` is increased by one and Compress is rerun.
"""
function _compress_plus_plus(
  kernel::SplitKernel,
  X::Matrix{Float64},
  n::Int;
  delta::Float64,
  rng::AbstractRNG,
  n_threads::Int = Threads.nthreads(),
)
  N = size(X, 1)
  g = _compress_g(N, n)
  seq = randperm(rng, N)
  while true
    K = _compress_halvings(N, g)
    δ_halve = K == 0 ? delta / 2 : delta / (2K)
    S_C = _compress(kernel, X, seq, g, δ_halve, rng; n_threads)
    if length(S_C) > n
      local_rows, swaps = kernel_thinning(kernel, X[S_C, :], n; delta = delta / 2, n_threads, rng)
      return S_C[local_rows], swaps
    end
    g += 1
  end
end
```

(In Task 3 the two inner `kernel_thinning` calls gain `compress = :never`.)

- [ ] **Step 4: Run the tests**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: all pass. The "beats random" check is a property from the paper; report numbers rather than loosening if it fails.

- [ ] **Step 5: Commit**

```bash
git add src/kernel_thinning.jl test/test_kernel_thinning.jl
git commit -m "feat: Add Compress and Compress++ over symmetrized kernel-thinning halving

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `compress` on `kernel_thinning` and `KernelThinningSplitter`

**Files:**

- Modify: `src/kernel_thinning.jl`; `test/test_kernel_thinning.jl` (append); `test/test_properties.jl` (append)

**Interfaces:**

- Produces: `kernel_thinning(...; compress::Symbol = :never)`; `KernelThinningSplitter(; ..., compress::Symbol = :auto)` with field `compress`; `_with_kernel` carries it; `show` prints it.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_kernel_thinning.jl`:

```julia
@testset "compress keyword" begin
  mixture = let rng = MersenneTwister(90), N = 8_000
    c = rand(rng, 1:4, N)
    centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
    centers[c, :] .+ randn(rng, N, 2)
  end
  X = SPlit.preprocess(mixture)

  @testset "kernel_thinning: :auto follows the cost rule, :always and :never are explicit" begin
    a = SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :auto, rng = MersenneTwister(1))
    @test a == SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :always, rng = MersenneTwister(1))
    @test a != SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :never, rng = MersenneTwister(1))
    b = SPlit.kernel_thinning(EnergyKernel(), X, 1_600; compress = :auto, rng = MersenneTwister(2))
    @test b == SPlit.kernel_thinning(EnergyKernel(), X, 1_600; compress = :never, rng = MersenneTwister(2))
    @test b == SPlit.kernel_thinning(EnergyKernel(), X, 1_600; rng = MersenneTwister(2))   # default :never
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :sometimes)
    w = rand(MersenneTwister(3), 8_000)
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :always, weights = w)
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :always, target = X[1:100, :])
    @test SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :auto, weights = w, rng = MersenneTwister(4)) ==
          SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :never, weights = w, rng = MersenneTwister(4))
    # the complement rule composes with compress
    hi, _ = SPlit.kernel_thinning(EnergyKernel(), X, 7_800; compress = :always, rng = MersenneTwister(5))
    lo, _ = SPlit.kernel_thinning(EnergyKernel(), X, 200; compress = :always, rng = MersenneTwister(5))
    @test hi == sort(setdiff(1:8_000, lo))
  end

  @testset "KernelThinningSplitter: field, show, selectrows, datasplit unchanged" begin
    s = KernelThinningSplitter()
    @test s.compress === :auto
    @test KernelThinningSplitter(compress = :never).compress === :never
    @test_throws ArgumentError KernelThinningSplitter(compress = :maybe)
    @test occursin("compress=:auto", sprint(show, s))
    sel = selectrows(KernelThinningSplitter(rng = MersenneTwister(6)), mixture, 200)
    @test sel == selectrows(KernelThinningSplitter(compress = :always, rng = MersenneTwister(6)), mixture, 200)
    @test sel != selectrows(KernelThinningSplitter(compress = :never, rng = MersenneTwister(6)), mixture, 200)
    small = mixture[1:600, :]
    @test test_indices(datasplit(KernelThinningSplitter(rng = MersenneTwister(7)), small)) ==
          test_indices(datasplit(KernelThinningSplitter(compress = :never, rng = MersenneTwister(7)), small))
    @test_throws ArgumentError selectrows(KernelThinningSplitter(compress = :always), mixture, 200; weights = rand(8_000))
    folds = multiplet(KernelThinningSplitter(rng = MersenneTwister(8)), small, 3)
    @test sort(reduce(vcat, folds)) == 1:600
  end
end
```

Append to `test/test_properties.jl` inside the outer testset:

```julia
  @testset "Compress++ selections beat random subsets at n ≪ N" begin
    data = let rng = MersenneTwister(500), N = 8_000
      c = rand(rng, 1:4, N)
      centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
      centers[c, :] .+ randn(rng, N, 2)
    end
    X = SPlit.preprocess(data)
    rows = selectrows(KernelThinningSplitter(compress = :always, rng = MersenneTwister(501)), data, 200)
    q = energydistance(X[rows, :], X)
    random_q = mean(energydistance(X[randperm(MersenneTwister(5_000 + i), 8_000)[1:200], :], X) for i = 1:20)
    @test q < random_q
  end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: `MethodError` on the `compress` keyword.

- [ ] **Step 3: Implement**

In `kernel_thinning`: add `compress::Symbol = :never` to the signature; validate `compress in (:auto, :always, :never) || throw(ArgumentError("compress must be :auto, :always, or :never, got :$compress"))`; keep the `0 < n < N` and `delta` checks; the complement branch passes `compress` through; then, before the plain path:

```julia
  if compress === :always && (weights !== nothing || target !== nothing)
    throw(
      ArgumentError(
        "Compress++ is defined for the data's own distribution; pass compress = :never with weights or a reference",
      ),
    )
  end
  use_compress =
    compress === :always ||
    (compress === :auto && weights === nothing && target === nothing && _compress_pays_off(N, n))
  use_compress && return _compress_plus_plus(kernel, X, n; delta = Float64(delta), rng, n_threads)
```

Add `compress = :never` to the two inner `kernel_thinning` calls in `_symmetrized_halve` and `_compress_plus_plus`. Extend the docstring: the `compress` keyword, the `g`/cost rule, "Compress++ is used only for the data's own measure", and one sentence under "Differences from the paper" (HALVE = kernel thinning of the block; `δ` split evenly; uneven four-way splits; `g` tied to `n`).

`KernelThinningSplitter`: add field `compress::Symbol` (after `delta`), keyword `compress::Symbol = :auto` with the same validation, `_with_kernel` carries it, `_select_rows` passes `compress = s.compress`, `show` prints `compress=:auto`. Docstring bullet: "`compress`: `:auto` (default) runs Compress++ when `n ≪ N` makes it cheaper than plain kernel thinning and the target is the data itself (never at split ratios, so `datasplit` is unaffected); `:always`/`:never` force it."

- [ ] **Step 4: Run the tests**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl && julia --project=<worktree> <worktree>/test/test_properties.jl`, then the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/kernel_thinning.jl test/test_kernel_thinning.jl test/test_properties.jl
git commit -m "feat: Select Compress++ through the compress keyword of kernel thinning

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Full suite on 1.10 and 1.12

- [ ] `julia -t 4 --project=<worktree> -e "using Pkg; Pkg.test()"`; `julia +1.12 --project=<worktree> <worktree>/test/test_standardize.jl`, `test_kernel_thinning.jl`, `test_properties.jl`. Fix anything that fails (`fix:` commit); otherwise no commit.

---

### Task 5: splitiq parity

**Files:**

- Modify: `splitiq/src/splitiq/split.py`, `multiplet.py`, `quality.py`; create `splitiq/tests/test_standardize.py`; modify `splitiq/tests/test_kernel_thinning.py`; docs `splitiq/docs/getting-started.md`, `overview.md`, `splitiq/README.md`

**Interfaces:**

- Produces: `standardize: bool = True` on `datasplit`, `select_rows`, `multiplet`, `splitquality`, `compare` (passed as the Julia keyword `standardize`); `compress: Literal['auto', 'always', 'never'] = 'auto'` on `datasplit`, `select_rows`, `multiplet` (kernel thinning only; a non-default value with another method raises `ValueError` mentioning `compress`); `_build_splitter(..., start, delta, compress)`.

- [ ] **Step 0:** `cd <worktree>/splitiq && make test` against the pre-built `.julia_dev` (already set up).

- [ ] **Step 1: Write the failing tests**

Create `splitiq/tests/test_standardize.py`:

```python
"""Parity tests for standardize=False in splitiq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import compare, datasplit, energydistance, multiplet, select_rows, splitquality


def _embeddings(seed: int = 0, n: int = 300, p: int = 8) -> np.ndarray:
    m = np.random.default_rng(seed).standard_normal((n, p))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_standardize_false_changes_the_selection_and_keeps_a_partition() -> None:
    data = _embeddings(1)
    raw = datasplit(data, ratio=0.2, method='herding', kernel='energy', standardize=False)
    std = datasplit(data, ratio=0.2, method='herding', kernel='energy')
    assert sorted([*raw.train_indices.tolist(), *raw.test_indices.tolist()]) == list(range(300))
    assert not np.array_equal(raw.test_indices, std.test_indices)
    idx = select_rows(data, 60, method='twinning', standardize=False)
    assert len(set(idx.tolist())) == 60
    folds = multiplet(data, 3, method='twinning', standardize=False)
    assert sorted(np.concatenate(folds).tolist()) == list(range(300))


def test_splitquality_and_compare_score_raw_rows() -> None:
    data = _embeddings(2)
    result = datasplit(data, ratio=0.2, method='herding', kernel='energy', standardize=False)
    q = splitquality(data, result, standardize=False)
    assert q == pytest.approx(energydistance(data[result.train_indices], data[result.test_indices]))
    table = compare(data, methods=['herding', 'twinning'], standardize=False)  # adapt to compare's actual signature
    assert len(table) == 2


def test_dataframes_are_rejected_without_standardization() -> None:
    df = pd.DataFrame({'x': np.random.default_rng(3).standard_normal(50), 'g': ['a', 'b'] * 25})
    with pytest.raises(ValueError, match='standardize'):
        datasplit(df, method='herding', standardize=False)
```

Before writing the `compare` assertion, read `splitiq/src/splitiq/quality.py`/`split.py` for `compare`'s actual Python signature and adapt the call (keep the intent: `standardize=False` reaches Julia's `compare`). Append to `tests/test_kernel_thinning.py`:

```python
def test_compress_options() -> None:
    data = np.random.default_rng(20).standard_normal((4000, 2))
    auto = select_rows(data, 100, method='kernel_thinning', seed=1)
    always = select_rows(data, 100, method='kernel_thinning', compress='always', seed=1)
    never = select_rows(data, 100, method='kernel_thinning', compress='never', seed=1)
    assert np.array_equal(auto, always)
    assert not np.array_equal(auto, never)
    with pytest.raises(ValueError, match='compress'):
        select_rows(data, 100, method='herding', compress='never')
    with pytest.raises(ValueError, match='compress'):
        select_rows(data, 100, method='kernel_thinning', compress='maybe')  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        select_rows(data, 100, method='kernel_thinning', compress='always', weights=np.ones(4000))
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: Implement**: `standardize` keyword on the
  five functions (forwarded as `standardize=standardize` in the Julia calls, documented: "``False``
  uses a numeric array as it is — no centering, scaling, or constant-column removal — for
  cosine-normalized embeddings; a DataFrame then raises ``ValueError``"); `compress` on
  `datasplit`/`select_rows`/`multiplet` with `_DEFAULT_COMPRESS = 'auto'`, `CompressMode =
  Literal['auto', 'always', 'never']`, a guard `if compress != _DEFAULT_COMPRESS and method !=
  'kernel_thinning': raise ValueError("'compress' is a kernel-thinning option; use
  method='kernel_thinning'")`, `_build_kernel_thinning_splitter(..., compress)` passing
  `compress=jl.Symbol(compress)` (validate the literal first: unknown string → `ValueError`
  mentioning `compress`). `SplitResult`/docstrings updated. **Step 4:** `make test && make lint &&
  make typecheck && make format && make docs`. **Step 5: Docs:** getting-started (an "Embeddings"
  bullet: `standardize=False` and `compress`), overview (type mapping row for `standardize`), README
  signatures. **Step 6: Commit** `feat(splitiq): Mirror standardize and compress`.

---

### Task 6: The example

**Files:**

- Create: `examples/Project.toml`, `examples/llm_data_selection.jl`, `examples/README.md`; modify `.gitignore` (add `examples/data/`); output `docs/src/assets/examples/llm_selection.md`

- [ ] **Step 1: Project**

`examples/Project.toml`:

```toml
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
DuckDB = "d2f5444f-75bc-4fdf-ac35-56f514c445e1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SPlit = "6b22f1a1-5f68-5715-9b23-279ae752cb98"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

Instantiate with `julia --project=<worktree>/examples -e 'using Pkg; Pkg.develop(path="<worktree>"); Pkg.instantiate()'` (the Manifest is git-ignored like `benchmark/`'s — check `.gitignore` and add `examples/Manifest.toml` if needed). Add `examples/data/` to `.gitignore`.

- [ ] **Step 2: Script**

`examples/llm_data_selection.jl`:

```julia
# Selecting LLM training data from an embedding matrix with SPlit.jl.
#
# Downloads a public arXiv-abstract embedding dataset (5,000 abstracts,
# MiniLM 384-d; CC0), cosine-normalizes the rows, and selects n = 500
# abstracts (10%) with every splitter under three target measures — the
# data itself, a quality-weighted version (abstract length as a stand-in),
# and a target sub-population (the `cs` archive) — against uniform random
# and K-center greedy baselines. Also times Compress++ against plain kernel
# thinning for n = 250 ≪ N. Prints a markdown table and writes it to
# docs/src/assets/examples/llm_selection.md.
#
# Run (a few minutes): julia -t auto --project=examples examples/llm_data_selection.jl
# Options: --model minilm|arcticlarge, --out PATH, --n 500

using SPlit, DataFrames, DuckDB, Downloads, LinearAlgebra, Printf, Random, Statistics

const MODEL = let i = findfirst(==("--model"), ARGS); i === nothing ? "minilm" : ARGS[i+1] end
const N_SELECT = let i = findfirst(==("--n"), ARGS); i === nothing ? 500 : parse(Int, ARGS[i+1]) end
const OUT = let i = findfirst(==("--out"), ARGS)
  i === nothing ? joinpath(@__DIR__, "..", "docs", "src", "assets", "examples", "llm_selection.md") : ARGS[i+1]
end
const DATASET = "https://huggingface.co/datasets/sondalex/arxiv-abstracts-2021-embeddings-10000/resolve/main/data/arxiv-abstract-$(MODEL).parquet"

# ---- data
datadir = joinpath(@__DIR__, "data")
mkpath(datadir)
file = joinpath(datadir, "arxiv-abstract-$(MODEL).parquet")
isfile(file) || (println("downloading $DATASET"); Downloads.download(DATASET, file))
con = DBInterface.connect(DuckDB.DB)
df = DataFrame(DBInterface.execute(con, "SELECT categories, length(content) AS len, embedding FROM read_parquet('$file')"))
E = Matrix{Float64}(reduce(hcat, [Float64.(coalesce.(e, 0.0)) for e in df.embedding])')
E ./= norm.(eachrow(E))                                   # cosine-normalize
N, p = size(E)
w = min.(Float64.(df.len), quantile(Float64.(df.len), 0.99))   # quality proxy: abstract length, clipped
is_cs = [any(==("cs"), string.(c)) for c in df.categories]
R = E[is_cs, :]
println("N = $N, p = $p, target rows (cs) = $(size(R, 1)), n = $N_SELECT")

# ---- baselines
random_rows(n, rng) = randperm(rng, N)[1:n]
function kcenter_greedy(E, n, rng)                       # farthest-first traversal (Sener & Savarese 2018)
  sel = [rand(rng, 1:N)]
  mind = fill(Inf, N)
  for _ = 2:n
    last = sel[end]
    @views for i = 1:N
      mind[i] = min(mind[i], norm(E[i, :] .- E[last, :]))
    end
    push!(sel, argmax(mind))
  end
  return sel
end

# ---- scoring: energy distance of the selection to the measure the setting optimizes, plus to the plain data
score_plain(sel) = energydistance(E[sel, :], E)
score_weighted(sel) = energydistance(E[sel, :], E; weights_y = w)
score_target(sel) = energydistance(E[sel, :], R)

rows = DataFrame(setting = String[], method = String[], optimized = Float64[], plain = Float64[], seconds = Float64[])
function record!(setting, method, sel, seconds, scorer)
  push!(rows, (setting, method, scorer(sel), score_plain(sel), seconds))
end

splitters(seed) = [
  ("herding · energy", HerdingSplitter(kernel = EnergyKernel(), rng = MersenneTwister(seed))),
  ("twinning", TwinningSplitter()),
  ("kernel thinning · energy", KernelThinningSplitter(rng = MersenneTwister(seed))),
  ("support points · energy", SupportPointSplitter(kappa = 1_000, max_iterations = 100, rng = MersenneTwister(seed))),
]

for (setting, kwargs, scorer, skip) in (
  ("plain", (;), score_plain, String[]),
  ("weights = length", (; weights = w), score_weighted, ["twinning"]),
  ("reference = cs", (; reference = R), score_target, ["twinning"]),
)
  # random: mean of 5 seeds
  rs = [random_rows(N_SELECT, MersenneTwister(100 + i)) for i = 1:5]
  push!(rows, (setting, "random", mean(scorer.(rs)), mean(score_plain.(rs)), 0.0))
  t = @elapsed sel = kcenter_greedy(E, N_SELECT, MersenneTwister(7))
  record!(setting, "k-center greedy", sel, t, scorer)
  for (label, s) in splitters(1)
    label in skip && continue
    selectrows(s, E[1:200, :], 20; standardize = false, kwargs...)   # warm-up (JIT)
    t = @elapsed sel = selectrows(s, E, N_SELECT; standardize = false, kwargs...)
    record!(setting, label, sel, t, scorer)
  end
end

# ---- Compress++ against plain kernel thinning at n ≪ N
let n = 250
  for (label, s) in (
    ("kernel thinning · compress = :never", KernelThinningSplitter(compress = :never, rng = MersenneTwister(3))),
    ("kernel thinning · compress = :always", KernelThinningSplitter(compress = :always, rng = MersenneTwister(3))),
  )
    selectrows(s, E[1:400, :], 20; standardize = false)
    t = @elapsed sel = selectrows(s, E, n; standardize = false)
    record!("plain, n = $n", label, sel, t, score_plain)
  end
  rs = [random_rows(n, MersenneTwister(200 + i)) for i = 1:5]
  push!(rows, ("plain, n = $n", "random", mean(score_plain.(rs)), mean(score_plain.(rs)), 0.0))
end

# ---- table
io = IOBuffer()
println(io, "| setting | method | energy distance to the optimized measure | energy distance to the data | seconds |")
println(io, "|---|---|---:|---:|---:|")
for r in eachrow(rows)
  @printf(io, "| %s | %s | %.3g | %.3g | %s |\n", r.setting, r.method, r.optimized, r.plain, r.method == "random" ? "–" : @sprintf("%.2g", r.seconds))
end
table = String(take!(io))
print(table)
mkpath(dirname(OUT))
write(OUT, table)
println("wrote $OUT")
```

Verify the DuckDB list column arrives as `Vector{Union{Missing,Float32}}` per row (it does on this machine; `coalesce` handles the `Missing` element type). If `energydistance` does not accept `weights_y` for one-sided weights, check `src/quality.jl`'s `energydistance` signature (it does: `weights_x`, `weights_y`).

- [ ] **Step 3: Run**

`julia -t auto --project=<worktree>/examples <worktree>/examples/llm_data_selection.jl` on an idle machine; a few minutes. Inspect the table: every optimized method must beat `random` on the measure it optimizes; if one does not, that is a finding to report (do not tune the example to hide it). Commit `examples/`, `.gitignore`, and `docs/src/assets/examples/llm_selection.md`:

```bash
git commit -m "docs: Add the LLM data-selection example on arXiv abstract embeddings

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

`examples/README.md`: three lines — what the script does, how to run it, where the table goes.

---

### Task 7: Docs

**Files:** `docs/src/40-llm-data-selection.md` (new), `docs/src/10-methods.md`, `docs/src/85-roadmap.md`, `docs/src/index.md`, `docs/src/30-python.md`, `README.md`, `AGENTS.md`

- [ ] **Step 0: Framing refresh (user request, 2026-09-04).** The package started as a train/test
  splitter and is now a distribution-preserving subset-selection library (splitting, k-fold
  multiplets, row selection toward the data, a weighted measure, or a reference, with four selection
  methods). Reframe the top-level texts accordingly, keeping every existing sentence that is still
  true and rewriting only the framing: `README.md` (title line/intro paragraph, the "what it does"
  bullets, the API section's grouping — splitting, selection, folds, quality), `docs/src/index.md`
  (the opening paragraph and "Overview" section; keep "Quick start" but lead with a selection
  example beside the split example), the module docstring at the top of `src/SPlit.jl` (if any) and
  the one-line `description` fields in `Project.toml` (if present), `CITATION.cff`
  (`title`/`abstract` if present), `splitiq/pyproject.toml` `description`, `splitiq/README.md` and
  `splitiq/docs/index.md`/`overview.md` intros, and the roadmap page's "Vision" paragraph (state
  that the reframing has happened: "SPlit.jl is now a distribution-preserving subset selection
  library …" while keeping the history). Do not rename the package or any API. Keep "SPlit" and the
  Joseph & Vakayil origin in the first sentence. One consistent one-line description to reuse
  everywhere: "Distribution-preserving subset selection for tabular data and embeddings: optimal
  train/test splits, k-fold multiplets, and training-data selection by support points, kernel
  herding, twinning, and kernel thinning."

- [ ] **Step 1: Methods page** — append two sections:

```markdown
## [Compress++](@id compress)

For ``n \ll N`` the ``O(N^2)`` cost of kernel thinning is dominated by the
first halving of all ``N`` rows and by the data term. Compress++
(Shetty, Dwivedi & Mackey, 2022) avoids both: it never halves more than
about ``2^{g+1}\sqrt{N}`` rows at once.

- **Compress.** A random permutation of the rows is split into four
  consecutive parts, each part is compressed recursively (sequences of
  at most ``4^g`` rows are returned as they are), the four results are
  concatenated and halved. The halving is kernel thinning of the block to
  half its size, and with probability one half its complement is kept
  instead, so each halving is unbiased for the block's mean embedding.
  The output has about ``2^g \sqrt N`` rows.
- **Thin.** Kernel thinning then selects ``n`` rows from the compressed
  set, with the compressed rows as its data term and swap candidates.

`KernelThinningSplitter(compress = :auto)` runs Compress++ when the
target measure is the data itself and the estimated cost
``4^g N (4\log_4 N + 1)`` is below plain kernel thinning's ``1.5 N^2``,
with ``g = \max(4, \lceil \log_2(2n/\sqrt N) \rceil)`` (the paper's
experiments use ``g = 4``; the second term keeps the compressed set at
about ``2n`` rows). At split ratios the rule never fires, so `datasplit`
is unchanged; it engages through `selectrows` and `multiplet` when ``n``
is a few percent of ``N`` or less. `:always` and `:never` force either
path.

**Differences from the paper.** The halving algorithm is kernel thinning
of the block (split and swap), the failure probability is split evenly
over the halvings and the final thinning, the four parts may differ in
size by one row, and ``g`` grows with ``n`` so that any ``n`` can be
requested. Compress++ is not defined for `weights` or `reference`;
`:auto` falls back to plain kernel thinning there.

## Skipping preprocessing

`standardize = false` on `datasplit`, `selectrows`, `multiplet`,
`splitquality`, and `compare` skips step 1 entirely: a numeric matrix (or
vector, as one column) is used as it is, with no centering, scaling, or
constant-column removal, and a reference is used the same way. Use it
when the rows already live in the geometry you want to preserve, such as
cosine-normalized embeddings, where per-column standardization would
distort angles. A `DataFrame` needs the encoding of step 1 and is
rejected with `standardize = false`.
```

Add the References entry for Shetty, Dwivedi & Mackey (2022), *ICLR*.

- [ ] **Step 2: New page** `docs/src/40-llm-data-selection.md`:

````markdown
# [Selecting LLM training data](@id llm-data-selection)

Given an embedding matrix (one row per document) and a budget of ``n``
rows, the splitters choose rows whose empirical distribution stays close
to a target measure: the whole corpus, a quality-weighted corpus, or a
smaller target set. This page is the workflow and a decision table; the
numbers come from `examples/llm_data_selection.jl` on 5,000 arXiv
abstracts embedded with MiniLM (384 dimensions, CC0).

## Workflow

1. Embed and cosine-normalize (`x / ‖x‖`), then pass `standardize = false`
   so the angles are preserved.
2. Pick the measure: nothing (match the corpus), `weights` (a quality
   score per row; the selection matches the weighted corpus), or
   `reference` (a target sample; the selection matches it while drawing
   from the corpus).
3. Pick the method from the table below, call `selectrows`, and check
   with `energydistance` (or `splitquality` for a split).

```julia
using SPlit, Random
idx = selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 500; standardize = false)
idx_w = selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 500; weights = quality, standardize = false)
idx_t = selectrows(KernelThinningSplitter(rng = MersenneTwister(1)), E, 500; reference = E_target, standardize = false)
few = selectrows(KernelThinningSplitter(), E, 100; standardize = false)   # Compress++ engages at n ≪ N
energydistance(E[idx, :], E)
```

```python
from splitiq import select_rows, energydistance
idx = select_rows(E, 500, method='herding', kernel='energy', standardize=False)
few = select_rows(E, 100, method='kernel_thinning', standardize=False)     # compress='auto'
```

## Which method

| N | n / N | weights or reference? | method | why |
|---|---|---|---|---|
| ≤ 10⁴ | any | any | `HerdingSplitter(EnergyKernel())` | exact, fastest at this size; `KernelThinningSplitter` when MMD is the criterion |
| 10⁵–10⁶ | split ratio | no | `TwinningSplitter` | `O(N log N)`; 140 s at N = 10⁶ |
| 10⁵–10⁶ | split ratio | yes | `HerdingSplitter` or `KernelThinningSplitter` | the only weighted/targeted methods; `O(N²)` |
| ≥ 10⁵ | ≤ a few % | no | `KernelThinningSplitter(compress = :auto)` | Compress++: near-linear |
| ≥ 10⁵ | ≤ a few % | yes | `HerdingSplitter` | weighted/targeted data term, `O(N²)` |

<fill from the example's table: one paragraph on what the example shows — how far each method is below random and K-center greedy under each measure, and the Compress++ vs plain timing at n = 250; link `assets/examples/llm_selection.md`.>

## What this does not settle

Combining a quality score with distribution matching through a weighted
empirical measure is natural but not validated in the literature; the
example only shows that the weighted selections track the weighted corpus
better than unweighted ones do. If downstream results disagree,
stratified selection by quality quantile is the alternative to compare
against.
````

- [ ] **Step 3: Roadmap** — M5 "Done (2026-09-04): …" with the four deliverables; Current-state rows `standardize = false` (done) and `Compress++` (done, `KernelThinningSplitter(compress = :auto)`); open questions: categorical handling in embedding mode → resolved by `standardize = false` (embeddings have no categorical columns); weighted energy distance as the combination rule → "first measured in `examples/llm_data_selection.jl` (see the LLM data-selection page); left open pending downstream results"; changelog `- 2026-09-04: M5 (embedding workflow, Compress++, data-selection guide) done.` Also fix the M6 "Evaluate only after M4 gives a baseline" if it reads stale (leave otherwise).

- [ ] **Step 4:** `docs/src/index.md` quick start gains `idx = selectrows(HerdingSplitter(), embeddings, 500; standardize = false)` with a comment; the overview bullet list mentions the new page. `docs/src/30-python.md`: `standardize=False` and `compress` in a short "Embeddings" subsection. `README.md`: `standardize` and `compress` in the API bullets; a pointer to the new page. `AGENTS.md` gotchas: "`standardize = false` skips constant-column removal too and rejects DataFrames; Compress++ (`compress`) is defined only for the data's own measure, `:auto` never fires at split ratios, `g = max(4, ⌈log₂(2n/√N)⌉)`; the example under `examples/` is not run in CI and its table under `docs/src/assets/examples/` is committed output."

- [ ] **Step 5:** Docs build clean; commit `docs: Add the LLM data-selection guide, Compress++, and standardize docs`.

---

### Task 8: Quality gate

- [ ] Julia suites on 1.10 (`-t 4`, `-t 1`) and 1.12; docs build; splitiq `make test && make lint && make typecheck && make docs`; `pre-commit run --all-files`; `git diff origin/main...HEAD -- test/ | grep "^-" | grep -v "^---"` empty; `src/optimizer.jl`, `src/herding.jl`, `src/twinning.jl`, `src/preprocessing.jl` unchanged; every new keyword defaults to today's behavior (grep the signatures).
- [ ] Report; the PR is opened only after the user confirms.
