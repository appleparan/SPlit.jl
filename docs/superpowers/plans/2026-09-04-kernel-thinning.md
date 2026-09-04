# Kernel Thinning (M4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `KernelThinningSplitter` (generalized kernel thinning with the target kernel: kernel halving, KT-SPLIT, KT-SWAP), mirror it in splitiq, add it to the main benchmark, and document it.

**Architecture:** A new `src/kernel_thinning.jl` holds the core (`_swap_params`, `_kernel_halving`, `_kt_split`, `_kt_swap`, `kernel_thinning`) and the splitter, which plugs into the `_select_rows`/`_with_kernel` protocol; `datasplit`, `selectrows`, `multiplet`, `compare`, `splitquality` are untouched. The KT-SWAP data term reuses herding's four `_data_term` methods through a small helper `_target_data_term` factored out of `herd` (behavior-neutral). `weights`/`reference` act on the KT-SWAP objective only.

**Tech Stack:** Julia 1.10+, Random/StatsBase (`randperm`, `sample`), Documenter.jl, CairoMakie (benchmark), Python 3.13 + juliacall (splitiq), pytest, uv, pre-commit.

**Spec:** `docs/superpowers/specs/2026-09-04-kernel-thinning-design.md`

## Global Constraints

- Existing public signatures and numerical results are unchanged; `datasplit`, `selectrows`, `multiplet`, `compare`, `splitquality` gain no new keywords; `HerdingSplitter` results stay bit-identical after the `_target_data_term` refactor.
- `KernelThinningSplitter(; kernel = EnergyKernel(), ratio = 0.2, delta = 0.5, n_threads = Threads.nthreads(), rng = Random.default_rng())`; `0 < delta < 1`.
- Sizes: `m = ⌊log₂(N/n)⌋`, `L = n·2^m`; `n > N ÷ 2` is an `ArgumentError` ("kernel thinning selects at most half of the rows"). The KT-SPLIT input is the first `L` rows of `randperm(rng, N)`.
- Kernel halving follows KT 2024 Alg. 2 exactly: `b² = k(x,x)+k(x′,x′)−2k(x,x′)`; `a = max(bσ√(2log(2/δ)), b²)`, `σ² ← σ² + b²·max(0, 1 + (b²−2a)σ²/a²)`; `α = Σ_{z∈S₂} f(z) − Σ_{z∈S₁} f(z)` with `f = k(x,·) − k(x′,·)`; swap with probability `min(1, ½·max(0, 1 − α/a))`; `b = 0` → no swap, `σ` unchanged, no rng draw. `δ_step = δ/(m·L)` at every level.
- KT-SWAP: baseline = `sample(rng, 1:N, n; replace = false)`; objective `(1/n²)Σ_{a,b∈S}k(a,b) − (2/n)Σ_{a∈S}d(a)`; one pass, positions `1:n`, candidates `z ∉ S`, swap only when `Δ(z) < 0`, ties to the lowest row index; `iterations` = number of swaps, `converged = true`.
- Threaded sums use a fixed chunk size of 1,024 with partial sums added in order, so results are identical for any `n_threads`.
- All randomness through the caller's `rng` (order: `randperm`, then coin flips level by level, then the baseline sample); nothing in `src/` seeds or prints. Never cite or compare with other implementations. Docstrings sit directly above what they document.
- Existing tests are append-only. `src/herding.jl` changes only by the extraction of `_target_data_term`.
- Every Julia capability lands in splitiq in this branch, with tests and a docs mention.
- Run one test file with `julia --project=<worktree> <worktree>/test/<file>.jl`; the suite with `julia --project=<worktree> -e "using Pkg; Pkg.test()"`; also check new files on `julia +1.12`. Commit messages `<type>: <Capitalized description>` + trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; pre-commit runs on commit (JuliaFormatter 1.0.62 under Julia 1.10, markdownlint, ruff), never bypass it.
- Work only in `/home/appleparan/src/SPlit.jl/.claude/worktrees/feat-kernel-thinning` (branch `feat/kernel-thinning`). Use absolute paths in shell commands.

---

## File structure

| File | Responsibility |
|---|---|
| `src/kernel_thinning.jl` (new) | `_KH_CHUNK`, `_swap_params`, `_kernel_diff_sum`, `_kernel_halving`, `_kt_split`, `_self_kernel_sum`, `_coreset_sums!`, `_kt_swap`, `kernel_thinning`, `KernelThinningSplitter`, `_with_kernel`, `_select_rows`, `show` |
| `src/herding.jl` | `_target_data_term` extracted from `herd` |
| `src/SPlit.jl` | `include("kernel_thinning.jl")` after `multiplet.jl`; export `KernelThinningSplitter` |
| `test/test_kernel_thinning.jl` (new), `test/test_herding.jl` (append one test), `test/test_properties.jl` (append), `test/runtests.jl` | tests |
| `benchmark/run.jl` | two new methods, seven-entry plotting |
| `docs/src/assets/benchmarks/results.md`, `quality.png`, `time.png`, `selection.png` | regenerated |
| `docs/src/10-methods.md`, `20-benchmarks.md`, `85-roadmap.md`, `index.md`, `30-python.md`, `AGENTS.md`, `README.md` | docs |
| `splitiq/src/splitiq/split.py`, `multiplet.py`, `splitiq/tests/test_kernel_thinning.py` (new), `splitiq/docs/getting-started.md`, `overview.md`, `splitiq/README.md` | parity |

---

### Task 1: Kernel halving core

**Files:**

- Create: `src/kernel_thinning.jl`
- Modify: `src/SPlit.jl` (include only)
- Create: `test/test_kernel_thinning.jl`; modify `test/runtests.jl`

**Interfaces:**

- Produces:
  - `const _KH_CHUNK = 1_024`
  - `_swap_params(σ::Float64, b::Float64, δ::Float64) -> (a::Float64, σ′::Float64)`
  - `_kernel_diff_sum(kernel, Xt::Matrix{Float64}, idx::Vector{Int}, x::AbstractVector, x′::AbstractVector, n_threads::Int) -> Float64` (= `Σ_{j∈idx} k(x_j,x) − k(x_j,x′)`, `Xt` is `permutedims(X)`, rows as columns)
  - `_kernel_halving(kernel, Xt, seq::Vector{Int}, δ_step::Float64, rng::AbstractRNG; n_threads::Int) -> (S₁::Vector{Int}, S₂::Vector{Int})`

- [ ] **Step 1: Write the failing tests**

Create `test/test_kernel_thinning.jl`:

```julia
using Test
using SPlit
using Random
using Statistics
using DataFrames
using LinearAlgebra

# Kernel halving exactly as KT 2024, Alg. 2, with the paper's α form
# (Σ over all previous points minus twice the Σ over S₁), consuming one rng
# draw per step with a > 0, as the implementation does.
function naive_kernel_halving(kernel, X, seq, δ, rng)
  S1 = Int[]
  S2 = Int[]
  σ = 0.0
  for i = 1:(length(seq)÷2)
    x, x′ = seq[2i-1], seq[2i]
    k(u, v) = SPlit.kernelvalue(kernel, X[u, :], X[v, :])
    b = sqrt(max(k(x, x) + k(x′, x′) - 2k(x, x′), 0.0))
    if b > 0
      a = max(b * σ * sqrt(2 * log(2 / δ)), b^2)
      σ = sqrt(σ^2 + b^2 * max(0.0, 1 + (b^2 - 2a) * σ^2 / a^2))
      prev = seq[1:(2i-2)]
      α = sum(k(j, x) - k(j, x′) for j in prev; init = 0.0) -
          2 * sum(k(z, x) - k(z, x′) for z in S1; init = 0.0)
      if rand(rng) < min(1.0, 0.5 * max(0.0, 1 - α / a))
        x, x′ = x′, x
      end
    end
    push!(S1, x)
    push!(S2, x′)
  end
  return S1, S2
end

@testset "kernel halving" begin
  @testset "swap parameters follow the paper's update" begin
    a, σ = SPlit._swap_params(0.0, 2.0, 0.1)
    @test a == 4.0 && σ == 2.0                      # σ = 0: a = b², σ² = b²
    a2, σ2 = SPlit._swap_params(σ, 1.0, 0.1)
    @test a2 == max(1.0 * σ * sqrt(2 * log(20.0)), 1.0)
    @test σ2^2 ≈ σ^2 + 1.0 * max(0.0, 1 + (1.0 - 2a2) * σ^2 / a2^2)
    @test SPlit._swap_params(1.5, 0.0, 0.1) == (0.0, 1.5)   # identical rows: no threshold, σ kept
  end

  @testset "difference sums match a plain loop and ignore n_threads" begin
    X = SPlit.preprocess(randn(MersenneTwister(1), 3_000, 3))
    Xt = permutedims(X)
    idx = collect(1:2_900)
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      @views ref = sum(SPlit.kernelvalue(kernel, X[j, :], X[2_999, :]) -
                       SPlit.kernelvalue(kernel, X[j, :], X[3_000, :]) for j in idx)
      @views s1 = SPlit._kernel_diff_sum(kernel, Xt, idx, Xt[:, 2_999], Xt[:, 3_000], 1)
      @views s4 = SPlit._kernel_diff_sum(kernel, Xt, idx, Xt[:, 2_999], Xt[:, 3_000], 4)
      @test s1 ≈ ref
      @test s1 == s4
    end
  end

  @testset "halving equals the paper's algorithm step for step" begin
    for (kernel, seed) in ((EnergyKernel(), 2), (GaussianKernel(0.8), 3))
      X = SPlit.preprocess(randn(MersenneTwister(seed), 201, 2))
      seq = randperm(MersenneTwister(seed + 10), 201)
      S1, S2 = SPlit._kernel_halving(kernel, permutedims(X), seq, 0.5 / 200, MersenneTwister(7); n_threads = 2)
      T1, T2 = naive_kernel_halving(kernel, X, seq, 0.5 / 200, MersenneTwister(7))
      @test S1 == T1 && S2 == T2
      @test length(S1) == 100 && sort(vcat(S1, S2)) == sort(seq[1:200])   # odd trailing row dropped
    end
  end

  @testset "reproducible under the same rng, independent of n_threads" begin
    X = SPlit.preprocess(randn(MersenneTwister(4), 400, 3))
    Xt = permutedims(X)
    seq = collect(1:400)
    a = SPlit._kernel_halving(EnergyKernel(), Xt, seq, 1e-3, MersenneTwister(5); n_threads = 1)
    b = SPlit._kernel_halving(EnergyKernel(), Xt, seq, 1e-3, MersenneTwister(5); n_threads = 4)
    @test a == b
  end

  @testset "halves are more balanced than random halves" begin
    rng = MersenneTwister(6)
    N = 400
    c = rand(rng, 1:4, N)
    centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
    X = SPlit.preprocess(centers[c, :] .+ randn(rng, N, 2))
    Xt = permutedims(X)
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      S1, _ = SPlit._kernel_halving(kernel, Xt, collect(1:N), 0.5 / N, MersenneTwister(8); n_threads = 2)
      q = mmd(X[S1, :], X, kernel)
      random_q = mean(mmd(X[randperm(MersenneTwister(100 + i), N)[1:200], :], X, kernel) for i = 1:20)
      @test q < random_q
    end
  end
end
```

Add `include("test_kernel_thinning.jl")` to `test/runtests.jl` after `include("test_multiplet.jl")`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: `UndefVarError: _swap_params`.

- [ ] **Step 3: Implement**

Create `src/kernel_thinning.jl`:

```julia
"""
Generalized kernel thinning with the target kernel (Dwivedi & Mackey 2022,
Alg. 1, 1a, 1b; kernel halving from Dwivedi & Mackey 2024, Alg. 2).
KT-SPLIT halves a shuffled sequence of rows `m` times by probabilistic
kernel halving into `2^m` candidate subsets; KT-SWAP keeps the candidate
(or a uniform random baseline) with the smallest MMD to the target measure
and refines it by one pass of best single-row swaps over the whole data.
"""

using Random
using StatsBase: sample

# Rows per chunk of the threaded kernel sums. Partial sums are added in
# chunk order, so results do not depend on `n_threads`.
const _KH_CHUNK = 1_024

# Swap threshold and updated swapping parameter (KT 2024, Alg. 2,
# `get_swap_params`). `b = 0` (identical rows) leaves σ unchanged and
# returns a = 0, which callers treat as "no swap".
function _swap_params(σ::Float64, b::Float64, δ::Float64)
  b == 0.0 && return 0.0, σ
  a = max(b * σ * sqrt(2 * log(2 / δ)), b^2)
  σ2 = σ^2 + b^2 * max(0.0, 1 + (b^2 - 2a) * σ^2 / a^2)
  return a, sqrt(σ2)
end

# Σ_{j ∈ idx} (k(x_j, x) − k(x_j, x′)) over columns `idx` of `Xt`, in
# fixed chunks so the floating-point result is independent of `n_threads`.
function _kernel_diff_sum(
  kernel::SplitKernel,
  Xt::Matrix{Float64},
  idx::Vector{Int},
  x::AbstractVector,
  x′::AbstractVector,
  n_threads::Int,
)
  m = length(idx)
  m == 0 && return 0.0
  chunks = collect(Iterators.partition(1:m, _KH_CHUNK))
  partial = zeros(length(chunks))
  chunk_sum(chunk) = begin
    s = 0.0
    @views for t in chunk
      j = idx[t]
      s += kernelvalue(kernel, Xt[:, j], x) - kernelvalue(kernel, Xt[:, j], x′)
    end
    s
  end
  if n_threads == 1 || length(chunks) == 1
    for (c, chunk) in enumerate(chunks)
      partial[c] = chunk_sum(chunk)
    end
  else
    @sync for (c, chunk) in enumerate(chunks)
      Threads.@spawn partial[c] = chunk_sum(chunk)
    end
  end
  return sum(partial)
end

"""
    _kernel_halving(kernel, Xt, seq, δ_step, rng; n_threads) -> (S₁, S₂)

Kernel halving (Dwivedi & Mackey 2024, Alg. 2) of the row sequence `seq`
(indices into the columns of `Xt = permutedims(X)`): rows are taken two at
a time, one goes to each half, and the assignment is swapped with
probability `min(1, ½(1 − α/a)₊)` where `α = Σ_{z∈S₂} f(z) − Σ_{z∈S₁} f(z)`,
`f = k(x,·) − k(x′,·)`, and `a` is the paper's adaptive threshold for the
per-step failure probability `δ_step`. A trailing unpaired row is dropped.
"""
function _kernel_halving(
  kernel::SplitKernel,
  Xt::Matrix{Float64},
  seq::Vector{Int},
  δ_step::Float64,
  rng::AbstractRNG;
  n_threads::Int = Threads.nthreads(),
)
  steps = length(seq) ÷ 2
  S1 = sizehint!(Int[], steps)
  S2 = sizehint!(Int[], steps)
  σ = 0.0
  for i = 1:steps
    x, x′ = seq[2i-1], seq[2i]
    @views xv, xv′ = Xt[:, x], Xt[:, x′]
    b2 = kernelvalue(kernel, xv, xv) + kernelvalue(kernel, xv′, xv′) - 2kernelvalue(kernel, xv, xv′)
    a, σ = _swap_params(σ, sqrt(max(b2, 0.0)), δ_step)
    if a > 0
      α = _kernel_diff_sum(kernel, Xt, S2, xv, xv′, n_threads) -
          _kernel_diff_sum(kernel, Xt, S1, xv, xv′, n_threads)
      if rand(rng) < min(1.0, 0.5 * max(0.0, 1 - α / a))
        x, x′ = x′, x
      end
    end
    push!(S1, x)
    push!(S2, x′)
  end
  return S1, S2
end
```

In `src/SPlit.jl`, add `include("kernel_thinning.jl")` after `include("multiplet.jl")`.

- [ ] **Step 4: Run the tests**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: all pass. If "halving equals the paper's algorithm step for step" fails, print the first differing step and compare `α` from both forms: they must agree to about `1e-9`; a coin flip that lands within that band of `p` is the only legitimate cause, and then change the seed rather than the code.

- [ ] **Step 5: Commit**

```bash
git add src/kernel_thinning.jl src/SPlit.jl test/test_kernel_thinning.jl test/runtests.jl
git commit -m "feat: Add kernel halving with the paper's adaptive swap threshold

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: KT-SPLIT, KT-SWAP, and `kernel_thinning`

**Files:**

- Modify: `src/kernel_thinning.jl` (append), `src/herding.jl` (extract `_target_data_term`)
- Test: `test/test_kernel_thinning.jl` (append), `test/test_herding.jl` (append)

**Interfaces:**

- Consumes: `_kernel_halving`, `_resolve_target(X, weights, target, target_weights) -> (R, w_hat, w_bar)`, the four `_data_term` methods, `mmd`.
- Produces:
  - `_target_data_term(kernel, X, weights, target, target_weights, n_threads) -> Vector{Float64}` (in `src/herding.jl`)
  - `_kt_split(kernel, Xt, seq, m::Int, δ::Float64, rng; n_threads) -> Vector{Vector{Int}}` (`2^m` candidates of length `length(seq) ÷ 2^m`, level order)
  - `_self_kernel_sum(kernel, Xt, S, n_threads) -> Float64` (`Σ_{a,b∈S} k(a,b)`)
  - `_coreset_sums(kernel, Xt, S, N, n_threads) -> Vector{Float64}` (`c[y] = Σ_{a∈S} k(y,a)`)
  - `_kt_swap(kernel, Xt, candidates, baseline, d, n_threads) -> (rows::Vector{Int}, swaps::Int)`
  - `kernel_thinning(kernel, X, n; delta = 0.5, weights = nothing, target = nothing, target_weights = nothing, n_threads = Threads.nthreads(), rng = Random.default_rng()) -> (rows, swaps)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_kernel_thinning.jl`:

```julia
# KT-SWAP objective up to its constant: (1/n²) Σ_{a,b∈S} k(a,b) − (2/n) Σ_{a∈S} d(a).
function swap_objective(kernel, X, S, d)
  n = length(S)
  self = sum(SPlit.kernelvalue(kernel, X[a, :], X[b, :]) for a in S, b in S)
  return self / n^2 - 2 * sum(d[S]) / n
end

@testset "KT-SPLIT and KT-SWAP" begin
  X = SPlit.preprocess(randn(MersenneTwister(20), 480, 3))
  Xt = permutedims(X)

  @testset "KT-SPLIT: 2^m candidates of size n partitioning the sequence" begin
    seq = randperm(MersenneTwister(21), 480)
    cands = SPlit._kt_split(EnergyKernel(), Xt, seq, 3, 0.5, MersenneTwister(22); n_threads = 2)
    @test length(cands) == 8
    @test all(c -> length(c) == 60, cands)
    @test sort(reduce(vcat, cands)) == sort(seq)
    # level order: the first two candidates come from the first-level S₁
    S1, _ = SPlit._kernel_halving(EnergyKernel(), Xt, seq, 0.5 / (3 * 480), MersenneTwister(22); n_threads = 2)
    @test sort(vcat(cands[1], cands[2], cands[3], cands[4])) == sort(S1)
  end

  @testset "KT-SWAP: never worse than the baseline, monotone, distinct rows" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      d = SPlit._data_term(kernel, X, 2)
      seq = randperm(MersenneTwister(23), 480)
      cands = SPlit._kt_split(kernel, Xt, seq, 2, 0.5, MersenneTwister(24); n_threads = 2)
      baseline = randperm(MersenneTwister(25), 480)[1:120]
      rows, swaps = SPlit._kt_swap(kernel, Xt, cands, baseline, d, 2)
      @test length(rows) == 120 && allunique(rows) && all(in(1:480), rows)
      obj = swap_objective(kernel, X, rows, d)
      @test obj <= minimum(swap_objective(kernel, X, c, d) for c in vcat(cands, [baseline])) + 1e-12
      @test mmd(X[rows, :], X, kernel) <= mmd(X[baseline, :], X, kernel) + 1e-12
      @test swaps >= 0
      # the objective differs from the exact MMD² by a constant: check on two candidates
      c1, c2 = cands[1], cands[2]
      @test (swap_objective(kernel, X, c1, d) - swap_objective(kernel, X, c2, d)) ≈
            (mmd(X[c1, :], X, kernel) - mmd(X[c2, :], X, kernel)) atol = 1e-9
    end
  end

  @testset "KT-SWAP result is independent of n_threads" begin
    d = SPlit._data_term(EnergyKernel(), X, 1)
    cands = SPlit._kt_split(EnergyKernel(), Xt, collect(1:480), 2, 0.5, MersenneTwister(26); n_threads = 1)
    baseline = collect(1:120)
    @test SPlit._kt_swap(EnergyKernel(), Xt, cands, baseline, d, 1) ==
          SPlit._kt_swap(EnergyKernel(), Xt, cands, baseline, d, 4)
  end

  @testset "kernel_thinning: sizes, validation, reproducibility" begin
    rows, swaps = SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(30))
    @test length(rows) == 96 && allunique(rows)
    @test (rows, swaps) == SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(30))
    @test rows != SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(31))[1]
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 241)   # > N ÷ 2
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 0)
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 96; delta = 0.0)
    @test_throws ArgumentError SPlit.kernel_thinning(GaussianKernel(), X, 96)   # unresolved kernel
    # ratio 0.25 uses every row in KT-SPLIT (L = N); 0.2 uses L = 0.8 N
    @test length(SPlit.kernel_thinning(EnergyKernel(), X, 120; rng = MersenneTwister(32))[1]) == 120
  end

  @testset "kernel_thinning beats random subsets under its own discrepancy" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      rows, _ = SPlit.kernel_thinning(kernel, X, 96; rng = MersenneTwister(33))
      q = mmd(X[rows, :], X, kernel)
      random_q = mean(mmd(X[randperm(MersenneTwister(200 + i), 480)[1:96], :], X, kernel) for i = 1:20)
      @test q < random_q
    end
  end

  @testset "weights and target enter through the swap objective" begin
    w = ones(480)
    @test SPlit.kernel_thinning(EnergyKernel(), X, 96; weights = w, rng = MersenneTwister(40)) ==
          SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(40))
    heavy = X[:, 1] .> 0
    w2 = ifelse.(heavy, 20.0, 1.0)
    plain, _ = SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(41))
    weighted, _ = SPlit.kernel_thinning(EnergyKernel(), X, 96; weights = w2, rng = MersenneTwister(41))
    @test count(heavy[weighted]) > count(heavy[plain])
    R = X[heavy, :]
    targeted, _ = SPlit.kernel_thinning(EnergyKernel(), X, 96; target = R, rng = MersenneTwister(41))
    @test count(heavy[targeted]) > count(heavy[plain])
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 96; weights = w2, target = R)
  end
end
```

Append to `test/test_herding.jl` (inside a new top-level testset at the end):

```julia
@testset "_target_data_term matches the four data terms" begin
  X = SPlit.preprocess(randn(MersenneTwister(300), 60, 2))
  R = X[1:20, :]
  w = rand(MersenneTwister(301), 60)
  v = rand(MersenneTwister(302), 20)
  k = GaussianKernel(1.0)
  @test SPlit._target_data_term(k, X, nothing, nothing, nothing, 2) == SPlit._data_term(k, X, 2)
  @test SPlit._target_data_term(k, X, w, nothing, nothing, 2) == SPlit._data_term(k, X, w ./ sum(w), 2)
  @test SPlit._target_data_term(k, X, nothing, R, nothing, 2) == SPlit._data_term(k, X, R, 2)
  @test SPlit._target_data_term(k, X, nothing, R, v, 2) == SPlit._data_term(k, X, R, v ./ sum(v), 2)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: `UndefVarError: _kt_split`.

- [ ] **Step 3: Implement**

In `src/herding.jl`, add before `herd`:

```julia
# Data term of every row of `X` against the target measure: the data
# itself, the weighted data, or a (weighted) reference. Validation happens in
# `_resolve_target`; the four `_data_term` methods do the work.
function _target_data_term(kernel::SplitKernel, X::Matrix{Float64}, weights, target, target_weights, n_threads::Int)
  R, _, v_bar = _resolve_target(X, weights, target, target_weights)
  if target === nothing
    return v_bar === nothing ? _data_term(kernel, X, n_threads) : _data_term(kernel, X, v_bar, n_threads)
  end
  return v_bar === nothing ? _data_term(kernel, X, R, n_threads) : _data_term(kernel, X, R, v_bar, n_threads)
end
```

and in `herd`, replace the `R, _, v_bar = _resolve_target(...)` line and the `d = if ... end` block by `d = _target_data_term(kernel, X, weights, target, target_weights, n_threads)`.

Append to `src/kernel_thinning.jl`:

```julia
# KT-SPLIT (Dwivedi & Mackey 2022, Alg. 1a) as `m` rounds of kernel halving
# (KT 2024, Sec. 5.2): level j halves each of the 2^(j−1) sequences of level
# j − 1 with per-step failure probability δ/(m·L). Candidates are returned in
# level order (S₁ before S₂).
function _kt_split(
  kernel::SplitKernel,
  Xt::Matrix{Float64},
  seq::Vector{Int},
  m::Int,
  δ::Float64,
  rng::AbstractRNG;
  n_threads::Int = Threads.nthreads(),
)
  δ_step = δ / (m * length(seq))
  level = [seq]
  for _ = 1:m
    next = Vector{Vector{Int}}()
    for s in level
      S1, S2 = _kernel_halving(kernel, Xt, s, δ_step, rng; n_threads)
      push!(next, S1)
      push!(next, S2)
    end
    level = next
  end
  return level
end

# Threaded map over 1:N in fixed chunks; `f(y)` writes nothing shared.
function _chunked_foreach(f, N::Int, n_threads::Int)
  chunks = collect(Iterators.partition(1:N, _KH_CHUNK))
  if n_threads == 1 || length(chunks) == 1
    for chunk in chunks, y in chunk
      f(y)
    end
  else
    @sync for chunk in chunks
      Threads.@spawn for y in chunk
        f(y)
      end
    end
  end
  return nothing
end

# Σ_{a,b∈S} k(a, b), threaded over a.
function _self_kernel_sum(kernel::SplitKernel, Xt::Matrix{Float64}, S::Vector{Int}, n_threads::Int)
  rowsum = zeros(length(S))
  _chunked_foreach(length(S), n_threads) do i
    s = 0.0
    @views xa = Xt[:, S[i]]
    @views for b in S
      s += kernelvalue(kernel, xa, Xt[:, b])
    end
    rowsum[i] = s
  end
  return sum(rowsum)
end

# c[y] = Σ_{a∈S} k(y, a) for every row y, threaded over y.
function _coreset_sums(kernel::SplitKernel, Xt::Matrix{Float64}, S::Vector{Int}, N::Int, n_threads::Int)
  c = zeros(N)
  _chunked_foreach(N, n_threads) do y
    s = 0.0
    @views xy = Xt[:, y]
    @views for a in S
      s += kernelvalue(kernel, xy, Xt[:, a])
    end
    c[y] = s
  end
  return c
end

"""
    _kt_swap(kernel, Xt, candidates, baseline, d, n_threads) -> (rows, swaps)

KT-SWAP (Dwivedi & Mackey 2022, Alg. 1b) with the target measure encoded in
the data term `d` (`d[z] = Σ_l v̄_l k(z, r_l)`). Keeps the candidate (the
`baseline` included) with the smallest `(1/n²) Σ_{a,b∈S} k(a,b) − (2/n) Σ_{a∈S} d(a)`,
then makes one pass over its positions, replacing each row by the row of the
data outside the coreset that lowers the objective most (only if it lowers
it; ties to the lowest row index). Returns the refined coreset in position
order and the number of replacements.
"""
function _kt_swap(
  kernel::SplitKernel,
  Xt::Matrix{Float64},
  candidates::Vector{Vector{Int}},
  baseline::Vector{Int},
  d::Vector{Float64},
  n_threads::Int,
)
  N = size(Xt, 2)
  n = length(baseline)
  all(c -> length(c) == n, candidates) ||
    throw(ArgumentError("every candidate must have the baseline's size $n"))
  cands = vcat(candidates, [baseline])
  objective(S) = _self_kernel_sum(kernel, Xt, S, n_threads) / n^2 - 2 * sum(@view d[S]) / n
  best = argmin(map(objective, cands))
  S = copy(cands[best])
  inS = falses(N)
  inS[S] .= true
  c = _coreset_sums(kernel, Xt, S, N, n_threads)
  kdiag = [(@views kernelvalue(kernel, Xt[:, y], Xt[:, y])) for y = 1:N]
  swaps = 0
  chunks = collect(Iterators.partition(1:N, _KH_CHUNK))
  best_z = zeros(Int, length(chunks))
  best_Δ = zeros(length(chunks))
  for i = 1:n
    s = S[i]
    @views xs = Xt[:, s]
    base = -(kdiag[s] + 2 * (c[s] - kdiag[s])) / n^2 + 2 * d[s] / n
    fill!(best_z, 0)
    fill!(best_Δ, 0.0)
    scan(ci) = begin
      bz, bΔ = 0, 0.0
      @views for z in chunks[ci]
        inS[z] && continue
        Δ = (kdiag[z] + 2 * (c[z] - kernelvalue(kernel, Xt[:, z], xs))) / n^2 - 2 * d[z] / n + base
        if Δ < bΔ
          bz, bΔ = z, Δ
        end
      end
      best_z[ci], best_Δ[ci] = bz, bΔ
    end
    if n_threads == 1 || length(chunks) == 1
      foreach(scan, eachindex(chunks))
    else
      @sync for ci in eachindex(chunks)
        Threads.@spawn scan(ci)
      end
    end
    ci = argmin(best_Δ)                       # first chunk with the smallest Δ → lowest row index among ties
    best_Δ[ci] < 0.0 || continue
    z = best_z[ci]
    @views xz = Xt[:, z]
    _chunked_foreach(N, n_threads) do y
      @views c[y] += kernelvalue(kernel, Xt[:, y], xz) - kernelvalue(kernel, Xt[:, y], xs)
    end
    S[i] = z
    inS[s] = false
    inS[z] = true
    swaps += 1
  end
  return S, swaps
end

"""
    kernel_thinning(kernel, X, n; delta = 0.5, weights = nothing, target = nothing,
                    target_weights = nothing, n_threads = Threads.nthreads(),
                    rng = Random.default_rng()) -> (rows, swaps)

Select `n ≤ N/2` rows of `X` by generalized kernel thinning with the target
kernel (Dwivedi & Mackey 2022): with `m = ⌊log₂(N/n)⌋`, the first `L = n·2^m`
rows of a random permutation are split by `m` rounds of kernel halving into
`2^m` candidate subsets of size `n` (KT-SPLIT), and KT-SWAP keeps the candidate
(or a uniform random baseline) with the smallest MMD² to the target measure
and refines it by one pass of best single-row swaps over all `N` rows. `delta`
is the failure probability `δ` of the paper's guarantees (`δ_i = δ/L`);
`weights`, `target`, `target_weights` define the target measure as in
[`herd`](@ref) and act on KT-SWAP only. Cost: `O(L²)` kernel evaluations for
KT-SPLIT, `O(N²)` for the data term, `O(nN)` for KT-SWAP, all threaded.
Deterministic given `rng` and independent of `n_threads`.

# Differences from the paper

Target-kernel thinning is used (no square-root kernel). The papers thin `N`
to `⌊N/2^m⌋`; here `n` is given and only `L = n·2^m` shuffled rows enter
KT-SPLIT, the rest take part through KT-SWAP (equal to the paper when
`N/n = 2^m`). Swap candidates exclude rows already in the coreset so the
result is a set of distinct rows. `weights`/`target` change only the
KT-SWAP objective. Compress++ is not implemented.
"""
function kernel_thinning(
  kernel::SplitKernel,
  X::Matrix{Float64},
  n::Int;
  delta::Real = 0.5,
  weights::Union{Nothing,AbstractVector} = nothing,
  target::Union{Nothing,AbstractMatrix} = nothing,
  target_weights::Union{Nothing,AbstractVector} = nothing,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  isresolved(kernel) ||
    throw(ArgumentError("kernel parameters must be resolved; call resolve first"))
  N = size(X, 1)
  0 < n <= N ÷ 2 || throw(
    ArgumentError("kernel thinning selects at most half of the rows: n must be in 1:$(N ÷ 2), got $n"),
  )
  0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1), got $delta"))
  d = _target_data_term(kernel, X, weights, target, target_weights, n_threads)
  m = 0
  while n * 2^(m + 1) <= N
    m += 1
  end
  L = n * 2^m
  seq = randperm(rng, N)[1:L]
  Xt = permutedims(X)
  candidates = _kt_split(kernel, Xt, seq, m, Float64(delta), rng; n_threads)
  baseline = sample(rng, 1:N, n; replace = false)
  return _kt_swap(kernel, Xt, candidates, baseline, d, n_threads)
end
```

- [ ] **Step 4: Run the tests**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl && julia --project=<worktree> <worktree>/test/test_herding.jl`
Expected: all pass, and the herding file's existing tests unchanged in count. If a "beats random" property fails, report the two numbers; do not loosen.

- [ ] **Step 5: Commit**

```bash
git add src/kernel_thinning.jl src/herding.jl test/test_kernel_thinning.jl test/test_herding.jl
git commit -m "feat: Add KT-SPLIT, KT-SWAP, and kernel_thinning over the target measure

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `KernelThinningSplitter`

**Files:**

- Modify: `src/kernel_thinning.jl` (append), `src/SPlit.jl` (export)
- Test: `test/test_kernel_thinning.jl` (append), `test/test_properties.jl` (append)

**Interfaces:**

- Consumes: `kernel_thinning`; the `_select_rows`/`_with_kernel` protocol.
- Produces: `struct KernelThinningSplitter{K<:SplitKernel,R<:AbstractRNG} <: AbstractSplitter` with fields `kernel::K`, `ratio::Float64`, `delta::Float64`, `n_threads::Int`, `rng::R`; keyword constructor as in the Global Constraints.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_kernel_thinning.jl`:

```julia
@testset "KernelThinningSplitter" begin
  @testset "construction, validation, show" begin
    s = KernelThinningSplitter()
    @test s isa AbstractSplitter
    @test s.kernel isa EnergyKernel && s.ratio == 0.2 && s.delta == 0.5
    @test KernelThinningSplitter(kernel = GaussianKernel(), ratio = 1 // 4, delta = 0.1).delta == 0.1
    @test_throws ArgumentError KernelThinningSplitter(ratio = 0.0)
    @test_throws ArgumentError KernelThinningSplitter(delta = 1.0)
    @test_throws ArgumentError KernelThinningSplitter(n_threads = 0)
    @test occursin("KernelThinningSplitter(kernel=EnergyKernel(), ratio=0.2, delta=0.5)", sprint(show, s))
  end

  @testset "datasplit: partition, sizes, report, both kernels" begin
    data = randn(MersenneTwister(50), 300, 3)
    for kernel in (EnergyKernel(), GaussianKernel())
      r = datasplit(KernelThinningSplitter(kernel = kernel, rng = MersenneTwister(51)), data)
      @test length(test_indices(r)) == 60 && length(train_indices(r)) == 240
      @test sort(vcat(train_indices(r), test_indices(r))) == 1:300
      @test r.converged && r.iterations >= 0 && r.selected === :test
      @test SPlit.isresolved(r.method.kernel)
    end
    r25 = datasplit(KernelThinningSplitter(ratio = 0.25, rng = MersenneTwister(52)), data)
    @test length(test_indices(r25)) == 75
  end

  @testset "ratio > 0.5 puts the selected rows in train; selectrows agrees" begin
    data = randn(MersenneTwister(53), 200, 2)
    r = datasplit(KernelThinningSplitter(ratio = 0.7, rng = MersenneTwister(54)), data)
    @test length(test_indices(r)) == 140 && r.selected === :train
    @test train_indices(r) == selectrows(KernelThinningSplitter(ratio = 0.7, rng = MersenneTwister(54)), data, 60)
    @test_throws ArgumentError selectrows(KernelThinningSplitter(), data, 101)
  end

  @testset "reproducible with rng; DataFrame and vector inputs; compare" begin
    data = randn(MersenneTwister(55), 240, 2)
    a = datasplit(KernelThinningSplitter(rng = MersenneTwister(1)), data)
    b = datasplit(KernelThinningSplitter(rng = MersenneTwister(1)), data)
    @test test_indices(a) == test_indices(b)
    df = DataFrame(x = randn(MersenneTwister(56), 90), g = repeat(["a", "b", "c"], 30))
    @test length(test_indices(datasplit(KernelThinningSplitter(rng = MersenneTwister(2)), df))) == 18
    @test length(test_indices(datasplit(KernelThinningSplitter(rng = MersenneTwister(3)), randn(MersenneTwister(57), 50)))) == 10
    c = compare([KernelThinningSplitter(rng = MersenneTwister(4)), HerdingSplitter(kernel = EnergyKernel())], data)
    @test DataFrame(c).method == ["KernelThinningSplitter", "HerdingSplitter"]
    @test all(isfinite, c.qualities)
  end

  @testset "weights and reference through datasplit and selectrows" begin
    data = randn(MersenneTwister(58), 300, 2)
    s = KernelThinningSplitter(rng = MersenneTwister(5))
    @test test_indices(datasplit(s, data; weights = ones(300))) ==
          test_indices(datasplit(KernelThinningSplitter(rng = MersenneTwister(5)), data))
    heavy = data[:, 1] .> 0
    plain = selectrows(KernelThinningSplitter(rng = MersenneTwister(6)), data, 60)
    weighted = selectrows(KernelThinningSplitter(rng = MersenneTwister(6)), data, 60; weights = ifelse.(heavy, 20.0, 1.0))
    targeted = selectrows(KernelThinningSplitter(rng = MersenneTwister(6)), data, 60; reference = data[heavy, :])
    @test count(heavy[weighted]) > count(heavy[plain])
    @test count(heavy[targeted]) > count(heavy[plain])
    folds = multiplet(KernelThinningSplitter(rng = MersenneTwister(7)), data, 4)
    @test sort(reduce(vcat, folds)) == 1:300
  end
end
```

Append to `test/test_properties.jl`, inside the outer testset:

```julia
  @testset "kernel-thinning splits beat random splits under energy distance and MMD" begin
    mixture = let rng = MersenneTwister(400), N = 400
      c = rand(rng, 1:4, N)
      centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
      centers[c, :] .+ randn(rng, N, 2)
    end
    for (seed, data) in ((401, mixture), (402, randn(MersenneTwister(402), 400, 4)))
      for kernel in (EnergyKernel(), GaussianKernel(1.0))
        s = KernelThinningSplitter(kernel = kernel, rng = MersenneTwister(seed))
        r = datasplit(s, data)
        q = splitquality(data, r; kernel)
        n_test = length(test_indices(r))
        random_qs = map(1:25) do i
          perm = randperm(MersenneTwister(1_000 * seed + i), size(data, 1))
          fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
          splitquality(data, fake; kernel)
        end
        @test q < mean(random_qs)
      end
    end
  end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl`
Expected: `UndefVarError: KernelThinningSplitter`.

- [ ] **Step 3: Implement**

Append to `src/kernel_thinning.jl`:

````julia
"""
    KernelThinningSplitter(; kernel = EnergyKernel(), ratio = 0.2, delta = 0.5,
                             n_threads = Threads.nthreads(), rng = Random.default_rng())

Split by generalized kernel thinning with the target kernel (Dwivedi & Mackey
2022; kernel halving from Dwivedi & Mackey 2024): the smaller side is chosen by
[`kernel_thinning`](@ref), so it minimizes the MMD² (energy distance for
`EnergyKernel`) to the data without continuous optimization or a
nearest-neighbor step, with the papers' `O(√(log n / n))` MMD guarantee for
the KT-SPLIT candidates and a KT-SWAP result never worse than a uniform
random subset. Cost is `O(N²)` kernel evaluations like `HerdingSplitter`;
near-linear time needs Compress++, which is not implemented.

- `kernel`: `EnergyKernel()` (default) or `GaussianKernel(σ)`; a `:median`
  bandwidth is resolved at `datasplit` time and stored in `result.method`.
- `ratio`: fraction of rows assigned to the test set, in (0, 1).
- `delta`: the failure probability `δ` of the kernel-thinning guarantees
  (`δ_i = δ/L` per halving step); the papers' experiments use `0.5`.
- `rng`: the input shuffle, the halving coin flips, and the baseline draw.

`SplitResult.converged` is always `true`; `iterations` is the number of
KT-SWAP replacements. `weights` and `reference` act on the KT-SWAP objective
only (see [`kernel_thinning`](@ref) for the differences from the paper).

# Examples

```julia
data = randn(MersenneTwister(1), 1_000, 5)
result = datasplit(KernelThinningSplitter(rng = MersenneTwister(2)), data)
```
"""
struct KernelThinningSplitter{K<:SplitKernel,R<:AbstractRNG} <: AbstractSplitter
  kernel::K
  ratio::Float64
  delta::Float64
  n_threads::Int
  rng::R
end

function KernelThinningSplitter(;
  kernel::SplitKernel = EnergyKernel(),
  ratio::Real = 0.2,
  delta::Real = 0.5,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  ratio = Float64(ratio)
  delta = Float64(delta)
  0 < ratio < 1 || throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
  0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1), got $delta"))
  n_threads > 0 || throw(ArgumentError("n_threads must be positive, got $n_threads"))
  return KernelThinningSplitter(kernel, ratio, delta, n_threads, rng)
end

_with_kernel(s::KernelThinningSplitter, kernel) =
  KernelThinningSplitter(kernel, s.ratio, s.delta, s.n_threads, s.rng)

function _select_rows(s::KernelThinningSplitter, kernel, X, n; weights, target, target_weights)
  rows, swaps = kernel_thinning(
    kernel,
    X,
    n;
    delta = s.delta,
    weights,
    target,
    target_weights,
    n_threads = s.n_threads,
    rng = s.rng,
  )
  return rows, true, swaps
end

function Base.show(io::IO, s::KernelThinningSplitter)
  print(io, "KernelThinningSplitter(kernel=$(s.kernel), ratio=$(s.ratio), delta=$(s.delta))")
end
````

In `src/SPlit.jl`, change the splitter export line to
`export AbstractSplitter, SupportPointSplitter, HerdingSplitter, TwinningSplitter, KernelThinningSplitter, SplitResult, datasplit`.

- [ ] **Step 4: Run the tests**

Run: `julia --project=<worktree> <worktree>/test/test_kernel_thinning.jl && julia --project=<worktree> <worktree>/test/test_properties.jl`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/kernel_thinning.jl src/SPlit.jl test/test_kernel_thinning.jl test/test_properties.jl
git commit -m "feat: Add KernelThinningSplitter through datasplit, selectrows, and compare

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Full Julia suite on 1.10 and 1.12

- [ ] `julia -t 4 --project=<worktree> -e "using Pkg; Pkg.test()"` and `julia +1.12 --project=<worktree> <worktree>/test/test_kernel_thinning.jl && julia +1.12 --project=<worktree> <worktree>/test/test_properties.jl && julia +1.12 --project=<worktree> <worktree>/test/test_herding.jl`: all pass. Kernel thinning consumes `rng` through `randperm`, `rand`, and `sample`, all `MersenneTwister` in the tests, so results are stable across Julia versions; if a 1.12 run differs, that is a real finding to report.
- [ ] Fix anything that fails, commit as `fix:` (no commit if clean).

---

### Task 5: splitiq parity

**Files:**

- Modify: `splitiq/src/splitiq/split.py`, `splitiq/src/splitiq/multiplet.py`
- Create: `splitiq/tests/test_kernel_thinning.py`
- Modify: `splitiq/docs/getting-started.md`, `splitiq/docs/overview.md`, `splitiq/README.md`

**Interfaces:**

- Consumes: Julia `KernelThinningSplitter(; kernel, ratio, delta, n_threads, rng)`.
- Produces:
  - `SplitMethod = Literal['support_points', 'herding', 'twinning', 'kernel_thinning']`; `_METHODS` gains `'kernel_thinning'`; `_DEFAULT_DELTA = 0.5`.
  - `datasplit(..., delta: float = 0.5)`, `select_rows(..., delta: float = 0.5)`, `multiplet(..., delta: float = 0.5)`.
  - `_build_splitter(jl, method, kernel, kernel_obj, ratio, kappa, max_iterations, tolerance, n_threads, rng, start, delta)` (new trailing `delta`).

- [ ] **Step 0: Dev environment**

Run: `cd <worktree>/splitiq && make setup && make julia-dev && make test`. Expected: the existing suite passes against this worktree's SPlit.

- [ ] **Step 1: Write the failing tests**

Create `splitiq/tests/test_kernel_thinning.py`:

```python
"""Parity tests for method='kernel_thinning' in splitiq."""

from __future__ import annotations

import numpy as np
import pytest

from splitiq import datasplit, energydistance, multiplet, select_rows


def _data(seed: int = 0, n: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 3))


def test_kernel_thinning_datasplit_partitions_and_reports() -> None:
    data = _data(1)
    result = datasplit(data, ratio=0.2, method='kernel_thinning', seed=1)
    assert result.method == 'kernel_thinning'
    assert result.kernel == 'energy'
    assert result.bandwidth is None
    assert result.converged and result.iterations >= 0
    assert result.selected == 'test'
    assert sorted([*result.train_indices.tolist(), *result.test_indices.tolist()]) == list(range(300))
    again = datasplit(data, ratio=0.2, method='kernel_thinning', seed=1)
    assert np.array_equal(again.test_indices, result.test_indices)
    gauss = datasplit(data, ratio=0.25, method='kernel_thinning', kernel='gaussian', seed=2)
    assert gauss.bandwidth is not None and len(gauss.test_indices) == 75


def test_kernel_thinning_beats_random_under_energy_distance() -> None:
    data = _data(2, 400)
    result = datasplit(data, ratio=0.2, method='kernel_thinning', seed=3)
    q = energydistance(data[result.test_indices], data[result.train_indices])
    rng = np.random.default_rng(4)
    random_qs = []
    for _ in range(10):
        perm = rng.permutation(400)
        random_qs.append(energydistance(data[perm[:80]], data[perm[80:]]))
    assert q < float(np.mean(random_qs))


def test_select_rows_delta_and_multiplet() -> None:
    data = _data(5)
    idx = select_rows(data, 60, method='kernel_thinning', delta=0.1, seed=6)
    assert len(set(idx.tolist())) == 60
    folds = multiplet(data, 4, method='kernel_thinning', seed=7)
    assert sorted(np.concatenate(folds).tolist()) == list(range(300))
    with pytest.raises(ValueError):
        select_rows(data, 200, method='kernel_thinning')   # more than half


def test_kernel_thinning_option_errors() -> None:
    data = _data(8)
    with pytest.raises(ValueError, match='delta'):
        datasplit(data, method='herding', delta=0.1)
    with pytest.raises(ValueError, match='delta'):
        datasplit(data, method='kernel_thinning', delta=1.5)
    with pytest.raises(ValueError, match='kappa'):
        datasplit(data, method='kernel_thinning', kappa=50)
    with pytest.raises(ValueError, match='start'):
        datasplit(data, method='kernel_thinning', start='random')
    with pytest.raises(ValueError, match='method'):
        datasplit(data, method='thinning')  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd <worktree>/splitiq && PYTHON_JULIACALL_PROJECT=$PWD/.julia_dev PYTHON_JULIACALL_EXE=$(command -v julia) uv run pytest tests/test_kernel_thinning.py -q`
Expected: failures on the unknown method / unknown `delta` keyword.

- [ ] **Step 3: Implement**

In `splitiq/src/splitiq/split.py`:

1. `SplitMethod` and `_METHODS` gain `'kernel_thinning'`; add `_DEFAULT_DELTA = 0.5`.
2. `datasplit` and `select_rows` gain `delta: float = 0.5` after `start`, documented as: "Failure probability of the kernel-thinning guarantees (``method='kernel_thinning'`` only; the papers use ``0.5``). Any other value with another method raises ``ValueError``." Their `method` docstrings add ``'kernel_thinning'`` (generalized kernel thinning, Dwivedi & Mackey 2022/2024; energy or Gaussian kernel; selects at most half of the rows). Pass `delta` to `_build_splitter`.
3. `_build_splitter` gains a trailing `delta: float` argument. Before the twinning branch:

```python
    if delta != _DEFAULT_DELTA and method != 'kernel_thinning':
        msg = "'delta' is a kernel-thinning option; use method='kernel_thinning'"
        raise ValueError(msg)
    if method == 'kernel_thinning':
        return _build_kernel_thinning_splitter(
            jl, kernel_obj, ratio, kappa, max_iterations, tolerance, n_threads, rng, delta
        )
```

and the helper:

```python
def _build_kernel_thinning_splitter(
    jl: JuliaValue,
    kernel_obj: JuliaValue,
    ratio: float,
    kappa: int | None,
    max_iterations: int,
    tolerance: float,
    n_threads: int | None,
    rng: JuliaValue | None,
    delta: float,
) -> JuliaValue:
    """Build a Julia ``KernelThinningSplitter``; it has no optimizer options.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        kernel_obj: A Julia ``SplitKernel`` value (energy or Gaussian).
        ratio: Fraction of rows assigned to the test set.
        kappa: Must be ``None``.
        max_iterations: Must be the default.
        tolerance: Must be the default.
        n_threads: Number of threads, or ``None`` to omit the keyword.
        rng: A Julia RNG value, or ``None`` to omit the keyword.
        delta: Failure probability of the kernel-thinning guarantees, in (0, 1).

    Returns:
        A Julia ``KernelThinningSplitter`` value.

    Raises:
        ValueError: If an optimizer option is set, or if Julia rejects the
            arguments (e.g. `delta` outside (0, 1)).
    """
    if kappa != _DEFAULT_KAPPA or max_iterations != _DEFAULT_MAX_ITERATIONS or tolerance != _DEFAULT_TOLERANCE:
        msg = "kernel thinning has no 'kappa'/'max_iterations'/'tolerance' options; leave them at their defaults"
        raise ValueError(msg)
    kwargs = _splitter_kwargs(kernel_obj, ratio, n_threads, rng)
    kwargs['delta'] = float(delta)
    with _translate_error():
        return jl.KernelThinningSplitter(**kwargs)
```

The `start`-with-other-method guard already covers `start` for this method. Update `_build_splitter`'s docstring and `SplitResult.method`'s docstring.

In `splitiq/src/splitiq/multiplet.py`: add `delta: float = 0.5` (documented like above) and pass it as the last positional argument to `_build_splitter`; update the `method` docstring line.

- [ ] **Step 4: Run the gates**

`make test && make lint && make typecheck && make format` — all green.

- [ ] **Step 5: Docs**

`splitiq/docs/getting-started.md` Options list, after the twinning bullet:

```markdown
- `method='kernel_thinning'` runs generalized kernel thinning (Dwivedi & Mackey 2022, 2024) with
  the energy or Gaussian kernel: kernel halving into candidate subsets, then a swap pass over the
  whole data; `delta` is the papers' failure probability (default `0.5`). It selects at most half
  of the rows.
```

`splitiq/docs/overview.md`: `method` line lists the four methods. `splitiq/README.md`: add `delta=0.5` to the signatures where `start=None` appears and mention the method next to the others.

- [ ] **Step 6: Commit**

```bash
git add splitiq/src/splitiq splitiq/tests/test_kernel_thinning.py splitiq/docs splitiq/README.md
git commit -m "feat(splitiq): Mirror KernelThinningSplitter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Benchmark and docs

**Files:**

- Modify: `benchmark/run.jl`; regenerate `docs/src/assets/benchmarks/results.md`, `quality.png`, `time.png`, `selection.png`
- Modify: `docs/src/10-methods.md`, `20-benchmarks.md`, `85-roadmap.md`, `index.md`, `30-python.md`, `AGENTS.md`, `README.md`

- [ ] **Step 1: `benchmark/run.jl`**

In `methods(N; rng_seed)`, append after the herding entries:

```julia
    (
      "kernel thinning · energy",
      KernelThinningSplitter(kernel = EnergyKernel(), rng = MersenneTwister(rng_seed)),
    ),
    (
      "kernel thinning · gaussian",
      KernelThinningSplitter(kernel = GaussianKernel(), rng = MersenneTwister(rng_seed)),
    ),
```

Plotting: `methods_order` becomes the six optimized methods in that order followed by `"random"`; `colors = Makie.wong_colors()[1:7]`; `markers = [:circle, :rect, :utriangle, :diamond, :star5, :hexagon]`; every `methods_order[1:4]` becomes `methods_order[1:6]`; the selection figure gets `size = (1800, 300)` for seven panels. Check the benchmark project resolves this worktree (`julia --project=<worktree>/benchmark -e 'using SPlit; println(pathof(SPlit))'`; if not, `Pkg.develop(path="<worktree>"); Pkg.instantiate()`; the Manifest is git-ignored). Quick run: `julia -t auto --project=<worktree>/benchmark <worktree>/benchmark/run.jl --quick` (writes N = 200 outputs; discard them with `git checkout -- docs/src/assets/benchmarks`). Then the full run on an otherwise idle machine: `julia -t auto --project=<worktree>/benchmark <worktree>/benchmark/run.jl` (about 15–25 minutes). Commit the script and the four regenerated assets:

```bash
git add benchmark/run.jl docs/src/assets/benchmarks/results.md docs/src/assets/benchmarks/quality.png docs/src/assets/benchmarks/time.png docs/src/assets/benchmarks/selection.png
git commit -m "chore: Add kernel thinning to the main benchmark

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 2: Methods page**

Append to `docs/src/10-methods.md` after the twinning section:

```markdown
## [Kernel thinning](@id kernel-thinning)

`KernelThinningSplitter` (Dwivedi & Mackey, 2022, 2024) selects rows
directly, like herding, but by a randomized halving scheme with a
non-asymptotic MMD guarantee. It replaces steps 3 and 4 of the procedure
and keeps the kernel of step 2: with the target kernel ``k`` (the energy
kernel by default), the split kernel of the papers is ``k`` itself
("target kernel thinning").

1. **Kernel halving.** Rows are visited two at a time in a random order.
   For the pair ``(x, x')`` let ``f = k(x, \cdot) - k(x', \cdot)``,
   ``b^2 = \|f\|_k^2 = k(x,x) + k(x',x') - 2k(x,x')``, and
   ``\alpha = \sum_{z \in S_2} f(z) - \sum_{z \in S_1} f(z)`` over the two
   halves built so far. One row goes to each half; the assignment is
   swapped with probability ``\min(1, \tfrac12 (1 - \alpha/a)_+)``, where
   the threshold ``a`` is updated from ``b`` and a running parameter
   ``\sigma`` so that the halves stay balanced with probability at least
   ``1 - \delta`` (`delta`, default ``0.5``).
2. **KT-SPLIT.** With ``m = \lfloor \log_2 (N/n) \rfloor``, the first
   ``L = n 2^m`` rows of the shuffle are halved ``m`` times, giving ``2^m``
   candidate subsets of size ``n``.
3. **KT-SWAP.** A uniform random subset of size ``n`` joins the
   candidates; the one with the smallest
   ``\frac{1}{n^2}\sum_{a,b \in S} k(a,b) - \frac{2}{n}\sum_{a \in S} d(a)``
   (the MMD² to the target measure up to a constant, with
   ``d(a) = \sum_l \bar v_l k(a, r_l)`` the data term of step 4 of kernel
   herding) is refined by one pass over its positions, each replaced by
   the row outside the subset that lowers the objective most, if any.

The result is never worse than the random baseline, and each KT-SPLIT
candidate carries the papers' ``O(\sqrt{\log n / n})`` MMD bound. The cost
is ``O(L^2)`` kernel evaluations for the halvings, ``O(N^2)`` for the data
term and ``O(nN)`` for the swap pass, all threaded: the same class as
herding. The near-linear variant of the papers, Compress++, applies to
``n \approx \sqrt N`` and is planned for the embedding workflow.

**Differences from the paper.** The papers thin ``N`` rows to
``\lfloor N/2^m \rfloor``; here ``n`` is set by `ratio` (or the caller) and
the ``N - L`` rows left out of KT-SPLIT still enter KT-SWAP as candidates
and through the target measure; the two agree when ``N/n`` is a power of
two. Swap candidates exclude rows already selected, so the split is a set
of distinct rows. `weights` and `reference` change the KT-SWAP objective
only (the target measure ``P_w`` or ``P_R`` of the sections above); the
halvings always run on the unweighted candidate rows. `selectrows` with
``n > N/2`` is an error, since the procedure halves.
```

Add to the References list: Dwivedi, R., & Mackey, L. (2024). Kernel Thinning. *Journal of Machine Learning Research*, 25(152), 1–77; and Dwivedi, R., & Mackey, L. (2022). Generalized Kernel Thinning. *ICLR*.

- [ ] **Step 3: Benchmarks page**

Re-derive sections 1 and 2 of `docs/src/20-benchmarks.md` from the new `results.md`: the opening bold claim (which method is best or close to best in how many of the 8 cells, and which is fastest), the bullets of section 1, the per-cell winners table (now including the kernel-thinning rows), and section 2's timing sentence (add kernel thinning's range at N = 10,000). Keep section 3, "What each method picks" (update the panel count to seven in its caption), and section 4 unchanged except that the method table under "How it was run" gains two rows `| kernel thinning · energy |`KernelThinningSplitter(EnergyKernel())` | `delta = 0.5` | `delta = 0.5`|` and the Gaussian twin. Every number in the prose must be derivable from `results.md`.

- [ ] **Step 4: Roadmap, index, Python page, README, AGENTS.md**

- `docs/src/85-roadmap.md`: Current state row `|`KernelThinningSplitter` | done | Target-kernel KT (Dwivedi & Mackey, 2022, 2024): kernel halving, KT-SPLIT, KT-SWAP; energy or Gaussian kernel; `O(N²)`like herding. |`; M4 heading text → "Done (2026-09-04): added `KernelThinningSplitter` … Compress++ is not included: it applies to `n ≈ √N` root-thinning and moves to M5." Add a bullet to M5: "Compress++ (Shetty, Dwivedi & Mackey, 2022) for `selectrows` with `n ≪ N`, on top of `kernel_thinning`." Changelog: `- 2026-09-04: M4 (kernel thinning) done; Compress++ moved to M5.`
- `docs/src/index.md`, "Kernels and splitters": a short paragraph and example for `KernelThinningSplitter`; the Benchmarks pointer says "the four splitters".
- `docs/src/30-python.md`: a "Kernel thinning" subsection with `method='kernel_thinning'` and `delta`.
- `README.md`: API bullet for `KernelThinningSplitter`; `compare` wording lists four splitters; References gain the two papers.
- `AGENTS.md` gotcha: "`KernelThinningSplitter`: KT-SPLIT runs on the first `n·2^m` rows of a shuffle (all rows when `N/n` is a power of two); `weights`/`reference` act on the KT-SWAP objective only; `delta` is the papers' δ; swap candidates exclude selected rows; `n > N ÷ 2` is an error; cost is the herding class `O(N²)`; threaded sums use fixed 1,024-row chunks so results do not depend on `n_threads`."

- [ ] **Step 5: Build the docs and commit**

`julia --project=<worktree>/docs <worktree>/docs/make.jl 2>&1 | grep -iE "error|warning: (invalid|unresolved|missing|no doc)"` prints nothing beyond the repository-URL warning (`95-reference.md` uses `@autodocs`, so the new docstrings are picked up).

```bash
git add docs/src AGENTS.md README.md
git commit -m "docs: Document kernel thinning and re-derive the benchmark story

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Quality gate

- [ ] `julia -t 4 --project=<worktree> -e "using Pkg; Pkg.test()"`, `julia -t 1 ...`, and `julia +1.12 ...`: all pass.
- [ ] Docs build clean; from `splitiq/`: `make test && make lint && make typecheck && make docs`.
- [ ] `pre-commit run --all-files` clean.
- [ ] `git diff origin/main...HEAD -- test/ | grep "^-" | grep -v "^---"` empty; `git diff origin/main...HEAD -- src/splitter.jl src/optimizer.jl src/twinning.jl src/multiplet.jl src/quality.jl src/comparison.jl` empty; `src/herding.jl` diff is only the `_target_data_term` extraction.
- [ ] Report; the PR is opened only after the user confirms.
