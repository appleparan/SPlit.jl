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
  a = max(b * σ * sqrt(2 * log(2 / δ)), b^2)
  a == 0.0 && return 0.0, σ
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
    b2 =
      kernelvalue(kernel, xv, xv) + kernelvalue(kernel, xv′, xv′) -
      2kernelvalue(kernel, xv, xv′)
    a, σ = _swap_params(σ, sqrt(max(b2, 0.0)), δ_step)
    if a > 0
      α =
        _kernel_diff_sum(kernel, Xt, S2, xv, xv′, n_threads) -
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
function _self_kernel_sum(
  kernel::SplitKernel,
  Xt::Matrix{Float64},
  S::Vector{Int},
  n_threads::Int,
)
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
function _coreset_sums(
  kernel::SplitKernel,
  Xt::Matrix{Float64},
  S::Vector{Int},
  N::Int,
  n_threads::Int,
)
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
        Δ =
          (kdiag[z] + 2 * (c[z] - kernelvalue(kernel, Xt[:, z], xs))) / n^2 - 2 * d[z] / n + base
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

Select `n` rows of `X` by generalized kernel thinning with the target kernel
(Dwivedi & Mackey 2022). For `n ≤ N/2`: with `m = ⌊log₂(N/n)⌋`, the first
`L = n·2^m` rows of a random permutation are split by `m` rounds of kernel
halving into `2^m` candidate subsets of size `n` (KT-SPLIT), and KT-SWAP
keeps the candidate (or a uniform random baseline) with the smallest MMD²
to the target measure and refines it by one pass of best single-row swaps
over all `N` rows. For `n > N/2`, the result is the complement of a
kernel-thinning selection of the `N - n` rows not chosen (same `rng` order,
`delta`, and target measure; see "Differences from the paper" for why this
is a reasonable rule). `delta` is the failure probability `δ` of the
kernel-thinning guarantees: the papers' `δ_i = δ/L`, applied as `δ_i/m` at
every halving step; `weights`, `target`, `target_weights` define the target
measure as in [`herd`](@ref) and act on KT-SWAP only. Cost: `O(L²)` kernel
evaluations for KT-SPLIT, `O(N²)` for the data term, `O(nN)` for KT-SWAP,
all threaded. Deterministic given `rng` and independent of `n_threads`.

# Differences from the paper

Target-kernel thinning is used (no square-root kernel). The papers thin `N`
to `⌊N/2^m⌋`; here `n` is given and only `L = n·2^m` shuffled rows enter
KT-SPLIT, the rest take part through KT-SWAP (equal to the paper when
`N/n = 2^m`). Swap candidates exclude rows already in the coreset so the
result is a set of distinct rows. `weights`/`target` change only the
KT-SWAP objective. For `n > N/2` the two halves of kernel halving are
symmetric: for a subset `S` of size `n_c = N - n` with complement `C`, the
mean-embedding identity `μ_N = (n_c/N) μ_S + ((N - n_c)/N) μ_C` gives
`MMD(C, P_N) = (n_c/(N - n_c)) · MMD(S, P_N) ≤ MMD(S, P_N)` when
`n_c ≤ N/2`, so the complement is at least as close to the data as the
thinned set; for a weighted or reference target this identity does not
hold exactly, and the rule is applied the same way regardless. Compress++
is not implemented.
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
  0 < n < N || throw(ArgumentError("n must be in 1:$(N - 1), got $n"))
  0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1), got $delta"))
  if n > N ÷ 2
    rows_c, swaps = kernel_thinning(
      kernel,
      X,
      N - n;
      delta,
      weights,
      target,
      target_weights,
      n_threads,
      rng,
    )
    return setdiff(1:N, rows_c), swaps
  end
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
  bounds = floor.(Int, (0:4) .* (ℓ / 4))
  return [seq[(bounds[i]+1):bounds[i+1]] for i = 1:4]
end

# Number of HALVE calls Compress makes on an input of length ℓ.
function _compress_halvings(ℓ::Int, g::Int)
  ℓ <= 4^g && return 0
  bounds = floor.(Int, (0:4) .* (ℓ / 4))
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
  merged = reduce(
    vcat,
    (_compress(kernel, X, part, g, δ_halve, rng; n_threads) for part in _four_parts(seq)),
  )
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
      local_rows, swaps =
        kernel_thinning(kernel, X[S_C, :], n; delta = delta / 2, n_threads, rng)
      return S_C[local_rows], swaps
    end
    g += 1
  end
end

"""
    KernelThinningSplitter(; kernel = EnergyKernel(), ratio = 0.2, delta = 0.5,
                             n_threads = Threads.nthreads(), rng = Random.default_rng())

Split by generalized kernel thinning with the target kernel (Dwivedi & Mackey
2022; kernel halving from Dwivedi & Mackey 2024): the smaller side is chosen by
[`kernel_thinning`](@ref), so it minimizes the MMD² (energy distance for
`EnergyKernel`) to the data without continuous optimization or a
nearest-neighbor step, with the papers' high-probability MMD guarantee of
order `√(log n / n)` for the KT-SPLIT candidates and a KT-SWAP result never worse than a uniform
random subset. Cost is `O(N²)` kernel evaluations like `HerdingSplitter`;
near-linear time needs Compress++, which is not implemented.

- `kernel`: `EnergyKernel()` (default) or `GaussianKernel(σ)`; a `:median`
  bandwidth is resolved at `datasplit` time and stored in `result.method`.
- `ratio`: fraction of rows assigned to the test set, in (0, 1); above one
  half the selection is the complement of a kernel-thinning selection of
  the other side (see [`kernel_thinning`](@ref)).
- `delta`: the failure probability `δ` of the kernel-thinning guarantees:
  the papers' `δ_i = δ/L`, applied as `δ_i/m` at every halving step (the
  experiments use `δ = 0.5`).
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

function _select_rows(
  s::KernelThinningSplitter,
  kernel,
  X,
  n;
  weights,
  target,
  target_weights,
)
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
  print(
    io,
    "KernelThinningSplitter(kernel=$(s.kernel), ratio=$(s.ratio), delta=$(s.delta))",
  )
end
