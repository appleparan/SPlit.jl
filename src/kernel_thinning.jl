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
