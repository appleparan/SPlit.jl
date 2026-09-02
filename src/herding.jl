"""
Greedy kernel herding on the empirical distribution (Chen, Welling & Smola
2010, Eq. 8), restricted to rows not yet selected so the result is a subset.
Each selection is the greedy step of MMD² minimization between the selected
rows and the data (energy-distance minimization for `EnergyKernel`).
"""

using Random

# d_i = mean over all N rows of k(x_i, x_l), for every row i (including the
# self-term k(x_i, x_i), matching Eq. (8) exactly); chunked over threads.
# Computed on the transpose so each row is a contiguous column view.
function _data_term(kernel::SplitKernel, X::Matrix{Float64}, n_threads::Int)
  N = size(X, 1)
  Xt = permutedims(X)
  d = Vector{Float64}(undef, N)
  chunks = collect(Iterators.partition(1:N, cld(N, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for i in chunk
      s = 0.0
      @views xi = Xt[:, i]
      @views for l = 1:N
        s += kernelvalue(kernel, xi, Xt[:, l])
      end
      d[i] = s / N
    end
  end
  return d
end

"""
    herd(kernel, X, n; n_threads = Threads.nthreads()) -> Vector{Int}

Select `n` rows of `X` by kernel herding: the first row maximizes the mean
kernel value to the data, and each later row maximizes
`mean_l k(x, x_l) − (1/(T+1)) Σ_t k(x, s_t)` over rows not yet selected
(Chen, Welling & Smola 2010, Eq. 8). The data term is the exact mean over all
`N` rows, including the self-term, computed threaded (`O(N²)`); approximating
it with a `DiscrepancyEstimator` was evaluated and rejected — see the Methods
page for why the noise it introduces is correlated across candidate rows and
makes greedy selection unreliable. The procedure is deterministic given the
data and a numeric kernel; ties go to the lowest row index. Cost
`O(N² + nN)`.

# Examples

```julia
X = randn(MersenneTwister(1), 200, 3)
herd(GaussianKernel(1.0), X, 40)
```
"""
function herd(
  kernel::SplitKernel,
  X::Matrix{Float64},
  n::Int;
  n_threads::Int = Threads.nthreads(),
)
  isresolved(kernel) ||
    throw(ArgumentError("kernel parameters must be resolved; call resolve first"))
  N = size(X, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))

  d = _data_term(kernel, X, n_threads)
  c = zeros(N)
  used = falses(N)
  selected = Vector{Int}(undef, n)
  for T = 0:(n-1)
    best = 0
    bestscore = -Inf
    for i = 1:N
      used[i] && continue
      score = d[i] - c[i] / (T + 1)
      if score > bestscore
        bestscore = score
        best = i
      end
    end
    selected[T+1] = best
    used[best] = true
    @views for i = 1:N
      c[i] += kernelvalue(kernel, X[i, :], X[best, :])
    end
  end
  return selected
end

"""
    HerdingSplitter(; kernel = GaussianKernel(), ratio = 0.2,
                      n_threads = Threads.nthreads(), rng = Random.default_rng())

Split by greedy kernel herding (Chen, Welling & Smola 2010): the smaller
subset is chosen row by row to minimize the MMD² (energy distance for
`EnergyKernel`) to the whole data. The data term is always computed exactly
(threaded, `O(N²)`); see [`herd`](@ref) for why an approximate data term is
not offered. Deterministic given the data and a numeric kernel; `rng` only
feeds a `:median` bandwidth. See [`herd`](@ref) for the rule and cost.

# Examples

```julia
data = randn(MersenneTwister(1), 200, 3)
result = datasplit(HerdingSplitter(rng = MersenneTwister(2)), data)
```
"""
struct HerdingSplitter{K<:SplitKernel,R<:AbstractRNG} <: AbstractSplitter
  kernel::K
  ratio::Float64
  n_threads::Int
  rng::R
end

function HerdingSplitter(;
  kernel::SplitKernel = GaussianKernel(),
  ratio::Real = 0.2,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  ratio = Float64(ratio)
  0 < ratio < 1 || throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
  n_threads > 0 || throw(ArgumentError("n_threads must be positive, got $n_threads"))
  return HerdingSplitter(kernel, ratio, n_threads, rng)
end

function datasplit(s::HerdingSplitter, data)
  X = preprocess(data)
  n_total = size(X, 1)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))
  kernel = resolve(s.kernel, X, s.rng)
  fitted = HerdingSplitter(kernel, s.ratio, s.n_threads, s.rng)
  small = herd(kernel, X, n_small; n_threads = s.n_threads)
  rest = setdiff(1:n_total, small)
  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  return SplitResult(collect(train), collect(test), true, n_small, fitted)
end

function Base.show(io::IO, s::HerdingSplitter)
  print(io, "HerdingSplitter(kernel=$(s.kernel), ratio=$(s.ratio))")
end
