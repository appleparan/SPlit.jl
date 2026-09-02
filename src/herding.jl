"""
Greedy kernel herding on the empirical distribution (Chen, Welling & Smola
2010, Eq. 8), restricted to rows not yet selected so the result is a subset.
Each selection is the greedy step of MMD² minimization between the selected
rows and the data (energy-distance minimization for `EnergyKernel`).
"""

using Random
using StatsBase: sample

# d_i = mean over `rows` of k(x_i, x_l), for every row i; chunked over threads.
function _data_term(kernel::SplitKernel, X::Matrix{Float64}, rows, n_threads::Int)
  N = size(X, 1)
  d = Vector{Float64}(undef, N)
  chunks = collect(Iterators.partition(1:N, cld(N, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for i in chunk
      s = 0.0
      @views for l in rows
        s += kernelvalue(kernel, X[i, :], X[l, :])
      end
      d[i] = s / length(rows)
    end
  end
  return d
end

"""
    herd(kernel, X, n; kappa = nothing, n_threads = Threads.nthreads(),
         rng = Random.default_rng()) -> Vector{Int}

Select `n` rows of `X` by kernel herding: the first row maximizes the mean
kernel value to the data, and each later row maximizes
`mean_l k(x, x_l) − (1/(T+1)) Σ_t k(x, s_t)` over rows not yet selected
(Chen, Welling & Smola 2010, Eq. 8). With `kappa`, the data mean is estimated
from `kappa` rows drawn with `rng`; otherwise the procedure is deterministic
and `rng` is unused. Ties go to the lowest row index. Cost `O(N·|rows| + nN)`.
"""
function herd(
  kernel::SplitKernel,
  X::Matrix{Float64},
  n::Int;
  kappa::Union{Nothing,Int} = nothing,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  isresolved(kernel) ||
    throw(ArgumentError("kernel parameters must be resolved; call resolve first"))
  N = size(X, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  kappa === nothing ||
    kappa > 0 ||
    throw(ArgumentError("kappa must be positive, got $kappa"))

  rows =
    (kappa === nothing || kappa >= N) ? (1:N) :
    sort!(sample(rng, 1:N, kappa; replace = false))
  d = _data_term(kernel, X, rows, n_threads)
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
    HerdingSplitter(; kernel = GaussianKernel(), ratio = 0.2, kappa = nothing,
                      n_threads = Threads.nthreads(), rng = Random.default_rng())

Split by greedy kernel herding (Chen, Welling & Smola 2010): the smaller
subset is chosen row by row to minimize the MMD² (energy distance for
`EnergyKernel`) to the whole data. Deterministic given the data and a numeric
kernel; `rng` only drives the `kappa` subsample and a `:median` bandwidth.
See [`herd`](@ref) for the rule and cost.
"""
struct HerdingSplitter{K<:SplitKernel,R<:AbstractRNG} <: AbstractSplitter
  kernel::K
  ratio::Float64
  kappa::Union{Nothing,Int}
  n_threads::Int
  rng::R
end

function HerdingSplitter(;
  kernel::SplitKernel = GaussianKernel(),
  ratio::Real = 0.2,
  kappa::Union{Nothing,Int} = nothing,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  ratio = Float64(ratio)
  0 < ratio < 1 || throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
  kappa === nothing ||
    kappa > 0 ||
    throw(ArgumentError("kappa must be positive, got $kappa"))
  n_threads > 0 || throw(ArgumentError("n_threads must be positive, got $n_threads"))
  return HerdingSplitter(kernel, ratio, kappa, n_threads, rng)
end

function datasplit(s::HerdingSplitter, data)
  X = preprocess(data)
  n_total = size(X, 1)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))
  kernel = resolve(s.kernel, X, s.rng)
  fitted = HerdingSplitter(kernel, s.ratio, s.kappa, s.n_threads, s.rng)
  small = herd(kernel, X, n_small; kappa = s.kappa, n_threads = s.n_threads, rng = s.rng)
  rest = setdiff(1:n_total, small)
  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  return SplitResult(collect(train), collect(test), true, n_small, fitted)
end

function Base.show(io::IO, s::HerdingSplitter)
  print(io, "HerdingSplitter(kernel=$(s.kernel), ratio=$(s.ratio))")
end
