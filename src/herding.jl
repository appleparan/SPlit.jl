"""
Greedy kernel herding on the empirical distribution (Chen, Welling & Smola
2010, Eq. 8), restricted to rows not yet selected so the result is a subset.
Each selection is the greedy step of MMD² minimization between the selected
rows and the data (energy-distance minimization for `EnergyKernel`).
"""

using LinearAlgebra: dot
using Random

# d_i = mean over all N rows of k(x_i, x_l), for every row i (including the
# self-term k(x_i, x_i), matching Eq. (8) exactly); chunked over threads.
# Computed on the transpose so each row is a contiguous column view.
function _data_term(
  ::Exact,
  kernel::SplitKernel,
  X::Matrix{Float64},
  n_threads::Int,
  ::AbstractRNG,
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
        s += kernelvalue(kernel, xi, Xt[:, l])
      end
      d[i] = s / N
    end
  end
  return d
end

# Sliced energy-distance data term: d_i = -1/N Σ_l ‖x_i − x_l‖ estimated from
# `e.k` random one-dimensional projections. For sorted projections with
# prefix sums P_r = Σ_{l≤r} u_(l) and rank r_i, Σ_l |u_i − u_l| =
# u_i(2r_i − N) − 2P_{r_i} + P_N (see the spec's projection identity).
function _data_term(
  e::RandomSlices,
  ::EnergyKernel,
  X::Matrix{Float64},
  ::Int,
  rng::AbstractRNG,
)
  N, p = size(X)
  κ = sphere_constant(p)
  Θ = _project_directions(rng, p, e.k)
  d = zeros(N)
  rank = Vector{Int}(undef, N)
  for j = 1:e.k
    @views θ = Θ[:, j]
    u = X * θ
    order = sortperm(u)
    @inbounds for (r, i) in enumerate(order)
      rank[i] = r
    end
    P = cumsum(u[order])
    PN = P[N]
    @inbounds for i = 1:N
      r = rank[i]
      S = u[i] * (2r - N) - 2 * P[r] + PN
      d[i] -= S / (e.k * κ * N)
    end
  end
  return d
end

# Random-Fourier-features data term: d_i = mean_l k(x_i, x_l) ≈ z(x_i) · z̄
# with z̄ the mean feature vector over all N rows (same φ for every row).
function _data_term(
  e::RandomFeatures,
  k::GaussianKernel{Float64},
  X::Matrix{Float64},
  ::Int,
  rng::AbstractRNG,
)
  N, p = size(X)
  φ = FourierFeatureMap(k, p, e.D, rng)
  z̄ = _feature_mean(φ, X)
  Xt = permutedims(X)
  d = Vector{Float64}(undef, N)
  @views for i = 1:N
    d[i] = dot(φ(Xt[:, i]), z̄)
  end
  return d
end

_data_term(e, k::SplitKernel, ::Matrix{Float64}, ::Int, ::AbstractRNG) =
  _undefined(e, "the herding data term under $(nameof(typeof(k)))")

"""
    herd(kernel, X, n; estimator = Exact(), n_threads = Threads.nthreads(),
         rng = Random.default_rng()) -> Vector{Int}

Select `n` rows of `X` by kernel herding: the first row maximizes the mean
kernel value to the data, and each later row maximizes
`mean_l k(x, x_l) − (1/(T+1)) Σ_t k(x, s_t)` over rows not yet selected
(Chen, Welling & Smola 2010, Eq. 8). The data term (the `mean_l k(x, x_l)`
part) is exact by default, including the self-term; `estimator` selects an
approximation such as [`RandomSlices`](@ref) or [`RandomFeatures`](@ref) for
large `N`, drawing its randomness from `rng`. The procedure is deterministic
given the data, a numeric kernel, and (for an approximate estimator) `rng`;
ties go to the lowest row index. Cost `O(N² + nN)` for `Exact`.

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
  estimator::DiscrepancyEstimator = Exact(),
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  isresolved(kernel) ||
    throw(ArgumentError("kernel parameters must be resolved; call resolve first"))
  _supports(estimator, kernel) || throw(
    ArgumentError(
      "$(nameof(typeof(estimator))) has no herding data term for $(nameof(typeof(kernel)))",
    ),
  )
  N = size(X, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))

  d = _data_term(estimator, kernel, X, n_threads, rng)
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
    HerdingSplitter(; kernel = GaussianKernel(), estimator = Exact(), ratio = 0.2,
                      n_threads = Threads.nthreads(), rng = Random.default_rng())

Split by greedy kernel herding (Chen, Welling & Smola 2010): the smaller
subset is chosen row by row to minimize the MMD² (energy distance for
`EnergyKernel`) to the whole data. `estimator` selects how the herding data
term is computed — [`Exact`](@ref) by default, or an approximation such as
[`RandomSlices`](@ref)/[`RandomFeatures`](@ref) for large data; `rng` feeds
both a `:median` bandwidth and the estimator's randomness. See [`herd`](@ref)
for the rule and cost.

# Examples

```julia
data = randn(MersenneTwister(1), 200, 3)
result = datasplit(HerdingSplitter(rng = MersenneTwister(2)), data)
```
"""
struct HerdingSplitter{K<:SplitKernel,E<:DiscrepancyEstimator,R<:AbstractRNG} <:
       AbstractSplitter
  kernel::K
  estimator::E
  ratio::Float64
  n_threads::Int
  rng::R
end

function HerdingSplitter(;
  kernel::SplitKernel = GaussianKernel(),
  estimator::DiscrepancyEstimator = Exact(),
  ratio::Real = 0.2,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
)
  ratio = Float64(ratio)
  0 < ratio < 1 || throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
  n_threads > 0 || throw(ArgumentError("n_threads must be positive, got $n_threads"))
  _supports(estimator, kernel) || throw(
    ArgumentError(
      "$(nameof(typeof(estimator))) has no herding data term for $(nameof(typeof(kernel)))",
    ),
  )
  return HerdingSplitter(kernel, estimator, ratio, n_threads, rng)
end

function datasplit(s::HerdingSplitter, data)
  X = preprocess(data)
  n_total = size(X, 1)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))
  kernel = resolve(s.kernel, X, s.rng)
  fitted = HerdingSplitter(kernel, s.estimator, s.ratio, s.n_threads, s.rng)
  small =
    herd(kernel, X, n_small; estimator = s.estimator, n_threads = s.n_threads, rng = s.rng)
  rest = setdiff(1:n_total, small)
  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  return SplitResult(collect(train), collect(test), true, n_small, fitted)
end

function Base.show(io::IO, s::HerdingSplitter)
  suffix = s.estimator isa Exact ? "" : ", estimator=$(s.estimator)"
  print(io, "HerdingSplitter(kernel=$(s.kernel), ratio=$(s.ratio)$suffix)")
end
