"""
Kernel types selecting the discrepancy that support points minimize.
"""

"""
    SplitKernel

Abstract supertype of kernels usable with [`SupportPointSplitter`](@ref).
Each kernel defines the discrepancy between the support-point set and the
data distribution that the optimizer minimizes.
"""
abstract type SplitKernel end

"""
    EnergyKernel()

The kernel `k(x, y) = −‖x − y‖`, whose maximum mean discrepancy between two
samples is the energy distance. Support points under this kernel are those of
Mak & Joseph (2018), optimized by their closed-form
majorization–minimization update.
"""
struct EnergyKernel <: SplitKernel end

using Distances: Euclidean, pairwise
using Random
using Statistics: median
using StatsBase: sample

"""
    GaussianKernel(bandwidth = :median)

The Gaussian (RBF) kernel `k(u, v) = exp(−‖u − v‖² / (2σ²))`. Support points
under this kernel minimize the squared maximum mean discrepancy (MMD²,
Gretton et al. 2012) by projected gradient descent with Armijo backtracking.

`bandwidth` is either a positive number `σ` or `:median`, in which case `σ` is
set at fit time to the median pairwise Euclidean distance of (a sample of)
the standardized data. The resolved value is stored in `result.method.kernel`.
"""
struct GaussianKernel{B<:Union{Float64,Symbol}} <: SplitKernel
  bandwidth::B
  function GaussianKernel(bandwidth::Real)
    isfinite(bandwidth) && bandwidth > 0 ||
      throw(ArgumentError("bandwidth must be a positive finite number, got $bandwidth"))
    return new{Float64}(Float64(bandwidth))
  end
  function GaussianKernel(bandwidth::Symbol)
    bandwidth === :median || throw(
      ArgumentError("bandwidth must be a positive number or :median, got :$bandwidth"),
    )
    return new{Symbol}(bandwidth)
  end
end
GaussianKernel() = GaussianKernel(:median)

"""
    kernelvalue(kernel, u, v) -> Float64

Evaluate the kernel at two points (vectors of equal length).
"""
function kernelvalue(k::GaussianKernel{Float64}, u::AbstractVector, v::AbstractVector)
  s = 0.0
  @inbounds for j in eachindex(u, v)
    s += (u[j] - v[j])^2
  end
  return exp(-s / (2 * k.bandwidth^2))
end

"""
    kernelgrad!(g, kernel, u, v) -> g

Write the gradient of `kernelvalue(kernel, u, v)` with respect to `u` into `g`.
"""
function kernelgrad!(
  g::AbstractVector,
  k::GaussianKernel{Float64},
  u::AbstractVector,
  v::AbstractVector,
)
  kv = kernelvalue(k, u, v)
  c = -kv / k.bandwidth^2
  @inbounds for j in eachindex(u, v)
    g[j] = c * (u[j] - v[j])
  end
  return g
end

"""
    isresolved(kernel) -> Bool

Whether every kernel parameter is numeric (no data-dependent placeholders).
"""
isresolved(::EnergyKernel) = true
isresolved(::GaussianKernel{Float64}) = true
isresolved(::GaussianKernel{Symbol}) = false

# Rows sampled for the median heuristic; all rows are used below this count.
const MEDIAN_HEURISTIC_ROWS = 1_000

"""
    resolve(kernel, data, rng) -> kernel with numeric parameters

Replace data-dependent placeholders. For `GaussianKernel(:median)` the
bandwidth becomes the median pairwise Euclidean distance over
`min(size(data, 1), 1_000)` rows drawn with `rng` (Gretton et al. 2012,
"median heuristic"). Numeric kernels are returned unchanged.
"""
resolve(k::EnergyKernel, ::AbstractMatrix, ::AbstractRNG) = k
resolve(k::GaussianKernel{Float64}, ::AbstractMatrix, ::AbstractRNG) = k
function resolve(::GaussianKernel{Symbol}, data::AbstractMatrix, rng::AbstractRNG)
  N = size(data, 1)
  m = min(N, MEDIAN_HEURISTIC_ROWS)
  rows = m == N ? (1:N) : sample(rng, 1:N, m; replace = false)
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
