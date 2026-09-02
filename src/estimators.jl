"""
Discrepancy estimators: how `energydistance`, `mmd`, `splitquality`, and
`HerdingSplitter` evaluate (or approximate) pairwise-kernel quantities.
Which estimator/kernel combinations exist is expressed by method dispatch.
"""

using LinearAlgebra: norm
using Random

"""
    DiscrepancyEstimator

Supertype of estimators selected by the `estimator` keyword: [`Exact`](@ref),
[`Subsample`](@ref), [`RandomSlices`](@ref), [`RandomFeatures`](@ref).
"""
abstract type DiscrepancyEstimator end

"""
    Exact()

Evaluate every pairwise term (block-wise, threaded; `O(nm)` kernel evaluations).
"""
struct Exact <: DiscrepancyEstimator end

"""
    Subsample(m, repeats = 8)

Average the exact statistic over `repeats` random size-`m` row subsets drawn
with `rng`. Carries a positive bias of order `1/m` (the zero diagonal of the
V-statistic); meant for comparing splits, not as an absolute value.
"""
struct Subsample <: DiscrepancyEstimator
  m::Int
  repeats::Int
  function Subsample(m::Integer, repeats::Integer = 8)
    m >= 2 || throw(ArgumentError("m must be at least 2, got $m"))
    repeats >= 1 || throw(ArgumentError("repeats must be positive, got $repeats"))
    return new(m, repeats)
  end
end

"""
    RandomSlices(k)

Estimate an energy distance from `k` random one-dimensional projections
(`rng`), each evaluated exactly by sorting: unbiased, `O(k (n + m) log(n + m))`.
Defined for the energy kernel only. See the Methods page for the identity
`E_θ |⟨θ, u⟩| = κ_p ‖u‖` behind it.
"""
struct RandomSlices <: DiscrepancyEstimator
  k::Int
  function RandomSlices(k::Integer)
    k >= 1 || throw(ArgumentError("k must be positive, got $k"))
    return new(k)
  end
end

"""
    RandomFeatures(D)

Estimate a Gaussian-kernel quantity with `D` random Fourier features
(Rahimi & Recht 2007) drawn with `rng`: unbiased, `O((n + m) D p)`. Defined for
`GaussianKernel` only.
"""
struct RandomFeatures <: DiscrepancyEstimator
  D::Int
  function RandomFeatures(D::Integer)
    D >= 1 || throw(ArgumentError("D must be positive, got $D"))
    return new(D)
  end
end

_undefined(e, what) =
  throw(ArgumentError("$(nameof(typeof(e))) is not defined for $(what)"))

"""
    sphere_constant(p) -> Float64

`κ_p = E_θ |⟨θ, e₁⟩|` for `θ` uniform on the unit sphere of `ℝ^p`:
`κ_p = Γ(p/2) / (√π Γ((p+1)/2))`, computed by the recursion
`κ_1 = 1`, `κ_2 = 2/π`, `κ_{p+2} = κ_p · p / (p + 1)`.
"""
function sphere_constant(p::Integer)
  p >= 1 || throw(ArgumentError("p must be positive, got $p"))
  κ = isodd(p) ? 1.0 : 2 / π
  q = isodd(p) ? 1 : 2
  while q < p
    κ *= q / (q + 1)
    q += 2
  end
  return κ
end

# p×k matrix of unit-norm directions (columns), drawn as normalized Gaussians.
function _project_directions(rng::AbstractRNG, p::Int, k::Int)
  Θ = randn(rng, p, k)
  for j = 1:k
    @views Θ[:, j] ./= norm(Θ[:, j])
  end
  return Θ
end

# Mean |a_i − a_j| over all ordered pairs of one sample (V-statistic, zero
# diagonal included), from the sorted sample:
# Σ_{i<j}(a_(j) − a_(i)) = Σ_i (2i − n − 1) a_(i).
function _within_mean_abs(sorted::AbstractVector{<:Real})
  n = length(sorted)
  s = 0.0
  @inbounds for i = 1:n
    s += (2i - n - 1) * sorted[i]
  end
  return 2s / n^2
end

# Mean |a_i − b_j| over all pairs, both inputs sorted, via prefix sums of `a`.
function _cross_mean_abs(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
  n, m = length(a), length(b)
  P = cumsum(a)
  total = 0.0
  @inbounds for y in b
    r = searchsortedlast(a, y)
    left = r == 0 ? 0.0 : P[r]
    total += (r * y - left) + ((P[n] - left) - (n - r) * y)
  end
  return total / (n * m)
end

# Exact one-dimensional energy distance of two samples (unsorted inputs).
function _ed1d(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
  sa = sort(collect(Float64, a))
  sb = sort(collect(Float64, b))
  return 2 * _cross_mean_abs(sa, sb) - _within_mean_abs(sa) - _within_mean_abs(sb)
end

# Sliced energy distance: κ_p^{-1} · mean over k directions of ED_1(Xθ, Yθ).
function _sliced_energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  k::Int,
  rng::AbstractRNG,
)
  p = size(X, 2)
  Θ = _project_directions(rng, p, k)
  U = X * Θ
  V = Y * Θ
  total = 0.0
  for j = 1:k
    @views total += _ed1d(U[:, j], V[:, j])
  end
  return total / (k * sphere_constant(p))
end

"""
    FourierFeatureMap(kernel::GaussianKernel{Float64}, p, D, rng)

Random Fourier features (Rahimi & Recht 2007) for a `p`-dimensional Gaussian
kernel: `z(x) = √(2/D) cos(Wx + b)` with `W ~ N(0, σ⁻² I)` and
`b ~ U[0, 2π]`, so that `E[z(x)ᵀz(y)] = kernelvalue(kernel, x, y)`. Drawn
once per call from `rng`; callable on a column vector.
"""
struct FourierFeatureMap
  W::Matrix{Float64}
  b::Vector{Float64}
  scale::Float64
end

function FourierFeatureMap(k::GaussianKernel{Float64}, p::Int, D::Int, rng::AbstractRNG)
  W = randn(rng, D, p) ./ k.bandwidth
  b = 2π .* rand(rng, D)
  return FourierFeatureMap(W, b, sqrt(2 / D))
end

(φ::FourierFeatureMap)(x::AbstractVector) = φ.scale .* cos.(φ.W * x .+ φ.b)

# Mean feature vector over the rows of X, block-wise (never materializes an
# N×D matrix for the whole input).
function _feature_mean(φ::FourierFeatureMap, X::AbstractMatrix; block::Int = 4_096)
  D = length(φ.b)
  n = size(X, 1)
  acc = zeros(D)
  for i0 = 1:block:n
    i1 = min(i0 + block - 1, n)
    @views Z = cos.(X[i0:i1, :] * φ.W' .+ φ.b')      # (rows × D)
    acc .+= vec(sum(Z; dims = 1))
  end
  return (φ.scale / n) .* acc
end

# Unbiased random-Fourier-features estimate of squared Gaussian MMD:
# ‖z̄_X − z̄_Y‖².
function _rff_mmd(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix,
  D::Int,
  rng::AbstractRNG,
)
  φ = FourierFeatureMap(k, size(X, 2), D, rng)
  return sum(abs2, _feature_mean(φ, X) .- _feature_mean(φ, Y))
end
