# Energy distances between two samples
using Distances

using LinearAlgebra
using Statistics: mean
import StatsAPI: pairwise
import StatsBase: sample

"""
    EnergyDistance(metric::T)

Create an Energy distance with metric `metric`.

```math
d(P, Q) = 2 \\mathbb{E}_{X, Y \\sim P, Q} [d(X, Y)] - \\mathbb{E}_{X, X' \\sim P} [d(X, X')] - \\mathbb{E}_{Y, Y' \\sim Q} [d(Y, Y')]
```

# Arguments
- `metric::T`: A metric for distance calculation. It must be a subtype of `PreMetric`.

# Example:
```julia
julia> dist = EnergyDistance(Euclidean())
```
"""
struct EnergyDistance{T<:Prematric} <: PreMetric
  metric::T
end

# Constructor
EnergyDistance(metric::T) where {T<:PreMetric} = EnergyDistance{T}(metric)

function (dist::EnergyDistance)(X::AbstractVector, Y::AbstractVector)
  mean_XY = mean(pairwise(dist.metric, X, Y))
  mean_XX = mean(pairwise(dist.metric, X, X))
  mean_YY = mean(pairwise(dist.metric, Y, Y))
  return 2 * mean_XY - mean_XX - mean_YY
end

function (dist::EnergyDistance)(P::Distribution, Q::Distribution, num_samples::Int)
  # Sample from the distributions
  X = rand(P, num_samples)
  Y = rand(Q, num_samples)

  return dist(X, Y)
end

function (dist::EnergyDistance)(X::AbstractVector, Y::AbstractVector, num_samples::Int)
  # Sample from the distributions
  X_sampled = sample(X, num_samples)
  Y_sampled = sample(Y, num_samples)

  return dist(X_sampled, Y_sampled)
end

function (dist::EnergyDistance)(P::Distribution, Y::AbstractVector, num_samples::Int)
  # Sample from the distributions
  P_sampled = rand(P, num_samples)
  Y_sampled = sample(Y, num_samples)

  return dist(P_sampled, Y_sampled)
end

function (dist::EnergyDistance)(X::AbstractVector, Q::Distribution, num_samples::Int)
  # Sample from the distributions
  X_sampled = sample(X, num_samples)
  Q_sampled = rand(Q, num_samples)

  return dist(X_sampled, Q_sampled)
end

function result_type(d::EnergyDistance, ::Type{T1}, ::Type{T2}) where {T1,T2}
  return typeof(dist(zero(T1), zero(T2)))
end

function _colwise!(dist, r, a, b)
  Q = dist.qmat
  get_colwise_dims(size(Q, 1), r, a, b)
  z = a .- b
  dot_percol!(r, Q * z, z)
end
