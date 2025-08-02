# Energy distances between two samples
using Distances
using Distances: Euclidean, Cityblock, PreMetric
using LinearAlgebra
using Statistics: mean
using Random

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
struct EnergyDistance{T<:PreMetric} <: PreMetric
  metric::T
end

"""
    compute_pairwise_distances(metric, X, Y)

Compute pairwise distances between vectors in X and Y using the given metric.
"""
function compute_pairwise_distances(metric, X::AbstractMatrix, Y::AbstractMatrix)
  n_x, n_y = size(X, 2), size(Y, 2)
  distances = Matrix{Float64}(undef, n_x, n_y)

  for i = 1:n_x
    for j = 1:n_y
      distances[i, j] = metric(X[:, i], Y[:, j])
    end
  end

  return distances
end

"""
    compute_pairwise_distances(metric, X)

Compute pairwise distances within vectors in X using the given metric.
"""
function compute_pairwise_distances(metric, X::AbstractMatrix)
  n = size(X, 2)
  distances = Matrix{Float64}(undef, n, n)

  for i = 1:n
    distances[i, i] = 0.0  # Distance to self is zero
    for j = (i+1):n
      d = metric(X[:, i], X[:, j])
      distances[i, j] = d
      distances[j, i] = d  # Symmetric
    end
  end

  return distances
end

function (dist::EnergyDistance)(X::AbstractMatrix, Y::AbstractMatrix)
  # X and Y should be matrices where each column is a sample
  mean_XY = mean(compute_pairwise_distances(dist.metric, X, Y))
  mean_XX = mean(compute_pairwise_distances(dist.metric, X))
  mean_YY = mean(compute_pairwise_distances(dist.metric, Y))
  return 2 * mean_XY - mean_XX - mean_YY
end

function (dist::EnergyDistance)(X::AbstractVector, Y::AbstractVector)
  # For 1D vectors, we need to compute distances between scalar values
  # Convert to column vectors (each element as a separate 1D observation)
  X_mat = reshape(X, 1, length(X))  # 1×n matrix
  Y_mat = reshape(Y, 1, length(Y))  # 1×m matrix
  return dist(X_mat, Y_mat)
end

"""
    sample_without_replacement(X, n)

Sample n elements from X without replacement.
"""
function sample_without_replacement(X::AbstractVector, n::Int)
  if n > length(X)
    throw(ArgumentError("Cannot sample $n elements from vector of length $(length(X))"))
  end
  indices = randperm(length(X))[1:n]
  return X[indices]
end

function (dist::EnergyDistance)(P::AbstractVector, Q::AbstractVector, num_samples::Int)
  # Sample from the vectors
  X_sampled = sample_without_replacement(P, min(num_samples, length(P)))
  Y_sampled = sample_without_replacement(Q, min(num_samples, length(Q)))

  return dist(X_sampled, Y_sampled)
end

"""
    energy_distance(X::AbstractMatrix, Y::AbstractMatrix; metric=Euclidean())

Compute energy distance between two samples X and Y.

# Arguments
- `X`: First sample as a matrix (each column is an observation)
- `Y`: Second sample as a matrix (each column is an observation)
- `metric`: Distance metric to use (default: Euclidean())

# Returns
- Energy distance between the two samples
"""
function energy_distance(X::AbstractMatrix, Y::AbstractMatrix; metric = Euclidean())
  dist = EnergyDistance(metric)
  return dist(X, Y)
end

"""
    energy_distance(X::AbstractVector, Y::AbstractVector; metric=Euclidean())

Compute energy distance between two 1D samples X and Y.
"""
function energy_distance(X::AbstractVector, Y::AbstractVector; metric = Euclidean())
  dist = EnergyDistance(metric)
  return dist(X, Y)
end
