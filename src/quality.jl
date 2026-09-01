"""
Split-quality diagnostics based on the energy distance
(Székely & Rizzo; the objective minimized by support points,
Mak & Joseph 2018).
"""

using Distances: Euclidean, pairwise
using Random
using Statistics
using StatsBase: sample

# Mean pairwise Euclidean distance between rows of X and rows of Y,
# accumulated block-wise so no n×n matrix is ever materialized.
function _mean_pairwise(X::AbstractMatrix, Y::AbstractMatrix; block::Int = 1_024)
  total = 0.0
  for i0 = 1:block:size(X, 1)
    i1 = min(i0 + block - 1, size(X, 1))
    for j0 = 1:block:size(Y, 1)
      j1 = min(j0 + block - 1, size(Y, 1))
      @views total += sum(pairwise(Euclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1))
    end
  end
  return total / (size(X, 1) * size(Y, 1))
end

function _exact_energydistance(X::AbstractMatrix, Y::AbstractMatrix; block::Int = 1_024)
  return 2 * _mean_pairwise(X, Y; block) - _mean_pairwise(X, X; block) -
         _mean_pairwise(Y, Y; block)
end

"""
    energydistance(X, Y; subsample = nothing, repeats = 8, rng = Random.default_rng())

Energy distance between two samples whose rows are observations.

With `subsample = nothing` (default) the exact value is computed with
block-wise accumulation (O(1) extra memory). With `subsample = m`, the value
is estimated as the mean of `repeats` energy distances between random
size-`m` row subsets — use this when the exact O(n²) time is prohibitive.

Vectors are treated as single-column samples.
"""
function energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix;
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  rng::AbstractRNG = Random.default_rng(),
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))

  if subsample === nothing || (size(X, 1) <= subsample && size(Y, 1) <= subsample)
    return _exact_energydistance(X, Y)
  end

  subsample > 1 || throw(ArgumentError("subsample must be at least 2."))
  repeats > 0 || throw(ArgumentError("repeats must be positive."))

  estimates = Vector{Float64}(undef, repeats)
  for r = 1:repeats
    xs = sample(rng, 1:size(X, 1), min(subsample, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(subsample, size(Y, 1)); replace = false)
    estimates[r] = _exact_energydistance(X[xs, :], Y[ys, :])
  end
  return mean(estimates)
end

energydistance(X::AbstractVector, Y::AbstractVector; kwargs...) = energydistance(
  reshape(collect(Float64, X), :, 1),
  reshape(collect(Float64, Y), :, 1);
  kwargs...,
)

"""
    splitquality(data, result::SplitResult;
                 exact_threshold = 4_000, subsample = 2_000, repeats = 8,
                 rng = Random.default_rng()) -> Float64

Energy distance between the train and test rows of `data` under the same
preprocessing `datasplit` applied. Smaller is better. Computed exactly when
the total row count is at most `exact_threshold`; otherwise estimated from
`repeats` random size-`subsample` subsets. The subsampled estimate carries a
positive bias of order `1/subsample`, so use it to compare splits rather than
as an absolute value.
"""
function splitquality(
  data,
  result::SplitResult;
  exact_threshold::Int = 4_000,
  subsample::Int = 2_000,
  repeats::Int = 8,
  rng::AbstractRNG = Random.default_rng(),
)
  X = preprocess(data)
  train = X[result.train_indices, :]
  test = X[result.test_indices, :]
  if size(train, 1) + size(test, 1) <= exact_threshold
    return energydistance(train, test)
  end
  return energydistance(train, test; subsample, repeats, rng)
end
