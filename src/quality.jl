"""
Split-quality diagnostics based on the energy distance
(Székely & Rizzo; the objective minimized by support points,
Mak & Joseph 2018).
"""

using Distances: Euclidean, SqEuclidean, pairwise
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

# Mean Gaussian kernel value over all row pairs of X and Y, block-wise.
function _mean_kernel(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix;
  block::Int = 1_024,
)
  scale = -1 / (2 * k.bandwidth^2)
  total = 0.0
  for i0 = 1:block:size(X, 1)
    i1 = min(i0 + block - 1, size(X, 1))
    for j0 = 1:block:size(Y, 1)
      j1 = min(j0 + block - 1, size(Y, 1))
      @views D = pairwise(SqEuclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1)
      total += sum(d -> exp(scale * d), D)
    end
  end
  return total / (size(X, 1) * size(Y, 1))
end

function _exact_mmd(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix;
  block::Int = 1_024,
)
  return _mean_kernel(k, X, X; block) + _mean_kernel(k, Y, Y; block) -
         2 * _mean_kernel(k, X, Y; block)
end

"""
    mmd(X, Y, kernel; subsample = nothing, repeats = 8, rng = Random.default_rng())

Squared maximum mean discrepancy (Gretton et al. 2012) between two samples
whose rows are observations, under `kernel`. For `EnergyKernel()` this is the
energy distance (see [`energydistance`](@ref)); for `GaussianKernel` the
V-statistic `mean k(X,X) + mean k(Y,Y) − 2 mean k(X,Y)` is accumulated
block-wise. A `:median` bandwidth is resolved on the pooled rows of `X` and
`Y` with `rng`.

With `subsample = m`, the value is the mean of `repeats` estimates on random
size-`m` row subsets; like the energy distance, this estimate carries a
positive bias of order `1/subsample` and is meant for comparing splits.
"""
function mmd(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  kernel::SplitKernel;
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  rng::AbstractRNG = Random.default_rng(),
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))
  kernel isa EnergyKernel && return energydistance(X, Y; subsample, repeats, rng)
  k = isresolved(kernel) ? kernel : resolve(kernel, vcat(X, Y), rng)

  if subsample === nothing || (size(X, 1) <= subsample && size(Y, 1) <= subsample)
    return _exact_mmd(k, X, Y)
  end
  subsample > 1 || throw(ArgumentError("subsample must be at least 2."))
  repeats > 0 || throw(ArgumentError("repeats must be positive."))
  estimates = Vector{Float64}(undef, repeats)
  for r = 1:repeats
    xs = sample(rng, 1:size(X, 1), min(subsample, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(subsample, size(Y, 1)); replace = false)
    estimates[r] = _exact_mmd(k, X[xs, :], Y[ys, :])
  end
  return mean(estimates)
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
                 kernel = EnergyKernel(), exact_threshold = 4_000, subsample = 2_000,
                 repeats = 8, rng = Random.default_rng()) -> Float64

Discrepancy between the train and test rows of `data` under the same
preprocessing `datasplit` applied — the energy distance by default, or
[`mmd`](@ref) under `kernel`. Smaller is better. Computed exactly when the
total row count is at most `exact_threshold`; otherwise estimated from
`repeats` random size-`subsample` subsets. The subsampled estimate carries a
positive bias of order `1/subsample`, so use it to compare splits rather than
as an absolute value.
"""
function splitquality(
  data,
  result::SplitResult;
  kernel::SplitKernel = EnergyKernel(),
  exact_threshold::Int = 4_000,
  subsample::Int = 2_000,
  repeats::Int = 8,
  rng::AbstractRNG = Random.default_rng(),
)
  X = preprocess(data)
  train = X[result.train_indices, :]
  test = X[result.test_indices, :]
  k = isresolved(kernel) ? kernel : resolve(kernel, X, rng)
  if size(train, 1) + size(test, 1) <= exact_threshold
    return mmd(train, test, k)
  end
  return mmd(train, test, k; subsample, repeats, rng)
end
