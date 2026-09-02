"""
Split-quality diagnostics based on the energy distance
(Székely & Rizzo; the objective minimized by support points,
Mak & Joseph 2018).
"""

using Distances: Euclidean, SqEuclidean, pairwise
using Random
using Statistics
using StatsBase: sample

# (i0:i1, j0:j1) row-block index pairs covering X × Y, in a fixed (outer X,
# inner Y) order independent of thread count.
function _block_ranges(n::Int, block::Int)
  return [(i0, min(i0 + block - 1, n)) for i0 = 1:block:n]
end

function _block_pairs(nx::Int, ny::Int, block::Int)
  xr = _block_ranges(nx, block)
  yr = _block_ranges(ny, block)
  return [(ix, iy) for ix in xr for iy in yr]
end

# Sum f(pair) over `pairs`, split across n_threads spawned tasks that each
# write disjoint entries of a preallocated vector, then reduced by `sum` in
# pair order — the result does not depend on the thread count.
function _threaded_block_sum(f, pairs::Vector, n_threads::Int)
  n = length(pairs)
  partial = zeros(Float64, n)
  n_threads = clamp(n_threads, 1, max(n, 1))
  chunks = Iterators.partition(1:n, cld(n, n_threads))
  Threads.@sync for chunk in chunks
    Threads.@spawn begin
      for idx in chunk
        partial[idx] = f(pairs[idx])
      end
    end
  end
  return sum(partial)
end

# Mean pairwise Euclidean distance between rows of X and rows of Y,
# accumulated block-wise so no n×n matrix is ever materialized.
function _mean_pairwise(
  X::AbstractMatrix,
  Y::AbstractMatrix;
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  pairs = _block_pairs(size(X, 1), size(Y, 1), block)
  total = _threaded_block_sum(pairs, n_threads) do (ix, iy)
    i0, i1 = ix
    j0, j1 = iy
    @views sum(pairwise(Euclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1))
  end
  return total / (size(X, 1) * size(Y, 1))
end

function _exact_energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix;
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  return 2 * _mean_pairwise(X, Y; block, n_threads) -
         _mean_pairwise(X, X; block, n_threads) - _mean_pairwise(Y, Y; block, n_threads)
end

# Mean Gaussian kernel value over all row pairs of X and Y, block-wise.
function _mean_kernel(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix;
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  scale = -1 / (2 * k.bandwidth^2)
  pairs = _block_pairs(size(X, 1), size(Y, 1), block)
  total = _threaded_block_sum(pairs, n_threads) do (ix, iy)
    i0, i1 = ix
    j0, j1 = iy
    @views D = pairwise(SqEuclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1)
    sum(d -> exp(scale * d), D)
  end
  return total / (size(X, 1) * size(Y, 1))
end

function _exact_mmd(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix;
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  return _mean_kernel(k, X, X; block, n_threads) + _mean_kernel(k, Y, Y; block, n_threads) -
         2 * _mean_kernel(k, X, Y; block, n_threads)
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
    energydistance(X, Y; estimator = Exact(), rng = Random.default_rng(),
                   n_threads = Threads.nthreads(), subsample = nothing, repeats = 8)

Energy distance between two samples whose rows are observations.

`estimator` selects how it is computed: [`Exact`](@ref) (default) evaluates
every pairwise term block-wise, threaded over `n_threads`; [`Subsample`](@ref)
averages the exact statistic over `repeats` random size-`m` row subsets drawn
with `rng` — this estimate carries a positive bias of order `1/m`, so use it
to compare splits rather than as an absolute value; [`RandomSlices`](@ref) averages
`k` random one-dimensional projections drawn with `rng` (unbiased). The
`subsample = m, repeats = r` keywords are a compatibility path equivalent to
`estimator = Subsample(m, r)`.

Vectors are treated as single-column samples.
"""
function energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix;
  estimator::DiscrepancyEstimator = Exact(),
  rng::AbstractRNG = Random.default_rng(),
  n_threads::Int = Threads.nthreads(),
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))
  subsample === nothing || (estimator = Subsample(subsample, repeats))
  return _energydistance(estimator, X, Y, rng, n_threads)
end

_energydistance(::Exact, X, Y, rng, n_threads) = _exact_energydistance(X, Y; n_threads)
function _energydistance(e::Subsample, X, Y, rng, n_threads)
  (size(X, 1) <= e.m && size(Y, 1) <= e.m) && return _exact_energydistance(X, Y; n_threads)
  estimates = Vector{Float64}(undef, e.repeats)
  for r = 1:e.repeats
    xs = sample(rng, 1:size(X, 1), min(e.m, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(e.m, size(Y, 1)); replace = false)
    estimates[r] = _exact_energydistance(X[xs, :], Y[ys, :]; n_threads)
  end
  return mean(estimates)
end
_energydistance(e::RandomSlices, X, Y, rng, n_threads) =
  _sliced_energydistance(X, Y, e.k, rng)
_energydistance(e::DiscrepancyEstimator, X, Y, rng, n_threads) =
  _undefined(e, "the energy distance")

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
