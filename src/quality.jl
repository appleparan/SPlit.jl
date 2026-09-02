"""
Split-quality diagnostics based on the energy distance
(Székely & Rizzo; the objective minimized by support points,
Mak & Joseph 2018).
"""

using Distances: Euclidean, SqEuclidean, pairwise
using LinearAlgebra: dot
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
  return [(xblock, yblock) for xblock in xr for yblock in yr]
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
  total = _threaded_block_sum(pairs, n_threads) do (xblock, yblock)
    i0, i1 = xblock
    j0, j1 = yblock
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

# Weighted mean pairwise distance Σᵢⱼ wxᵢ wyⱼ ‖xᵢ − yⱼ‖ with both weight
# vectors scaled to sum one (so no 1/(nm) division), block-wise.
function _mean_pairwise(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  pairs = _block_pairs(size(X, 1), size(Y, 1), block)
  return _threaded_block_sum(pairs, n_threads) do (xblock, yblock)
    i0, i1 = xblock
    j0, j1 = yblock
    @views D = pairwise(Euclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1)
    @views dot(wx[i0:i1], D, wy[j0:j1])
  end
end

function _exact_energydistance(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  return 2 * _mean_pairwise(X, Y, wx, wy; block, n_threads) -
         _mean_pairwise(X, X, wx, wx; block, n_threads) -
         _mean_pairwise(Y, Y, wy, wy; block, n_threads)
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
  total = _threaded_block_sum(pairs, n_threads) do (xblock, yblock)
    i0, i1 = xblock
    j0, j1 = yblock
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

# Weighted mean kernel value Σᵢⱼ wxᵢ wyⱼ k(xᵢ, yⱼ), weights scaled to sum one.
function _mean_kernel(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  scale = -1 / (2 * k.bandwidth^2)
  pairs = _block_pairs(size(X, 1), size(Y, 1), block)
  return _threaded_block_sum(pairs, n_threads) do (xblock, yblock)
    i0, i1 = xblock
    j0, j1 = yblock
    @views D = pairwise(SqEuclidean(), X[i0:i1, :], Y[j0:j1, :]; dims = 1)
    D .= exp.(scale .* D)
    @views dot(wx[i0:i1], D, wy[j0:j1])
  end
end

function _exact_mmd(
  k::GaussianKernel{Float64},
  X::AbstractMatrix,
  Y::AbstractMatrix,
  wx::AbstractVector{Float64},
  wy::AbstractVector{Float64};
  block::Int = 1_024,
  n_threads::Int = Threads.nthreads(),
)
  return _mean_kernel(k, X, X, wx, wx; block, n_threads) +
         _mean_kernel(k, Y, Y, wy, wy; block, n_threads) -
         2 * _mean_kernel(k, X, Y, wx, wy; block, n_threads)
end

# Weights of a drawn subsample rescaled to sum one.
function _renormalized(w::AbstractVector)
  s = sum(w)
  s > 0 || throw(ArgumentError("a subsample drew only zero-weight rows; use a larger m"))
  return w ./ s
end

"""
    mmd(X, Y, kernel; estimator = Exact(), rng = Random.default_rng(),
        n_threads = Threads.nthreads(), subsample = nothing, repeats = 8)

Squared maximum mean discrepancy (Gretton et al. 2012) between two samples
whose rows are observations, under `kernel`.

`estimator` selects how it is computed: [`Exact`](@ref) (default) accumulates
the V-statistic `mean k(X,X) + mean k(Y,Y) − 2 mean k(X,Y)` block-wise,
threaded over `n_threads`; [`Subsample`](@ref) averages the exact statistic
over `repeats` random size-`m` row subsets drawn with `rng` — this estimate
carries a positive bias of order `1/m`, so use it to compare splits rather
than as an absolute value; [`RandomFeatures`](@ref) estimates it with `D`
random Fourier features drawn with `rng` (unbiased), and is defined for
`GaussianKernel` only. For `EnergyKernel()` this delegates to
[`energydistance`](@ref) with the same `estimator` (so [`RandomSlices`](@ref)
works and `RandomFeatures` raises `ArgumentError`). A `:median` bandwidth is
resolved on the pooled rows of `X` and `Y` with `rng`. The
`subsample = m, repeats = r` keywords are a compatibility path equivalent to
`estimator = Subsample(m, r)`.

`weights_x` and `weights_y` (one non-negative entry per row of `X` and
`Y`; `nothing` means uniform) turn each sample into a weighted empirical
distribution, `Σᵢ w̄ᵢ δ(xᵢ)` with `w̄` scaled to sum one, and the statistic
becomes `Σ w̄ᵢ w̄ₖ k(xᵢ, xₖ) + Σ v̄ⱼ v̄ₗ k(yⱼ, yₗ) − 2 Σ w̄ᵢ v̄ⱼ k(xᵢ, yⱼ)`.
`Subsample` draws rows uniformly and rescales the drawn weights to sum
one. Weights proportional to duplication counts are equivalent to
duplicating rows. A constant weight vector is treated as `nothing`, so
uniform weights take the unweighted path and reproduce its value exactly.
"""
function mmd(
  X::AbstractMatrix,
  Y::AbstractMatrix,
  kernel::SplitKernel;
  estimator::DiscrepancyEstimator = Exact(),
  rng::AbstractRNG = Random.default_rng(),
  n_threads::Int = Threads.nthreads(),
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  weights_x::Union{Nothing,AbstractVector} = nothing,
  weights_y::Union{Nothing,AbstractVector} = nothing,
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))
  subsample === nothing || (estimator = Subsample(subsample, repeats))
  weights_x === nothing || _check_weights(weights_x, size(X, 1))
  weights_y === nothing || _check_weights(weights_y, size(Y, 1))
  weights_x = _uniform_as_nothing(weights_x)
  weights_y = _uniform_as_nothing(weights_y)
  kernel isa EnergyKernel &&
    return energydistance(X, Y; estimator, rng, n_threads, weights_x, weights_y)
  k = isresolved(kernel) ? kernel : resolve(kernel, vcat(X, Y), rng)
  weights_x === nothing &&
    weights_y === nothing &&
    return _mmd(estimator, k, X, Y, rng, n_threads)
  wx = _side_weights(weights_x, size(X, 1))
  wy = _side_weights(weights_y, size(Y, 1))
  return _mmd(estimator, k, X, Y, wx, wy, rng, n_threads)
end

_mmd(::Exact, k, X, Y, rng, n_threads) = _exact_mmd(k, X, Y; n_threads)
function _mmd(e::Subsample, k, X, Y, rng, n_threads)
  (size(X, 1) <= e.m && size(Y, 1) <= e.m) && return _exact_mmd(k, X, Y; n_threads)
  estimates = Vector{Float64}(undef, e.repeats)
  for r = 1:e.repeats
    xs = sample(rng, 1:size(X, 1), min(e.m, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(e.m, size(Y, 1)); replace = false)
    estimates[r] = _exact_mmd(k, X[xs, :], Y[ys, :]; n_threads)
  end
  return mean(estimates)
end
_mmd(e::RandomFeatures, k::GaussianKernel{Float64}, X, Y, rng, n_threads) =
  _rff_mmd(k, X, Y, e.D, rng)
_mmd(e::DiscrepancyEstimator, k, X, Y, rng, n_threads) =
  _undefined(e, "mmd under $(nameof(typeof(k)))")

_mmd(::Exact, k, X, Y, wx, wy, rng, n_threads) = _exact_mmd(k, X, Y, wx, wy; n_threads)
function _mmd(e::Subsample, k, X, Y, wx, wy, rng, n_threads)
  (size(X, 1) <= e.m && size(Y, 1) <= e.m) && return _exact_mmd(k, X, Y, wx, wy; n_threads)
  estimates = Vector{Float64}(undef, e.repeats)
  for r = 1:e.repeats
    xs = sample(rng, 1:size(X, 1), min(e.m, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(e.m, size(Y, 1)); replace = false)
    estimates[r] = _exact_mmd(
      k,
      X[xs, :],
      Y[ys, :],
      _renormalized(wx[xs]),
      _renormalized(wy[ys]);
      n_threads,
    )
  end
  return mean(estimates)
end
_mmd(e::DiscrepancyEstimator, k, X, Y, wx, wy, rng, n_threads) =
  _undefined(e, "mmd under $(nameof(typeof(k))) with weights")

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

`weights_x` and `weights_y` (one non-negative entry per row; `nothing`
means uniform) give the energy distance between the weighted empirical
distributions `Σᵢ w̄ᵢ δ(xᵢ)` and `Σⱼ v̄ⱼ δ(yⱼ)`, with the weights scaled to
sum one: `2 Σ w̄ᵢ v̄ⱼ ‖xᵢ − yⱼ‖ − Σ w̄ᵢ w̄ₖ ‖xᵢ − xₖ‖ − Σ v̄ⱼ v̄ₗ ‖yⱼ − yₗ‖`.
Weights proportional to duplication counts are equivalent to duplicating
rows. A constant weight vector is treated as `nothing`, so uniform weights
take the unweighted path and reproduce its value exactly.

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
  weights_x::Union{Nothing,AbstractVector} = nothing,
  weights_y::Union{Nothing,AbstractVector} = nothing,
)
  size(X, 2) == size(Y, 2) ||
    throw(ArgumentError("Samples must have the same number of columns."))
  subsample === nothing || (estimator = Subsample(subsample, repeats))
  weights_x === nothing || _check_weights(weights_x, size(X, 1))
  weights_y === nothing || _check_weights(weights_y, size(Y, 1))
  weights_x = _uniform_as_nothing(weights_x)
  weights_y = _uniform_as_nothing(weights_y)
  weights_x === nothing &&
    weights_y === nothing &&
    return _energydistance(estimator, X, Y, rng, n_threads)
  wx = _side_weights(weights_x, size(X, 1))
  wy = _side_weights(weights_y, size(Y, 1))
  return _energydistance(estimator, X, Y, wx, wy, rng, n_threads)
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

_energydistance(::Exact, X, Y, wx, wy, rng, n_threads) =
  _exact_energydistance(X, Y, wx, wy; n_threads)
function _energydistance(e::Subsample, X, Y, wx, wy, rng, n_threads)
  (size(X, 1) <= e.m && size(Y, 1) <= e.m) &&
    return _exact_energydistance(X, Y, wx, wy; n_threads)
  estimates = Vector{Float64}(undef, e.repeats)
  for r = 1:e.repeats
    xs = sample(rng, 1:size(X, 1), min(e.m, size(X, 1)); replace = false)
    ys = sample(rng, 1:size(Y, 1), min(e.m, size(Y, 1)); replace = false)
    estimates[r] = _exact_energydistance(
      X[xs, :],
      Y[ys, :],
      _renormalized(wx[xs]),
      _renormalized(wy[ys]);
      n_threads,
    )
  end
  return mean(estimates)
end
_energydistance(e::DiscrepancyEstimator, X, Y, wx, wy, rng, n_threads) =
  _undefined(e, "the energy distance with weights")

energydistance(X::AbstractVector, Y::AbstractVector; kwargs...) = energydistance(
  reshape(collect(Float64, X), :, 1),
  reshape(collect(Float64, Y), :, 1);
  kwargs...,
)

# Automatic fallback for splitquality's estimator = nothing above
# exact_threshold, chosen by the selection experiment
# (docs/src/assets/benchmarks/estimators.md, embedded on the Design
# experiments page): on the four Phase 2b datasets at N = 10,000,
# worst-case max error is 14x lower for RandomSlices(64) (0.00197 ->
# 0.00014) at 9% of Subsample(2_000, 8)'s mean time, and 330x lower for
# RandomFeatures(512) (0.000175 -> 5.36e-7) at 22% of its time.
const ENERGY_FALLBACK = RandomSlices(64)
const GAUSSIAN_FALLBACK = RandomFeatures(512)

_fallback_estimator(::EnergyKernel) = ENERGY_FALLBACK
_fallback_estimator(::GaussianKernel) = GAUSSIAN_FALLBACK

"""
    splitquality(data, result::SplitResult;
                 kernel = EnergyKernel(), estimator = nothing,
                 exact_threshold = 20_000, subsample = nothing, repeats = 8,
                 rng = Random.default_rng(), n_threads = Threads.nthreads()) -> Float64

Discrepancy between the train and test rows of `data` under the same
preprocessing `datasplit` applied — the energy distance by default, or
[`mmd`](@ref) under `kernel`. Smaller is better.

`estimator = nothing` (the default) computes exactly ([`Exact`](@ref)) when
the total row count is at most `exact_threshold`, and otherwise falls back to
a fixed [`DiscrepancyEstimator`](@ref) chosen by the selection experiment on
the Design experiments page (currently [`RandomSlices`](@ref)`(64)` for
`EnergyKernel` and [`RandomFeatures`](@ref)`(512)` for `GaussianKernel` — see
`_fallback_estimator`). Pass any `DiscrepancyEstimator` to override. The old
`subsample = m, repeats = r` keywords are a compatibility path: when
`subsample` is given explicitly, it always wins over `estimator` and maps to
`Subsample(m, r)`.
"""
function splitquality(
  data,
  result::SplitResult;
  kernel::SplitKernel = EnergyKernel(),
  estimator::Union{Nothing,DiscrepancyEstimator} = nothing,
  exact_threshold::Int = 20_000,
  subsample::Union{Nothing,Int} = nothing,
  repeats::Int = 8,
  rng::AbstractRNG = Random.default_rng(),
  n_threads::Int = Threads.nthreads(),
)
  X = preprocess(data)
  train = X[result.train_indices, :]
  test = X[result.test_indices, :]
  k = isresolved(kernel) ? kernel : resolve(kernel, X, rng)
  chosen = if subsample !== nothing
    Subsample(subsample, repeats)
  elseif estimator !== nothing
    estimator
  elseif size(train, 1) + size(test, 1) <= exact_threshold
    Exact()
  else
    _fallback_estimator(k)
  end
  return mmd(train, test, k; estimator = chosen, rng, n_threads)
end
