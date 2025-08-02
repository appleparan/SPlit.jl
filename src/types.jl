"""
Type hierarchy for Julia-native SPlit implementation.
"""

using Distances
using Random

# Abstract type hierarchy
abstract type SplittingMethod end
abstract type SplittingResult end

"""
    SupportPointSplitter{M<:PreMetric} <: SplittingMethod

A splitting method based on support points optimization.

# Fields
- `metric::M`: Distance metric for optimization and subsampling
- `ratio::Float64`: Split ratio (0 < ratio < 1)
- `max_iterations::Int`: Maximum optimization iterations
- `tolerance::Float64`: Convergence tolerance
- `n_threads::Int`: Number of threads for parallel computation
- `kappa::Union{Nothing,Int}`: Subsample size for stochastic optimization
- `rng::AbstractRNG`: Random number generator for reproducibility
"""
struct SupportPointSplitter{M<:PreMetric} <: SplittingMethod
  metric::M
  ratio::Float64
  max_iterations::Int
  tolerance::Float64
  n_threads::Int
  kappa::Union{Nothing,Int}
  rng::AbstractRNG

  function SupportPointSplitter(
    metric::M = Euclidean();
    ratio::Float64 = 0.2,
    max_iterations::Int = 500,
    tolerance::Float64 = 1e-10,
    n_threads::Int = Threads.nthreads(),
    kappa::Union{Nothing,Int} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
  ) where {M<:PreMetric}

    if !(0 < ratio < 1)
      throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
    end

    if max_iterations <= 0
      throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
    end

    if tolerance <= 0
      throw(ArgumentError("tolerance must be positive, got $tolerance"))
    end

    if n_threads <= 0
      throw(ArgumentError("n_threads must be positive, got $n_threads"))
    end

    if kappa !== nothing && kappa <= 0
      throw(ArgumentError("kappa must be positive when specified, got $kappa"))
    end

    new{M}(metric, ratio, max_iterations, tolerance, n_threads, kappa, rng)
  end
end

"""
    SplitResult{T} <: SplittingResult

Result of a data splitting operation.

# Fields
- `train_indices::Vector{Int}`: Indices of training data
- `test_indices::Vector{Int}`: Indices of test data
- `quality::Union{Float64,Nothing}`: Split quality metric (if computed)
- `convergence::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations used
- `method::SplittingMethod`: Method used for splitting
"""
struct SplitResult{T<:SplittingMethod} <: SplittingResult
  train_indices::Vector{Int}
  test_indices::Vector{Int}
  quality::Union{Float64,Nothing}
  convergence::Bool
  iterations::Int
  method::T
end

# Convenience constructors
function SplitResult(
  train_indices::Vector{Int},
  test_indices::Vector{Int},
  method::T,
) where {T<:SplittingMethod}
  SplitResult(train_indices, test_indices, nothing, false, 0, method)
end

# Base interface implementations

function Base.show(io::IO, method::SupportPointSplitter)
  print(io, "SupportPointSplitter(")
  print(io, "metric=$(method.metric), ")
  print(io, "ratio=$(method.ratio), ")
  print(io, "max_iterations=$(method.max_iterations)")
  print(io, ")")
end

function Base.show(io::IO, result::SplitResult)
  print(io, "SplitResult(")
  print(io, "train=$(length(result.train_indices)), ")
  print(io, "test=$(length(result.test_indices))")
  if result.quality !== nothing
    print(io, ", quality=$(round(result.quality, digits=4))")
  end
  print(io, ")")
end

# Property access
"""
    ratio(method::SplittingMethod) -> Float64

Get the split ratio from a splitting method.
"""
ratio(method::SupportPointSplitter) = method.ratio

"""
    metric(method::SplittingMethod) -> PreMetric

Get the distance metric from a splitting method.
"""
metric(method::SupportPointSplitter) = method.metric

# Iterator interface for SplitResult
Base.iterate(result::SplitResult) = (result.train_indices, :test)
Base.iterate(result::SplitResult, state::Symbol) =
  state === :test ? (result.test_indices, nothing) : nothing
Base.eltype(::Type{SplitResult{T}}) where {T} = Vector{Int}
Base.IteratorSize(::Type{SplitResult{T}}) where {T} = Base.HasLength()
Base.length(result::SplitResult) = 2  # train and test indices

"""
    train_indices(result::SplittingResult) -> Vector{Int}

Extract training indices from split result.
"""
train_indices(result::SplitResult) = result.train_indices

"""
    test_indices(result::SplittingResult) -> Vector{Int}

Extract test indices from split result.
"""
test_indices(result::SplitResult) = result.test_indices

"""
    quality(result::SplittingResult) -> Union{Float64,Nothing}

Get the quality metric from split result.
"""
quality(result::SplitResult) = result.quality
