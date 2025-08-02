"""
Julia-native interface for data splitting using multiple dispatch.
"""

using DataFrames
using Statistics
using Random

# Main splitting interface
"""
    split(method::SplittingMethod, data) -> SplitResult

Split data using the specified method.

# Examples
```julia
# Basic usage
splitter = SupportPointSplitter(Euclidean(); ratio=0.2)
result = SPlit.split(splitter, data)  # Use qualified name to avoid Base.split conflict
train_idx, test_idx = result  # Iterator interface

# With different metrics
energy_splitter = SupportPointSplitter(EnergyDistance(Euclidean()))
result = SPlit.split(energy_splitter, data)

# Access results
println("Training size: ", length(train_indices(result)))
println("Test size: ", length(test_indices(result)))
```
"""
function split end

# Convenience alias to avoid Base.split conflict
"""
    datasplit(method::SplittingMethod, data) -> SplitResult

Alias for `split` to avoid naming conflict with `Base.split`.
"""
const datasplit = split

# Multiple dispatch for different data types
"""
    split(method::SupportPointSplitter, data::AbstractMatrix) -> SplitResult

Split matrix data where each row is an observation.
"""
function split(method::SupportPointSplitter, data::AbstractMatrix)
  # Preprocess data
  processed_data = _preprocess_matrix(data)

  # Perform splitting
  return _split_support_points(method, processed_data, size(data, 1))
end

"""
    split(method::SupportPointSplitter, data::DataFrame) -> SplitResult

Split DataFrame with automatic categorical encoding.
"""
function split(method::SupportPointSplitter, data::DataFrame)
  # Preprocess DataFrame (handle categoricals, standardize)
  processed_data = _preprocess_dataframe(data)

  # Perform splitting
  return _split_support_points(method, processed_data, nrow(data))
end

"""
    split(method::SupportPointSplitter, data::AbstractVector) -> SplitResult

Split 1D vector data.
"""
function split(method::SupportPointSplitter, data::AbstractVector)
  # Convert to matrix format
  data_matrix = reshape(Float64.(data), :, 1)

  # Preprocess and split
  processed_data = _preprocess_matrix(data_matrix)
  return _split_support_points(method, processed_data, length(data))
end

# Convenience methods with quality assessment
"""
    split_with_quality(method::SplittingMethod, data) -> SplitResult

Split data and compute quality metric.
"""
function split_with_quality(method::SplittingMethod, data)
  result = split(method, data)
  return _add_quality_assessment(result, data, method)
end

# Batch splitting interface
"""
    split(methods::Vector{<:SplittingMethod}, data) -> Vector{SplitResult}

Apply multiple splitting methods to the same data.
"""
function split(methods::Vector{<:SplittingMethod}, data)
  return [split(method, data) for method in methods]
end

# Aliases for split_with_quality
const datasplit_with_quality = split_with_quality

# Indexing interface
"""
    getindex(data, result::SplitResult, subset::Symbol) -> SubArray

Extract training or test data using split result.

# Examples
```julia
result = split(splitter, data)
train_data = data[result, :train]
test_data = data[result, :test]
```
"""
Base.getindex(data::AbstractArray, result::SplitResult, subset::Symbol) =
  subset === :train ? view(data, result.train_indices, :) :
  subset === :test ? view(data, result.test_indices, :) :
  throw(ArgumentError("subset must be :train or :test, got :$subset"))

Base.getindex(data::DataFrame, result::SplitResult, subset::Symbol) =
  subset === :train ? view(data, result.train_indices, :) :
  subset === :test ? view(data, result.test_indices, :) :
  throw(ArgumentError("subset must be :train or :test, got :$subset"))

# Internal implementation functions
function _split_support_points(
  method::SupportPointSplitter,
  processed_data::Matrix{Float64},
  n_total::Int,
)
  n_subset = round(Int, min(method.ratio, 1 - method.ratio) * n_total)
  p = size(processed_data, 2)

  # Determine stochastic optimization settings
  kappa = method.kappa === nothing ? n_total : min(n_total, method.kappa)
  use_stochastic = kappa < n_total

  # Compute support points
  support_points, convergence, iterations = _compute_support_points(
    n_subset,
    p,
    processed_data,
    kappa;
    method.max_iterations,
    method.tolerance,
    method.n_threads,
    method.metric,
    use_stochastic,
    method.rng,
  )

  # Perform subsampling
  test_indices = _subsample_indices(processed_data, support_points, method.metric)
  train_indices = setdiff(1:n_total, test_indices)

  return SplitResult(train_indices, test_indices, nothing, convergence, iterations, method)
end

function _preprocess_matrix(data::AbstractMatrix)
  # Convert to Float64 and handle missing values
  if any(ismissing, data)
    throw(ArgumentError("Missing values not supported"))
  end

  processed = Float64.(data)

  # Remove constant columns
  constant_cols = [std(processed[:, j]) ≈ 0 for j in axes(processed, 2)]
  if all(constant_cols)
    throw(ArgumentError("All columns are constant"))
  end

  if any(constant_cols)
    processed = processed[:, .!constant_cols]
  end

  # Standardize columns
  for j in axes(processed, 2)
    col_mean = mean(processed[:, j])
    col_std = std(processed[:, j])
    if col_std > 0
      processed[:, j] = (processed[:, j] .- col_mean) ./ col_std
    end
  end

  return processed
end

function _preprocess_dataframe(data::DataFrame)
  # Use existing format_data function for now
  # TODO: Reimplement in Julia-native style
  return format_data(data)
end

function _add_quality_assessment(result::SplitResult, data, method::SplittingMethod)
  # Compute energy distance quality metric
  if size(data, 1) < 4  # Need minimum samples for meaningful quality assessment
    return SplitResult(
      result.train_indices,
      result.test_indices,
      nothing,
      result.convergence,
      result.iterations,
      result.method,
    )
  end

  train_data = data[result.train_indices, :]
  test_data = data[result.test_indices, :]

  # Use energy distance with Euclidean metric for quality assessment
  energy_dist = EnergyDistance(Euclidean())
  quality_score = energy_dist(Matrix(train_data'), Matrix(test_data'))

  return SplitResult(
    result.train_indices,
    result.test_indices,
    quality_score,
    result.convergence,
    result.iterations,
    result.method,
  )
end

# Wrapper functions that delegate to existing implementations
function _compute_support_points(
  n,
  p,
  data,
  kappa;
  max_iterations,
  tolerance,
  n_threads,
  metric,
  use_stochastic,
  rng,
)
  # Set RNG seed for reproducibility
  Random.seed!(rng, 42)

  # Call existing function with metric parameter
  points = compute_support_points(
    n,
    p,
    data,
    kappa;
    max_iterations,
    tolerance,
    n_threads,
    use_stochastic,
    metric,
  )

  # For now, assume convergence and return max iterations
  # TODO: Modify compute_support_points to return convergence info
  convergence = true  # Placeholder
  iterations = max_iterations  # Placeholder

  return points, convergence, iterations
end

function _subsample_indices(data, support_points, metric)
  return subsample_indices(data, support_points; metric = metric)
end
