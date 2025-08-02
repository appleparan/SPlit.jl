"""
Main SPlit functions for optimal data splitting.
"""

using DataFrames
using Statistics
using Base.Threads
using Random
using Polynomials
using StatsModels
using Distances

include("data_preprocessing.jl")
include("support_points.jl")
include("subsampling.jl")

"""
    split_data(data; split_ratio::Float64=0.2, kappa::Union{Nothing,Int}=nothing,
               max_iterations::Int=500, tolerance::Float64=1e-10,
               n_threads::Int=Threads.nthreads(), metric::PreMetric=Euclidean())

Split a dataset optimally for training and testing using the support points method.

This is the main function of SPlit.jl, equivalent to the `SPlit()` function in the R package.

# Arguments
- `data`: Input dataset (Matrix or DataFrame)
- `split_ratio`: Ratio for the smaller subset (default: 0.2 for 80-20 split)
- `kappa`: Subsample size for stochastic optimization (default: use all data)
- `max_iterations`: Maximum iterations for support points optimization
- `tolerance`: Convergence tolerance for optimization
- `n_threads`: Number of threads for parallel computation
- `metric`: Distance metric to use (default: Euclidean(), can use EnergyDistance())

# Returns
- Vector of indices for the smaller subset

# Examples
```julia
using SPlit
using Random
using Distances

# Generate sample data
Random.seed!(123)
n = 100
X = randn(n, 2)
Y = X[:, 1] .+ X[:, 2].^2 .+ 0.1 * randn(n)
data = hcat(X, Y)

# Split data with default Euclidean distance
test_indices = split_data(data)
train_data = data[setdiff(1:n, test_indices), :]
test_data = data[test_indices, :]

# Use Energy Distance for splitting (better for complex distributions)
using SPlit: EnergyDistance
test_indices_energy = split_data(data; metric=EnergyDistance(Euclidean()))

# Use Manhattan distance
test_indices_manhattan = split_data(data; metric=Cityblock())
```
"""
function split_data(
  data;
  split_ratio::Float64 = 0.2,
  kappa::Union{Nothing,Int} = nothing,
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  n_threads::Int = Threads.nthreads(),
  metric::PreMetric = Euclidean(),
)

  # Validate split_ratio
  if split_ratio <= 0 || split_ratio >= 1
    throw(ArgumentError("split_ratio should be in (0, 1)."))
  end

  # Preprocess data
  processed_data = format_data(data)
  n_total, p = length(axes(processed_data, 1)), length(axes(processed_data, 2))

  # Determine number of points for smaller subset
  n_subset = round(Int, min(split_ratio, 1 - split_ratio) * n_total)

  # Set up kappa (subsample size for stochastic optimization)
  if kappa === nothing
    kappa = n_total  # Use all data
    use_stochastic = false
  else
    if kappa <= 0
      throw(ArgumentError("kappa should be positive."))
    end
    kappa = min(n_total, ceil(Int, kappa * n_subset))
    use_stochastic = kappa < n_total
  end

  # Compute support points
  println("Computing support points...")
  support_points = compute_support_points(
    n_subset,
    p,
    processed_data,
    kappa;
    max_iterations = max_iterations,
    tolerance = tolerance,
    n_threads = n_threads,
    use_stochastic = use_stochastic,
    metric = metric,
  )

  # Perform subsampling to get indices
  println("Performing subsampling...")
  indices = subsample_indices(processed_data, support_points; metric = metric)

  return indices
end

# Convenience function with original R naming
"""
    split_data_r(data; splitRatio::Float64=0.2, kappa::Union{Nothing,Int}=nothing,
                 maxIterations::Int=500, tolerance::Float64=1e-10,
                 nThreads::Int=Threads.nthreads(), metric::PreMetric=Euclidean())

Alias for `split_data` with R-style parameter naming for compatibility.
"""
function split_data_r(
  data;
  splitRatio::Float64 = 0.2,
  kappa::Union{Nothing,Int} = nothing,
  maxIterations::Int = 500,
  tolerance::Float64 = 1e-10,
  nThreads::Int = Threads.nthreads(),
  metric::PreMetric = Euclidean(),
)
  return split_data(
    data;
    split_ratio = splitRatio,
    kappa = kappa,
    max_iterations = maxIterations,
    tolerance = tolerance,
    n_threads = nThreads,
    metric = metric,
  )
end

"""
    optimal_split_ratio(x, y; method::String="simple", degree::Int=2)

Find the optimal splitting ratio by estimating the number of model parameters.

# Arguments
- `x`: Input matrix or vector
- `y`: Response variable
- `method`: "simple" (uses √n rule) or "regression" (stepwise regression)
- `degree`: Polynomial degree for regression method

# Returns
- Optimal split ratio for testing set

# Examples
```julia
using Random

Random.seed!(123)
X = randn(100, 3)
Y = X[:, 1] + X[:, 2]^2 + 0.1 * randn(100)
optimal_ratio = optimal_split_ratio(X, Y)
```

# References
Joseph, V. R. (2022). Optimal Ratio for Data Splitting. Statistical Analysis & Data Mining: The ASA Data Science Journal, 15(4), 537-546.
"""
function optimal_split_ratio(x, y; method::String = "simple", degree::Int = 2)
  if method == "regression"
    if !all(isa.(y, Number)) || any(ismissing, y)
      @warn "Using method='simple' for non-numeric response"
      p = sqrt(length(axes(unique(x, dims = 1), 1)))
    else
      # For simplicity in this implementation, we'll use the simple method
      # A full stepwise regression implementation would be quite complex
      @warn "Regression method not fully implemented, using simple method"
      p = sqrt(length(axes(unique(x, dims = 1), 1)))
    end
  else
    # Simple method: use square root of unique rows
    if isa(x, Vector)
      x = reshape(x, :, 1)
    end
    p = sqrt(length(axes(unique(x, dims = 1), 1)))
  end

  γ = 1 / (sqrt(p) + 1)
  return γ
end

# R-style alias
splitratio(x, y; method::String = "simple", degree::Int = 2) =
  optimal_split_ratio(x, y; method = method, degree = degree)
