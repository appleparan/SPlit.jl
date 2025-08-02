"""
Python bindings for SPlit.jl using PythonCall.jl

This module provides Python-compatible wrappers for the main SPlit.jl functions.
"""

using PythonCall
import ..SPlit: split_data, optimal_split_ratio, compute_support_points, format_data


"""
    py_split_data(X; kwargs...)

Python-compatible wrapper for split_data function.
Returns a named tuple with train_indices and test_indices.
"""
function py_split_data(X; kwargs...)
  # Convert Python array to Julia array if needed
  X_jl = pyconvert(Array, X)

  # Call the main split_data function
  indices = split_data(X_jl; kwargs...)

  # Determine train indices (complement of test indices)
  n_total = size(X_jl, 1)
  all_indices = collect(1:n_total)
  train_indices = setdiff(all_indices, indices)

  # Return as named tuple for Python
  return (train_indices = train_indices, test_indices = indices)
end

"""
    py_optimal_split_ratio(X; kwargs...)

Python-compatible wrapper for optimal_split_ratio function.
"""
function py_optimal_split_ratio(X; kwargs...)
  # Convert Python array to Julia array if needed
  X_jl = pyconvert(Array, X)

  # For now, just return a simple ratio since the full implementation
  # needs both X and y
  n = size(X_jl, 1)
  p_effective = sqrt(n)  # Simple heuristic

  γ = 1 / (sqrt(p_effective) + 1)
  return γ
end

"""
    py_compute_support_points(X; n_points=nothing, kwargs...)

Python-compatible wrapper for compute_support_points function.
"""
function py_compute_support_points(X; n_points = nothing, kwargs...)
  # Convert Python array to Julia array if needed
  X_jl = pyconvert(Array, X)

  n_data, p = size(X_jl)

  # Determine number of support points if not specified
  if n_points === nothing
    n_points = min(50, max(10, n_data ÷ 10))  # Heuristic
  end

  # Format data (standardize, etc.)
  processed_data = format_data(X_jl)

  # Set default kappa for subsampling
  kappa = get(kwargs, :kappa, n_data)
  use_stochastic = kappa < n_data

  # Call the support points computation
  points = compute_support_points(
    n_points,
    p,
    processed_data,
    kappa;
    use_stochastic = use_stochastic,
    kwargs...,
  )

  return points
end

# Export Python-compatible functions
export py_split_data, py_optimal_split_ratio, py_compute_support_points
