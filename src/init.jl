using Random
using Statistics
using LinearAlgebra
using Base.Threads

"""
    compute_support_points(num_points::Int, dimensions::Int, sample_points::Matrix{Float64},
                           num_subsamples::Int, max_iterations::Int=500, tolerance::Float64=1e-10,
                           num_threads::Int=nthreads())

Compute support points for a given sample of data points. This function aims to find a set of `num_points`
in `dimensions`-dimensional space that approximates the distribution of the given `sample_points`.

# References
- Joseph, V. Roshan, and Akhil Vakayil. "SPlit: An optimal method for data splitting." Technometrics 64.2 (2022): 166-176.

Parameters:
- `data::Matrix{Float64}`: The matrix of sampled data points.
- `n_subsamples::Int`: The number of subsamples to use in the optimization process.
- `max_iter::Int`: Maximum number of iterations for the optimization (default: 500).
- `tolerance::Float64`: Tolerance level for optimization convergence (default: 1e-10).
- `n_threads::Int`: Number of threads to use for parallel computation (default: nthreads()).

Returns:
- `Matrix{Float64}`: A matrix of optimized support points with size (`num_points`, `dimensions`).

Example:
```julia
sample_points = randn(100, 2)
optimized_points = compute_support_points(10, 2, sample_points, 50)
````
"""
function compute_support_points(
  data::Matrix{Float64},
  n_subsamples::Int,
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  n_threads::Int = nthreads(),
)
  # from data, determine the number of points and dimensions
  n_samples = length(axes(data, 1))
  n_features = length(axes(data, 2))

  # Determine if random sampling is needed
  random_sampling_flag = num_subsamples < n_samples

  # Initialize weights and compute n0(large N in the paper)
  weights = ones(length(axes(data, 1)))
  n0 = num_data * num_dims

  # Check the number of threads specified and ensure it's valid
  num_cores = max(1, min(n_threads, nthreads()))

  # Initialize bounds for each dimension of the data points
  bounds = [extrema(data[:, i]) for i = 1:n_features]

  # Apply jittering if there are duplicate points in the sample
  if any(count(x -> count(==(x), data), unique(data)) .> 1)
    data .= data .+ 1e-10 * randn(length(axes(data, 1)), length(axes(data, 2)))
    # Ensure jittered points are within bounds
    for i = 1:dimensions
      data[:, i] .= clamp.(data[:, i], bounds[i]...)
    end
  end

  # Initialize the points to optimize with jittering and clamping
  initial_indices = randperm(length(axes(data, 1)))[1:num_points]
  initial_points = sample_points[initial_indices, :] .+ 1e-10 * randn(n_samples, n_features)
  for i = 1:dimensions
    initial_points[:, i] .= clamp.(initial_points[:, i], bounds[i]...)
  end
end
