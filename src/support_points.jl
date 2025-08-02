"""
Support points computation for optimal data splitting.
"""

using LinearAlgebra
using Random
using Statistics
using Base.Threads
using StatsBase

"""
    jitter_data!(data::Matrix{Float64}, bounds::Matrix{Float64})

Add small random noise to data to handle duplicates, ensuring values stay within bounds.

# Arguments
- `data`: Data matrix to jitter (modified in-place)
- `bounds`: Bounds matrix with min/max for each dimension
"""
function jitter_data!(data::Matrix{Float64}, bounds::Matrix{Float64})
  n, p = length(axes(data, 1)), length(axes(data, 2))

  for j = 1:p
    # Add small jitter
    jitter_amount = (bounds[j, 2] - bounds[j, 1]) * 1e-6
    data[:, j] .+= jitter_amount .* (2 .* rand(n) .- 1)

    # Ensure bounds are respected
    data[:, j] = clamp.(data[:, j], bounds[j, 1], bounds[j, 2])
  end
end

"""
    compute_bounds(data::Matrix{Float64})

Compute min/max bounds for each dimension of the data.

# Arguments
- `data`: Input data matrix

# Returns
- Matrix with bounds (p×2) where p is number of dimensions
"""
function compute_bounds(data::Matrix{Float64})
  p = length(axes(data, 2))
  bounds = Matrix{Float64}(undef, p, 2)

  for j = 1:p
    bounds[j, 1] = minimum(data[:, j])  # min
    bounds[j, 2] = maximum(data[:, j])  # max
  end

  return bounds
end

"""
    initialize_support_points(n::Int, p::Int, data::Matrix{Float64}, bounds::Matrix{Float64})

Initialize support points by sampling from data with jitter.

# Arguments
- `n`: Number of support points to generate
- `p`: Number of dimensions
- `data`: Data matrix to sample from
- `bounds`: Bounds for each dimension

# Returns
- Matrix of initialized support points (n×p)
"""
function initialize_support_points(
  n::Int,
  p::Int,
  data::Matrix{Float64},
  bounds::Matrix{Float64},
)
  n_data = length(axes(data, 1))

  # Sample indices without replacement
  indices = sample(1:n_data, n, replace = false)
  points = data[indices, :]

  # Add jitter and apply bounds
  jitter_data!(points, bounds)

  return points
end

# Function to print progress
function print_progress(percent::Int, num_threads::Int)
  print(
    "\rOptimizing <$num_threads threads> [$(repeat('+', percent ÷ 5))$(repeat(' ', 20 - percent ÷ 5))] $percent%",
  )
end

"""
    compute_support_points(n::Int, p::Int, data::Matrix{Float64},
                          subsample_size::Int; max_iterations::Int=500,
                          tolerance::Float64=1e-10, n_threads::Int=Threads.nthreads(),
                          weights::Vector{Float64}=ones(length(axes(data, 1))),
                          use_stochastic::Bool=false)

Compute support points for optimal data representation using iterative optimization.

# Arguments
- `n`: Number of support points to compute
- `p`: Number of dimensions
- `data`: Input data matrix
- `subsample_size`: Size of subsample for stochastic optimization
- `max_iterations`: Maximum number of iterations
- `tolerance`: Convergence tolerance
- `n_threads`: Number of threads for parallel computation
- `weights`: Weights for data points
- `use_stochastic`: Whether to use stochastic optimization

# Returns
- Matrix of computed support points (n×p)
"""
function compute_support_points(
  n::Int,
  p::Int,
  data::Matrix{Float64},
  subsample_size::Int;
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  n_threads::Int = Threads.nthreads(),
  weights::Vector{Float64} = ones(length(axes(data, 1))),
  use_stochastic::Bool = false,
)
  n_data = length(axes(data, 1))

  # Compute bounds
  bounds = compute_bounds(data)

  # Handle duplicates by adding jitter
  data_copy = copy(data)
  if length(unique(eachrow(data))) < n_data
    jitter_data!(data_copy, bounds)
  end

  # Initialize support points
  points = initialize_support_points(n, p, data_copy, bounds)

  # Initialize parameters
  running_const = zeros(n)
  n0 = Float64(n * 0.2)  # Regularization parameter

  iteration = 0
  converged = false
  percent_complete = 0

  while !converged && iteration < max_iterations
    iteration += 1

    # Show progress
    percent = round(Int, 100 * iteration / max_iterations)
    if percent > percent_complete
      print_progress(percent, n_threads)
      percent_complete = percent
    end

    # Store previous points for convergence check
    prev_points = copy(points)

    # Choose subsample
    if use_stochastic && subsample_size < n_data
      subsample_indices = sample(1:n_data, subsample_size, replace = false)
    else
      subsample_indices = 1:n_data
    end

    subsample_data = data_copy[subsample_indices, :]
    subsample_weights = weights[subsample_indices]
    n_subsample = length(subsample_indices)

    # Update support points
    current_const = zeros(n)
    new_points = similar(points)

    @threads for m = 1:n
      xprime = zeros(p)

      # Repulsion from other support points
      for o = 1:n
        if o != m
          diff = points[m, :] - points[o, :]
          dist = norm(diff) + eps(Float64)
          xprime .+= diff ./ dist
        end
      end

      # Scale by ratio
      xprime .*= (n_subsample / n)

      # Attraction to data points
      for i = 1:n_subsample
        diff = subsample_data[i, :] - points[m, :]
        dist = norm(diff) + eps(Float64)

        current_const[m] += subsample_weights[i] / dist
        xprime .+= subsample_weights[i] .* subsample_data[i, :] ./ dist
      end

      # Update using running average
      alpha = n0 / (iteration + n0)
      denom = (1 - alpha) * running_const[m] + alpha * current_const[m]

      if denom > 0
        xprime = ((1 - alpha) * running_const[m] * points[m, :] + alpha * xprime) ./ denom
      else
        xprime = points[m, :]
      end

      # Apply bounds
      for j = 1:p
        xprime[j] = clamp(xprime[j], bounds[j, 1], bounds[j, 2])
      end

      new_points[m, :] = xprime
    end

    # Update points and running constants
    points .= new_points
    alpha = n0 / (iteration + n0)
    running_const = (1 - alpha) .* running_const .+ alpha .* current_const

    # Check convergence
    max_diff = 0.0
    for i = 1:n
      diff = norm(points[i, :] - prev_points[i, :])^2
      max_diff = max(max_diff, diff)
    end

    if max_diff < tolerance
      converged = true
      println("\nTolerance level reached.")
    end
  end

  if !converged
    println("\nMaximum iterations reached.")
  else
    println()
  end

  return points
end
