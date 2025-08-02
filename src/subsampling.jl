"""
Nearest neighbor subsampling for SPlit.jl
"""

using LinearAlgebra
using Distances

"""
    find_nearest_neighbors(data::Matrix{Float64}, support_points::Matrix{Float64};
                           metric::PreMetric=Euclidean())

Find the nearest support point for each data point using specified distance metric.

# Arguments
- `data`: Data matrix (n×p)
- `support_points`: Support points matrix (k×p)
- `metric`: Distance metric to use (default: Euclidean())

# Returns
- Vector of indices indicating which support point each data point is closest to
"""
function find_nearest_neighbors(
  data::Matrix{Float64},
  support_points::Matrix{Float64};
  metric::PreMetric = Euclidean(),
)
  n_data = length(axes(data, 1))
  n_support = length(axes(support_points, 1))

  nearest_indices = Vector{Int}(undef, n_data)

  for i = 1:n_data
    min_dist = Inf
    nearest_idx = 1

    for j = 1:n_support
      dist = metric(data[i, :], support_points[j, :])
      if dist < min_dist
        min_dist = dist
        nearest_idx = j
      end
    end

    nearest_indices[i] = nearest_idx
  end

  return nearest_indices
end

"""
    subsample_by_support_points(data::Matrix{Float64}, support_points::Matrix{Float64};
                               metric::PreMetric=Euclidean())

Perform subsampling by finding the nearest data point to each support point.
This implements a greedy approach where each support point "claims" its nearest data point,
with removal to ensure no data point is selected twice.

# Arguments
- `data`: Data matrix (n×p)
- `support_points`: Support points matrix (k×p)
- `metric`: Distance metric to use (default: Euclidean())

# Returns
- Vector of indices of the selected data points (length k)
"""
function subsample_by_support_points(
  data::Matrix{Float64},
  support_points::Matrix{Float64};
  metric::PreMetric = Euclidean(),
)
  n_data = length(axes(data, 1))
  n_support = length(axes(support_points, 1))

  selected_indices = Vector{Int}()
  available_indices = Set(1:n_data)

  # For each support point, find its nearest available data point
  for j = 1:n_support
    min_dist = Inf
    nearest_idx = -1

    for i in available_indices
      dist = metric(data[i, :], support_points[j, :])
      if dist < min_dist
        min_dist = dist
        nearest_idx = i
      end
    end

    if nearest_idx != -1
      push!(selected_indices, nearest_idx)
      delete!(available_indices, nearest_idx)
    end
  end

  return selected_indices
end

"""
    subsample_indices(data::Matrix{Float64}, support_points::Matrix{Float64};
                     metric::PreMetric=Euclidean())

Main subsampling function that returns indices of data points corresponding to support points.
This is the Julia equivalent of the `subsample` function from the R package.

# Arguments
- `data`: Data matrix (n×p)
- `support_points`: Support points matrix (k×p) where k ≤ n
- `metric`: Distance metric to use (default: Euclidean())

# Returns
- Vector of indices of the subsampled data points
"""
function subsample_indices(
  data::Matrix{Float64},
  support_points::Matrix{Float64};
  metric::PreMetric = Euclidean(),
)
  if length(axes(data, 2)) != length(axes(support_points, 2))
    throw(ArgumentError("Data and support points must have the same number of dimensions"))
  end

  if length(axes(support_points, 1)) > length(axes(data, 1))
    throw(ArgumentError("Cannot have more support points than data points"))
  end

  return subsample_by_support_points(data, support_points; metric = metric)
end
