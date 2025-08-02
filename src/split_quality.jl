"""
Quality assessment functions for data splits using energy distance.
"""

using Distances
using Statistics

"""
    evaluate_split_quality(data::Matrix{Float64}, train_indices::Vector{Int},
                          test_indices::Vector{Int}; metric::PreMetric=Euclidean())

Evaluate the quality of a data split using energy distance between train and test sets.
A smaller energy distance indicates more similar distributions (better split quality).

# Arguments
- `data`: Original data matrix
- `train_indices`: Indices of training set
- `test_indices`: Indices of test set
- `metric`: Distance metric to use for energy distance calculation

# Returns
- Energy distance between train and test distributions
"""
function evaluate_split_quality(
  data::Matrix{Float64},
  train_indices::Vector{Int},
  test_indices::Vector{Int};
  metric::PreMetric = Euclidean(),
)
  train_data = data[train_indices, :]
  test_data = data[test_indices, :]

  # Use energy distance to compare distributions
  energy_dist = EnergyDistance(metric)
  return energy_dist(Matrix(train_data'), Matrix(test_data'))
end

"""
    compare_split_methods(data; split_ratio::Float64=0.2, methods::Vector{PreMetric}=PreMetric[])

Compare different distance metrics for data splitting and return quality metrics.

# Arguments
- `data`: Input dataset
- `split_ratio`: Split ratio for comparison
- `methods`: Vector of distance metrics to compare

# Returns
- Dictionary with method names as keys and quality scores as values
"""
function compare_split_methods(
  data;
  split_ratio::Float64 = 0.2,
  methods::Vector{PreMetric} = PreMetric[
    Euclidean(),
    Cityblock(),
    EnergyDistance(Euclidean()),
  ],
)
  results = Dict{String,Float64}()
  n_total = size(data, 1)

  for (i, method) in enumerate(methods)
    method_name = string(typeof(method))

    # Split data using this method
    test_indices =
      split_data(data; split_ratio = split_ratio, metric = method, max_iterations = 50)
    train_indices = setdiff(1:n_total, test_indices)

    # Evaluate quality
    quality =
      evaluate_split_quality(data, train_indices, test_indices; metric = Euclidean())
    results[method_name] = quality

    println("$method_name: Energy distance = $quality")
  end

  return results
end

"""
    split_data_with_quality(data; split_ratio::Float64=0.2, metric::PreMetric=Euclidean(), kwargs...)

Split data and return both indices and quality assessment.

# Returns
- (test_indices, quality_score)
"""
function split_data_with_quality(
  data;
  split_ratio::Float64 = 0.2,
  metric::PreMetric = Euclidean(),
  kwargs...,
)
  test_indices = split_data(data; split_ratio = split_ratio, metric = metric, kwargs...)
  train_indices = setdiff(1:size(data, 1), test_indices)

  quality = evaluate_split_quality(data, train_indices, test_indices; metric = Euclidean())

  return test_indices, quality
end
