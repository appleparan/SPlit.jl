module SPlit

# Core type system and interfaces
include("types.jl")
include("interface.jl")
include("comparison.jl")

# Legacy functionality (for backward compatibility)
include("main.jl")
include("energy_distance.jl")
include("split_quality.jl")

# Main Julia-native API
export SplittingMethod, SplittingResult
export SupportPointSplitter, SplitResult
export datasplit, split_with_quality, datasplit_with_quality

# Convenience functions
export compare, best, summary, DefaultSplitters, quick_compare
export ratio, metric, train_indices, test_indices, quality
export SplitComparison

# Legacy API (for backward compatibility)
export split_data, split_data_r, optimal_split_ratio, splitratio

# Utility functions
export format_data, compute_support_points, subsample_indices
export helmert_contrasts, count_encoded_columns, encode_categorical!
export compute_bounds, jitter_data!, initialize_support_points
export find_nearest_neighbors, subsample_by_support_points

# Energy distance functions
export EnergyDistance,
  energy_distance, compute_pairwise_distances, sample_without_replacement

# Legacy split quality functions
export evaluate_split_quality, compare_split_methods, split_data_with_quality

end
