module SPlit

# Main functionality
include("main.jl")
include("energy_distance.jl")
include("split_quality.jl")

# Export main functions
export split_data, split_data_r, optimal_split_ratio, splitratio

# Export utility functions from submodules if needed
export format_data, compute_support_points, subsample_indices

# Export data preprocessing functions
export helmert_contrasts, count_encoded_columns, encode_categorical!

# Export support point functions
export compute_bounds, jitter_data!, initialize_support_points

# Export subsampling functions
export find_nearest_neighbors, subsample_by_support_points

# Export energy distance functions
export EnergyDistance,
  energy_distance, compute_pairwise_distances, sample_without_replacement

# Export split quality assessment functions
export evaluate_split_quality, compare_split_methods, split_data_with_quality

end
