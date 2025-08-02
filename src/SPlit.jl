module SPlit

# Main functionality
include("main.jl")

# Export main functions
export split_data, split_data_r, optimal_split_ratio, splitratio

# Export utility functions from submodules if needed
export format_data, compute_support_points, subsample_indices

end
