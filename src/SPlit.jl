module SPlit

# Main functionality
include("main.jl")
include("energy_distance.jl")

# Export main functions
export split_data, split_data_r, optimal_split_ratio, splitratio

# Export utility functions from submodules if needed
export format_data, compute_support_points, subsample_indices

# Export energy distance functions
export EnergyDistance, energy_distance

end
