module SPlit

include("kernels.jl")
include("preprocessing.jl")
include("optimizer.jl")
include("kdtree_selection.jl")
include("splitter.jl")
include("quality.jl")
include("ratio.jl")
include("comparison.jl")

export SplitKernel, EnergyKernel, GaussianKernel
export AbstractSplitter, SupportPointSplitter, SplitResult, datasplit
export train_indices, test_indices
export energydistance, mmd, splitquality
export optimal_split_ratio
export compare, SplitComparison, best

end
