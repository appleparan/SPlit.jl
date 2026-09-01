module SPlit

include("kernels.jl")
include("preprocessing.jl")
include("optimizer.jl")
include("kdtree_selection.jl")
include("splitter.jl")
include("quality.jl")

export SplitKernel, EnergyKernel
export SupportPointSplitter, SplitResult, datasplit
export train_indices, test_indices
export energydistance, splitquality

end
