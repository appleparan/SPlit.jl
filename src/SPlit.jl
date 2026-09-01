module SPlit

include("kernels.jl")
include("preprocessing.jl")
include("quality.jl")
include("optimizer.jl")
include("kdtree_selection.jl")
include("splitter.jl")

export SplitKernel, EnergyKernel
export SupportPointSplitter, SplitResult, datasplit
export train_indices, test_indices
export energydistance

end
