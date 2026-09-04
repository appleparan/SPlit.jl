module SPlit

include("kernels.jl")
include("weights.jl")
include("estimators.jl")
include("preprocessing.jl")
include("optimizer.jl")
include("kdtree_selection.jl")
include("splitter.jl")
include("herding.jl")
include("twinning.jl")
include("multiplet.jl")
include("kernel_thinning.jl")
include("quality.jl")
include("ratio.jl")
include("comparison.jl")

export SplitKernel, EnergyKernel, GaussianKernel
export DiscrepancyEstimator, Exact, Subsample, RandomSlices, RandomFeatures
export AbstractSplitter,
  SupportPointSplitter,
  HerdingSplitter,
  TwinningSplitter,
  KernelThinningSplitter,
  SplitResult,
  datasplit
export selectrows
export multiplet
export train_indices, test_indices
export energydistance, mmd, splitquality
export optimal_split_ratio
export compare, SplitComparison, best

end
