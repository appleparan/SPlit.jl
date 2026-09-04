using SPlit
using Test

@testset "SPlit.jl" begin
  include("test_preprocessing.jl")
  include("test_kernels.jl")
  include("test_weights.jl")
  include("test_estimators.jl")
  include("test_quality.jl")
  include("test_optimizer.jl")
  include("test_kdtree_selection.jl")
  include("test_splitter.jl")
  include("test_herding.jl")
  include("test_twinning.jl")
  include("test_multiplet.jl")
  include("test_kernel_thinning.jl")
  include("test_standardize.jl")
  include("test_time_series_windows.jl")
  include("test_ratio.jl")
  include("test_comparison.jl")
  include("test_properties.jl")
end
