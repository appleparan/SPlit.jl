using SPlit
using Test

@testset "SPlit.jl" begin
  # Include all test files
  include("test_data_preprocessing.jl")
  include("test_support_points.jl")
  include("test_subsampling.jl")
  include("test_main.jl")

  # Test energy distance functions if they exist
  if isfile("test_energy_distance.jl")
    include("test_energy_distance.jl")
  end
end
