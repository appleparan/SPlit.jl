using SPlit
using Test

@testset "SPlit.jl" begin
  # Test new Julia-native API first
  include("test_julia_native_api.jl")

  # Legacy API tests
  include("test_data_preprocessing.jl")
  include("test_support_points.jl")
  include("test_subsampling.jl")
  include("test_main.jl")

  # Test energy distance functions if they exist
  if isfile("test_energy_distance.jl")
    include("test_energy_distance.jl")
  end
end
