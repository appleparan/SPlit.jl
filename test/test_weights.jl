using Test
using SPlit

@testset "weight helpers" begin
  @testset "validation" begin
    @test_throws ArgumentError SPlit._check_weights([1.0, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([1.0, -1.0, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([1.0, NaN, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([1.0, Inf, 2.0], 3)
    @test_throws ArgumentError SPlit._check_weights([0.0, 0.0, 0.0], 3)
    @test SPlit._check_weights([0.0, 1.0, 2.0], 3) == [0.0, 1.0, 2.0]
  end

  @testset "normalization" begin
    w = SPlit._normalize_weights([1, 3], 2)
    @test w isa Vector{Float64}
    @test w == [0.25, 0.75]
    @test SPlit._normalize_weights([2.0, 2.0, 2.0], 3) == fill(1 / 3, 3)
    @test SPlit._uniform_weights(4) == fill(0.25, 4)
    @test SPlit._side_weights(nothing, 4) == fill(0.25, 4)
    @test SPlit._side_weights([1, 1, 2], 3) == [0.25, 0.25, 0.5]
  end

  @testset "mean-one scaling keeps uniform weights exactly 1.0" begin
    @test SPlit._mean_one_weights(ones(7)) == ones(7)
    @test all(SPlit._mean_one_weights(fill(0.3, 5)) .== 1.0)
    w = SPlit._mean_one_weights([1.0, 3.0])
    @test w == [0.5, 1.5]
    @test SPlit._mean_one_weights(fill(0.37, 150)) == ones(150)
    @test SPlit._mean_one_weights(SPlit._uniform_weights(7)) == ones(7)
    @test SPlit._mean_one_weights(fill(1 / 9, 9)) == ones(9)
  end
end
