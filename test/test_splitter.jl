using Test
using SPlit
using Random
using DataFrames

@testset "SupportPointSplitter" begin
  @testset "constructor validation" begin
    @test_throws ArgumentError SupportPointSplitter(ratio = 0.0)
    @test_throws ArgumentError SupportPointSplitter(ratio = 1.0)
    @test_throws ArgumentError SupportPointSplitter(max_iterations = 0)
    @test_throws ArgumentError SupportPointSplitter(tolerance = 0.0)
    @test_throws ArgumentError SupportPointSplitter(n_threads = 0)
    @test_throws ArgumentError SupportPointSplitter(kappa = 0)
    @test SupportPointSplitter().kernel isa EnergyKernel
  end

  @testset "accepts non-Float64 Real ratio and tolerance" begin
    s = SupportPointSplitter(ratio = 1 // 5, tolerance = 1.0f-8)
    @test s.ratio === 0.2
    @test s.tolerance === Float64(1.0f-8)
  end

  @testset "matrix split: sizes, partition, honest report" begin
    rng = MersenneTwister(1)
    data = randn(rng, 200, 3)
    s = SupportPointSplitter(ratio = 0.2, max_iterations = 100, rng = MersenneTwister(2))
    r = datasplit(s, data)
    @test length(test_indices(r)) == 40
    @test length(train_indices(r)) == 160
    @test sort(vcat(train_indices(r), test_indices(r))) == collect(1:200)
    @test r.iterations <= 100
    @test r.converged isa Bool
  end

  @testset "ratio > 0.5 puts the larger side in test" begin
    data = randn(MersenneTwister(3), 100, 2)
    s = SupportPointSplitter(ratio = 0.8, max_iterations = 50, rng = MersenneTwister(4))
    r = datasplit(s, data)
    @test length(test_indices(r)) == 80
    @test length(train_indices(r)) == 20
  end

  @testset "DataFrame with categoricals" begin
    df = DataFrame(x = randn(MersenneTwister(5), 90), g = repeat(["a", "b", "c"], 30))
    s = SupportPointSplitter(max_iterations = 50, rng = MersenneTwister(6))
    r = datasplit(s, df)
    @test length(test_indices(r)) == 18
    train_view = df[r, :train]
    @test nrow(train_view) == 72
  end

  @testset "vector input" begin
    v = randn(MersenneTwister(7), 50)
    r = datasplit(SupportPointSplitter(max_iterations = 50, rng = MersenneTwister(8)), v)
    @test length(test_indices(r)) == 10
  end

  @testset "reproducible with rng" begin
    data = randn(MersenneTwister(9), 100, 2)
    r1 =
      datasplit(SupportPointSplitter(max_iterations = 60, rng = MersenneTwister(10)), data)
    r2 =
      datasplit(SupportPointSplitter(max_iterations = 60, rng = MersenneTwister(10)), data)
    @test test_indices(r1) == test_indices(r2)
  end

  @testset "iteration and getindex sugar" begin
    data = randn(MersenneTwister(11), 60, 2)
    r =
      datasplit(SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(12)), data)
    train, test = r
    @test train == train_indices(r)
    @test test == test_indices(r)
    @test size(data[r, :train], 1) == length(train)
    @test_throws ArgumentError data[r, :validation]
  end

  @testset "legacy API is gone" begin
    @test !isdefined(SPlit, :split_data)
    @test !isdefined(SPlit, :format_data)
    @test !isdefined(SPlit, :EnergyDistance)
  end
end
