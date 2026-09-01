using Test
using SPlit
using Random
using DataFrames

@testset "optimal_split_ratio" begin
  @testset "gamma formula for known p" begin
    # 3 numeric predictors ⇒ p = 3 + 1 = 4 ⇒ γ = 1/(√4 + 1) = 1/3
    X = randn(MersenneTwister(1), 100, 3)
    y = randn(MersenneTwister(2), 100)
    @test isapprox(optimal_split_ratio(X, y), 1 / 3; atol = 1e-12)
    # 1 predictor ⇒ p = 2 ⇒ γ = 1/(√2 + 1)
    @test isapprox(
      optimal_split_ratio(randn(MersenneTwister(3), 50), y[1:50]),
      1 / (sqrt(2) + 1);
      atol = 1e-12,
    )
  end

  @testset "categorical predictors count encoded columns" begin
    df = DataFrame(x = randn(MersenneTwister(4), 60), g = repeat(["a", "b", "c"], 20))
    # 1 numeric + 2 Helmert columns + intercept ⇒ p = 4
    @test isapprox(optimal_split_ratio(df, randn(60)), 1 / 3; atol = 1e-12)
  end

  @testset "regression method errors until implemented" begin
    X = randn(MersenneTwister(5), 40, 2)
    @test_throws ErrorException optimal_split_ratio(X, randn(40); method = :regression)
  end

  @testset "unknown method errors" begin
    @test_throws ArgumentError optimal_split_ratio(randn(10, 2), randn(10); method = :other)
  end

  @testset "observation-count mismatch errors" begin
    X = randn(MersenneTwister(6), 10, 2)
    @test_throws ArgumentError optimal_split_ratio(X, randn(9))
  end
end
