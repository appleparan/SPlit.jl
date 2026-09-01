using Test
using SPlit
using Random
using Statistics

@testset "energydistance" begin
  @testset "identical samples give zero" begin
    X = [0.0 0.0; 1.0 1.0; 2.0 0.5]
    @test isapprox(energydistance(X, copy(X)), 0.0; atol = 1e-12)
  end

  @testset "hand-computed 1D values" begin
    # X = {0}, Y = {1}: 2·E|X−Y| − E|X−X'| − E|Y−Y'| = 2·1 − 0 − 0 = 2
    @test isapprox(energydistance([0.0], [1.0]), 2.0; atol = 1e-12)
    # X = {0,1}, Y = {0,1}: identical empirical distributions ⇒ 0
    @test isapprox(energydistance([0.0, 1.0], [0.0, 1.0]), 0.0; atol = 1e-12)
  end

  @testset "nonnegative and sensitive to shift" begin
    rng = MersenneTwister(7)
    X = randn(rng, 200, 3)
    Y = randn(rng, 180, 3)
    Yshift = Y .+ 1.5
    @test energydistance(X, Y) >= -1e-12
    @test energydistance(X, Yshift) > energydistance(X, Y)
  end

  @testset "block accumulation matches naive computation" begin
    rng = MersenneTwister(11)
    X = randn(rng, 130, 2)   # > default block only if block small; force blocks
    Y = randn(rng, 90, 2)
    exact = energydistance(X, Y)
    blocked = SPlit._exact_energydistance(X, Y; block = 17)
    @test isapprox(exact, blocked; atol = 1e-10)
  end

  @testset "subsampled estimate agrees with exact" begin
    rng = MersenneTwister(42)
    X = randn(rng, 600, 2)
    Y = randn(rng, 600, 2) .+ 0.5
    exact = energydistance(X, Y)
    est = energydistance(X, Y; subsample = 150, repeats = 30, rng = MersenneTwister(1))
    @test isapprox(est, exact; rtol = 0.25)
  end

  @testset "dimension mismatch errors" begin
    @test_throws ArgumentError energydistance(randn(5, 2), randn(5, 3))
  end
end
