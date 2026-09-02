using Test
using SPlit
using DataFrames
using CategoricalArrays
using Statistics

@testset "preprocess" begin
  @testset "matrix: standardizes columns" begin
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0]
    P = SPlit.preprocess(X)
    @test size(P) == (4, 2)
    for j in axes(P, 2)
      @test isapprox(mean(P[:, j]), 0.0; atol = 1e-12)
      @test isapprox(std(P[:, j]), 1.0; atol = 1e-12)
    end
  end

  @testset "matrix: drops constant columns" begin
    X = [1.0 5.0; 2.0 5.0; 3.0 5.0]
    P = SPlit.preprocess(X)
    @test size(P) == (3, 1)
  end

  @testset "matrix: all-constant errors" begin
    X = fill(2.0, 4, 2)
    @test_throws ArgumentError SPlit.preprocess(X)
  end

  @testset "missing values error" begin
    X = Matrix{Union{Float64,Missing}}([1.0 2.0; missing 3.0])
    @test_throws ArgumentError SPlit.preprocess(X)
    df = DataFrame(a = [1.0, missing, 3.0])
    @test_throws ArgumentError SPlit.preprocess(df)
  end

  @testset "vector input" begin
    P = SPlit.preprocess([1.0, 2.0, 3.0, 4.0])
    @test size(P) == (4, 1)
    @test isapprox(std(P[:, 1]), 1.0; atol = 1e-12)
  end

  @testset "dataframe: numeric + categorical (Helmert)" begin
    df = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], g = ["a", "b", "c", "a", "b", "c"])
    P = SPlit.preprocess(df)
    # 1 numeric + (3 levels - 1) Helmert columns = 3 columns
    @test size(P) == (6, 3)
    for j in axes(P, 2)
      @test isapprox(mean(P[:, j]), 0.0; atol = 1e-12)
    end
    # CategoricalVector behaves the same
    df2 = DataFrame(x = df.x, g = categorical(df.g))
    @test SPlit.preprocess(df2) == P
  end

  @testset "dataframe: unsupported column type errors" begin
    df = DataFrame(a = [1.0, 2.0], b = [[1], [2]])
    @test_throws ArgumentError SPlit.preprocess(df)
  end

  @testset "Union{Missing,T} columns without missing values are accepted" begin
    plain =
      DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], g = ["a", "b", "c", "a", "b", "c"])
    unioned = DataFrame(
      x = Vector{Union{Missing,Float64}}(plain.x),
      g = Vector{Union{Missing,String}}(plain.g),
    )
    @test SPlit.preprocess(unioned) == SPlit.preprocess(plain)
  end

  @testset "canonical Helmert level order is independent of row order" begin
    df = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], g = ["c", "a", "b", "c", "a", "b"])
    P = SPlit.preprocess(df)
    perm = [4, 1, 6, 2, 5, 3]
    shuffled = df[perm, :]
    P_shuffled = SPlit.preprocess(shuffled)
    @test P_shuffled[invperm(perm), :] == P
  end

  @testset "helmert_matrix" begin
    H = SPlit.helmert_matrix(3)
    @test size(H) == (3, 2)
    # each contrast column sums to zero
    @test all(isapprox.(sum(H; dims = 1), 0.0; atol = 1e-12))
    @test size(SPlit.helmert_matrix(1)) == (1, 0)
  end
end

@testset "weighted preprocess" begin
  using Random

  @testset "nothing dispatches to the unweighted method" begin
    data = randn(MersenneTwister(30), 50, 3)
    @test SPlit.preprocess(data, nothing) == SPlit.preprocess(data)
  end

  @testset "weighted mean 0 and weighted variance 1 per column" begin
    rng = MersenneTwister(31)
    data = randn(rng, 80, 3) .* [1.0 5.0 0.1] .+ [2.0 -1.0 0.0]
    w = rand(rng, 80)
    X = SPlit.preprocess(data, w)
    wn = w ./ sum(w)
    for j = 1:3
      μ = sum(wn .* X[:, j])
      σ2 = sum(wn .* (X[:, j] .- μ) .^ 2) / (1 - sum(abs2, wn))
      @test isapprox(μ, 0.0; atol = 1e-12)
      @test isapprox(σ2, 1.0; atol = 1e-12)
    end
  end

  @testset "uniform weights match the unweighted result up to rounding" begin
    data = randn(MersenneTwister(32), 60, 2)
    M = SPlit._encode(data)
    X_w = SPlit._standardize!(copy(M), fill(1 / 60, 60))
    @test isapprox(X_w, SPlit.preprocess(data); atol = 1e-12)
  end

  @testset "constant weight vector is treated as nothing and matches exactly" begin
    data = randn(MersenneTwister(32), 60, 2)
    @test SPlit.preprocess(data, ones(60)) == SPlit.preprocess(data)
  end

  @testset "all weight on one row errors" begin
    w = zeros(10)
    w[3] = 1.0
    @test_throws ArgumentError SPlit.preprocess(randn(10, 2), w)
  end

  @testset "column constant on positive-weight rows errors" begin
    @test_throws ArgumentError SPlit.preprocess(
      [5.0 1.0; 5.0 2.0; 7.0 3.0],
      [1.0, 1.0, 0.0],
    )
  end

  @testset "DataFrame with categoricals accepts weights" begin
    df = DataFrame(x = randn(MersenneTwister(33), 30), g = repeat(["a", "b", "c"], 10))
    X = SPlit.preprocess(df, ones(30))
    @test size(X) == (30, 3)
  end

  @testset "wrong length errors" begin
    @test_throws ArgumentError SPlit.preprocess(randn(10, 2), ones(9))
  end
end
