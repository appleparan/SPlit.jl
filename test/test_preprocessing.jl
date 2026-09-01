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

  @testset "helmert_matrix" begin
    H = SPlit.helmert_matrix(3)
    @test size(H) == (3, 2)
    # each contrast column sums to zero
    @test all(isapprox.(sum(H; dims = 1), 0.0; atol = 1e-12))
    @test size(SPlit.helmert_matrix(1)) == (1, 0)
  end
end
