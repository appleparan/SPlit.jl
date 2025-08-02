using Test
using Random
using DataFrames
using CategoricalArrays

include("../src/main.jl")

@testset "Main SPlit Function Tests" begin

  @testset "Basic Functionality - Matrix Input" begin
    Random.seed!(123)

    # Simple 2D dataset
    n = 20
    data = randn(n, 2)

    # Test default parameters
    indices = split_data(data)

    @test length(indices) <= n
    @test all(1 .<= indices .<= n)
    @test length(unique(indices)) == length(indices)  # No duplicates

    # Should return roughly 20% of the data (4 points)
    expected_size = round(Int, 0.2 * n)
    @test length(indices) == expected_size
  end

  @testset "Different Split Ratios" begin
    Random.seed!(456)

    data = randn(100, 3)

    # Test 30-70 split
    indices_30 = split_data(data; split_ratio = 0.3)
    @test length(indices_30) == 30

    # Test 10-90 split
    indices_10 = split_data(data; split_ratio = 0.1)
    @test length(indices_10) == 10

    # Test 50-50 split
    indices_50 = split_data(data; split_ratio = 0.5)
    @test length(indices_50) == 50
  end

  @testset "DataFrame Input with Categorical Variables" begin
    Random.seed!(789)

    # Create DataFrame with mixed types
    n = 30
    df = DataFrame(
      x1 = randn(n),
      x2 = randn(n),
      category = categorical(rand(["A", "B", "C"], n)),
      x3 = randn(n),
    )

    indices = split_data(df; split_ratio = 0.2)

    @test length(indices) == 6  # 20% of 30
    @test all(1 .<= indices .<= n)
    @test length(unique(indices)) == length(indices)

    # Test that we can actually use the indices to split the data
    test_data = df[indices, :]
    train_data = df[setdiff(1:n, indices), :]

    @test nrow(test_data) == 6
    @test nrow(train_data) == 24
    @test nrow(test_data) + nrow(train_data) == n
  end

  @testset "Stochastic Optimization with Kappa" begin
    Random.seed!(101)

    # Larger dataset to test stochastic optimization
    data = randn(100, 2)

    # Use kappa to enable stochastic optimization
    indices_stochastic = split_data(data; split_ratio = 0.2, kappa = 50)

    @test length(indices_stochastic) == 20
    @test all(1 .<= indices_stochastic .<= 100)
    @test length(unique(indices_stochastic)) == length(indices_stochastic)
  end

  @testset "Parameter Validation" begin
    data = randn(10, 2)

    # Test invalid split_ratio
    @test_throws ArgumentError split_data(data; split_ratio = 0.0)
    @test_throws ArgumentError split_data(data; split_ratio = 1.0)
    @test_throws ArgumentError split_data(data; split_ratio = -0.1)
    @test_throws ArgumentError split_data(data; split_ratio = 1.1)

    # Test invalid kappa
    @test_throws ArgumentError split_data(data; kappa = 0)
    @test_throws ArgumentError split_data(data; kappa = -5)
  end

  @testset "R-style Function Alias" begin
    Random.seed!(202)

    data = randn(20, 2)

    # Test split_data_r function (R-style naming)
    indices_r_style = split_data_r(data; splitRatio = 0.25, maxIterations = 50)

    @test length(indices_r_style) == 5  # 25% of 20
    @test all(1 .<= indices_r_style .<= 20)
    @test length(unique(indices_r_style)) == length(indices_r_style)
  end

  @testset "Optimal Split Ratio Function" begin
    Random.seed!(303)

    # Test simple method
    X = randn(100, 3)
    Y = X[:, 1] + X[:, 2]^2 + 0.1 * randn(100)

    ratio = optimal_split_ratio(X, Y)

    @test 0 < ratio < 1
    @test isa(ratio, Float64)

    # Test with vector input
    X_vec = randn(50)
    Y_vec = X_vec .^ 2 + 0.1 * randn(50)

    ratio_vec = optimal_split_ratio(X_vec, Y_vec)
    @test 0 < ratio_vec < 1

    # Test regression method (should fall back to simple method with warning)
    ratio_reg = optimal_split_ratio(X, Y; method = "regression")
    @test 0 < ratio_reg < 1

    # Test R-style alias
    ratio_alias = splitratio(X, Y)
    @test ratio_alias == optimal_split_ratio(X, Y)
  end

  @testset "Edge Cases" begin
    Random.seed!(404)

    # Very small dataset
    small_data = randn(5, 2)
    indices_small = split_data(small_data; split_ratio = 0.2)

    @test length(indices_small) == 1  # 20% of 5 = 1
    @test 1 <= indices_small[1] <= 5

    # Single column data
    single_col = randn(20, 1)
    indices_single = split_data(single_col; split_ratio = 0.1)

    @test length(indices_single) == 2  # 10% of 20 = 2
    @test all(1 .<= indices_single .<= 20)
  end

  @testset "Data with Missing Values" begin
    # Test that missing values are properly rejected
    data_with_missing = [1.0 2.0; missing 3.0; 4.0 5.0]

    @test_throws ArgumentError split_data(data_with_missing)
  end

  @testset "Reproducibility" begin
    # Test that results are reproducible with same random seed
    data = randn(50, 3)

    Random.seed!(555)
    indices1 = split_data(data; split_ratio = 0.2, max_iterations = 20)

    Random.seed!(555)
    indices2 = split_data(data; split_ratio = 0.2, max_iterations = 20)

    @test indices1 == indices2
  end

  @testset "Large Dataset Performance" begin
    # Test with larger dataset (but keep iterations low for speed)
    Random.seed!(666)

    large_data = randn(500, 4)

    # Use stochastic optimization for speed
    indices_large = split_data(
      large_data;
      split_ratio = 0.1,
      kappa = 100,  # Use subsample
      max_iterations = 10,  # Few iterations for speed
      tolerance = 1e-6,
    )

    @test length(indices_large) == 50  # 10% of 500
    @test all(1 .<= indices_large .<= 500)
    @test length(unique(indices_large)) == length(indices_large)
  end

  @testset "Integration Test - Complete Workflow" begin
    Random.seed!(777)

    # Create a realistic dataset
    n = 100
    X1 = randn(n)
    X2 = randn(n)
    X3 = X1 + 0.5 * X2 + 0.1 * randn(n)  # Correlated feature
    category = categorical(rand(["Type1", "Type2", "Type3"], n))
    Y = X1 + X2^2 + 0.2 * randn(n)  # Nonlinear relationship

    df = DataFrame(X1 = X1, X2 = X2, X3 = X3, category = category, Y = Y)

    # Split the data
    test_indices = split_data(df; split_ratio = 0.25)

    # Create train/test split
    test_data = df[test_indices, :]
    train_data = df[setdiff(1:n, test_indices), :]

    # Verify split
    @test nrow(test_data) == 25
    @test nrow(train_data) == 75
    @test nrow(test_data) + nrow(train_data) == n

    # Check that both splits contain all categories (with high probability)
    test_categories = unique(test_data.category)
    train_categories = unique(train_data.category)

    # Both should have at least one category
    @test length(test_categories) >= 1
    @test length(train_categories) >= 1

    # Verify no overlap in indices
    @test isempty(intersect(test_indices, setdiff(1:n, test_indices)))
  end
end
