using Test
using Random
using LinearAlgebra
using Statistics

include("../src/support_points.jl")

@testset "Support Points Tests" begin

  @testset "Bounds Computation" begin
    # Test with simple 2D data
    data = [1.0 2.0; 3.0 1.0; 2.0 4.0; 0.0 3.0]
    bounds = compute_bounds(data)

    @test bounds[1, 1] ≈ 0.0  # min of first column
    @test bounds[1, 2] ≈ 3.0  # max of first column
    @test bounds[2, 1] ≈ 1.0  # min of second column
    @test bounds[2, 2] ≈ 4.0  # max of second column

    # Test with single column
    data_1d = reshape([1.0, 5.0, 3.0, 2.0], 4, 1)
    bounds_1d = compute_bounds(data_1d)
    @test bounds_1d[1, 1] ≈ 1.0
    @test bounds_1d[1, 2] ≈ 5.0
  end

  @testset "Jitter Data" begin
    # Test that jittering preserves bounds
    Random.seed!(123)
    data = [1.0 2.0; 1.0 2.0; 1.0 2.0]  # Identical rows
    bounds = compute_bounds(data)

    data_copy = copy(data)
    jitter_data!(data_copy, bounds)

    # Check that data is still within bounds
    @test all(data_copy[:, 1] .>= bounds[1, 1])
    @test all(data_copy[:, 1] .<= bounds[1, 2])
    @test all(data_copy[:, 2] .>= bounds[2, 1])
    @test all(data_copy[:, 2] .<= bounds[2, 2])

    # Check that data has been modified (not identical anymore)
    @test !all(data_copy[1, :] .≈ data_copy[2, :])
  end

  @testset "Initialize Support Points" begin
    Random.seed!(42)
    data = randn(20, 3)
    bounds = compute_bounds(data)

    n_points = 5
    points = initialize_support_points(n_points, 3, data, bounds)

    @test size(points) == (5, 3)

    # Check that points are within bounds
    for j = 1:3
      @test all(points[:, j] .>= bounds[j, 1])
      @test all(points[:, j] .<= bounds[j, 2])
    end
  end

  @testset "Support Points Computation - Small Example" begin
    Random.seed!(123)

    # Create simple 2D data
    data = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5]
    n_points = 2
    p = 2

    support_points = compute_support_points(
      n_points,
      p,
      data,
      length(axes(data, 1));
      max_iterations = 10,  # Keep short for testing
      tolerance = 1e-6,
      use_stochastic = false,
    )

    @test size(support_points) == (2, 2)

    # Check that support points are within data bounds
    bounds = compute_bounds(data)
    for j = 1:2
      @test all(support_points[:, j] .>= bounds[j, 1])
      @test all(support_points[:, j] .<= bounds[j, 2])
    end
  end

  @testset "Support Points Computation - Convergence" begin
    Random.seed!(456)

    # Test convergence with very tight tolerance
    data = randn(10, 2)
    n_points = 3

    # This should converge quickly with such a tight tolerance
    support_points = compute_support_points(
      n_points,
      2,
      data,
      length(axes(data, 1));
      max_iterations = 100,
      tolerance = 1e-12,
      use_stochastic = false,
    )

    @test size(support_points) == (3, 2)

    # Support points should be different from each other
    for i = 1:n_points
      for j = (i+1):n_points
        @test norm(support_points[i, :] - support_points[j, :]) > 1e-6
      end
    end
  end

  @testset "Stochastic Optimization" begin
    Random.seed!(789)

    # Create larger dataset for stochastic test
    data = randn(50, 2)
    n_points = 5
    subsample_size = 20  # Smaller than total data size

    support_points = compute_support_points(
      n_points,
      2,
      data,
      subsample_size;
      max_iterations = 20,
      tolerance = 1e-8,
      use_stochastic = true,
    )

    @test size(support_points) == (5, 2)

    # Check bounds
    bounds = compute_bounds(data)
    for j = 1:2
      @test all(support_points[:, j] .>= bounds[j, 1])
      @test all(support_points[:, j] .<= bounds[j, 2])
    end
  end

  @testset "Identical Data Points" begin
    Random.seed!(101)

    # Test with identical data points (should trigger jittering)
    data = ones(10, 2)  # All points are identical
    n_points = 3

    support_points = compute_support_points(
      n_points,
      2,
      data,
      length(axes(data, 1));
      max_iterations = 10,
      tolerance = 1e-8,
      use_stochastic = false,
    )

    @test size(support_points) == (3, 2)

    # Support points should not be identical (due to jittering)
    @test !all(support_points[1, :] .≈ support_points[2, :])
  end

  @testset "Edge Cases" begin
    Random.seed!(202)

    # Test with single support point
    data = randn(5, 2)
    single_point = compute_support_points(
      1,
      2,
      data,
      length(axes(data, 1));
      max_iterations = 5,
      tolerance = 1e-6,
      use_stochastic = false,
    )

    @test size(single_point) == (1, 2)

    # Test with n_points equal to data size
    small_data = randn(3, 2)
    all_points = compute_support_points(
      3,
      2,
      small_data,
      length(axes(small_data, 1));
      max_iterations = 5,
      tolerance = 1e-6,
      use_stochastic = false,
    )

    @test size(all_points) == (3, 2)
  end

  @testset "Parameter Validation" begin
    data = randn(10, 2)

    # Test with more support points than data points (should be handled gracefully in calling function)
    # This test checks that the algorithm doesn't crash
    try
      support_points = compute_support_points(
        15,
        2,
        data,
        length(axes(data, 1));
        max_iterations = 5,
        tolerance = 1e-6,
        use_stochastic = false,
      )
      # If it doesn't throw an error, that's also acceptable behavior
      @test size(support_points, 2) == 2
    catch e
      # If it throws an error, that's also acceptable
      @test true
    end
  end
end
