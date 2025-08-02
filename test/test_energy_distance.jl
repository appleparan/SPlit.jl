using Test
using Random
using LinearAlgebra
using Distances
using SPlit

@testset "Energy Distance Tests" begin

  @testset "EnergyDistance Construction" begin
    # Test construction with different metrics
    ed_euclidean = EnergyDistance(Euclidean())
    @test ed_euclidean.metric isa Euclidean

    ed_manhattan = EnergyDistance(Cityblock())
    @test ed_manhattan.metric isa Cityblock
  end

  @testset "Pairwise Distance Computation" begin
    # Test pairwise distances between matrices
    X = [1.0 2.0; 3.0 4.0]  # 2x2 matrix (2 dimensions, 2 samples)
    Y = [0.0 1.0; 1.0 2.0]  # 2x2 matrix (2 dimensions, 2 samples)

    metric = Euclidean()
    distances_XY = compute_pairwise_distances(metric, X, Y)

    @test size(distances_XY) == (2, 2)
    # Check specific distances
    @test distances_XY[1, 1] ≈ euclidean([1.0, 3.0], [0.0, 1.0])
    @test distances_XY[1, 2] ≈ euclidean([1.0, 3.0], [1.0, 2.0])

    # Test pairwise distances within matrix
    distances_XX = compute_pairwise_distances(metric, X)
    @test size(distances_XX) == (2, 2)
    @test distances_XX[1, 1] == 0.0  # Distance to self
    @test distances_XX[2, 2] == 0.0  # Distance to self
    @test distances_XX[1, 2] == distances_XX[2, 1]  # Symmetry
  end

  @testset "Energy Distance - Matrix Input" begin
    Random.seed!(123)

    # Create simple test data
    X = randn(3, 10)  # 3 dimensions, 10 samples
    Y = randn(3, 8)   # 3 dimensions, 8 samples

    ed = EnergyDistance(Euclidean())
    distance = ed(X, Y)

    @test isa(distance, Float64)
    @test distance >= 0  # Energy distance should be non-negative

    # Test with identical samples (should be zero)
    X_same = randn(2, 5)
    distance_same = ed(X_same, X_same)
    @test distance_same ≈ 0.0 atol = 1e-12
  end

  @testset "Energy Distance - Vector Input" begin
    Random.seed!(456)

    # Test 1D vectors
    x = randn(20)
    y = randn(15)

    ed = EnergyDistance(Euclidean())
    distance = ed(x, y)

    @test isa(distance, Float64)
    @test distance >= 0

    # Test with identical vectors
    distance_same = ed(x, x)
    @test distance_same ≈ 0.0 atol = 1e-12
  end

  @testset "Sample Without Replacement" begin
    Random.seed!(789)

    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Test normal sampling
    sampled = sample_without_replacement(X, 5)
    @test length(sampled) == 5
    @test length(unique(sampled)) == 5  # No duplicates
    @test all(s in X for s in sampled)  # All samples from original

    # Test sampling all elements
    sampled_all = sample_without_replacement(X, 10)
    @test length(sampled_all) == 10
    @test Set(sampled_all) == Set(X)

    # Test error for over-sampling
    @test_throws ArgumentError sample_without_replacement(X, 15)
  end

  @testset "Energy Distance with Sampling" begin
    Random.seed!(101)

    # Create larger vectors for sampling test
    P = randn(100)
    Q = randn(80)

    ed = EnergyDistance(Euclidean())

    # Test sampling version
    distance_sampled = ed(P, Q, 20)
    @test isa(distance_sampled, Float64)
    @test distance_sampled >= 0
  end

  @testset "Convenience Functions" begin
    Random.seed!(202)

    # Test matrix version (each row is a sample)
    X = randn(10, 2)  # 10 samples, 2 dimensions
    Y = randn(8, 2)   # 8 samples, 2 dimensions

    distance1 = energy_distance(X, Y)
    distance2 = energy_distance(X, Y; metric = Euclidean())

    @test distance1 ≈ distance2
    @test distance1 >= 0

    # Test vector version
    x = randn(15)
    y = randn(12)

    distance_vec = energy_distance(x, y)
    @test isa(distance_vec, Float64)
    @test distance_vec >= 0
  end

  @testset "Different Metrics" begin
    Random.seed!(303)

    X = randn(2, 5)
    Y = randn(2, 5)

    # Test with different metrics
    dist_euclidean = energy_distance(X, Y; metric = Euclidean())
    dist_manhattan = energy_distance(X, Y; metric = Cityblock())

    @test dist_euclidean >= 0
    @test dist_manhattan >= 0
    # Different metrics should generally give different results
    # (though they could be equal by chance)
  end

  @testset "Edge Cases" begin
    # Test with single sample
    X_single = randn(3, 1)
    Y_single = randn(3, 1)

    distance_single = energy_distance(X_single, Y_single)
    @test isa(distance_single, Float64)

    # Test with very small differences
    X_base = randn(2, 5)
    X_close = X_base .+ 1e-10 * randn(length(axes(X_base, 1)), length(axes(X_base, 2)))

    distance_close = energy_distance(X_base, X_close)
    @test distance_close >= -eps(Float64)
    @test distance_close < 1e-8  # Should be very small
  end

  @testset "Mathematical Properties" begin
    Random.seed!(404)

    # Test symmetry: d(X,Y) = d(Y,X)
    X = randn(8, 2)  # 8 samples, 2 dimensions
    Y = randn(6, 2)  # 6 samples, 2 dimensions

    dist_XY = energy_distance(X, Y)
    dist_YX = energy_distance(Y, X)

    @test dist_XY ≈ dist_YX atol = 1e-12

    # Test self-distance is zero: d(X,X) = 0
    dist_XX = energy_distance(X, X)
    @test dist_XX ≈ 0.0 atol = 1e-12
  end
end
