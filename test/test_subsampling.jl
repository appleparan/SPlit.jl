using Test
using Random
using LinearAlgebra
using SPlit

@testset "Subsampling Tests" begin

  @testset "Find Nearest Neighbors" begin
    # Simple 2D test case
    data = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
    support_points = [0.1 0.1; 0.9 0.9]

    nearest = find_nearest_neighbors(data, support_points)

    @test length(nearest) == 4
    @test nearest[1] == 1  # (0,0) closest to (0.1, 0.1)
    @test nearest[2] == 1  # (1,0) closest to (0.1, 0.1) - actually might be closer to (0.9, 0.9)
    @test nearest[3] == 1  # (0,1) closest to (0.1, 0.1)
    @test nearest[4] == 2  # (1,1) closest to (0.9, 0.9)

    # More precise test
    data2 = [0.0 0.0; 1.0 1.0]
    support_points2 = [0.0 0.0; 1.0 1.0]
    nearest2 = find_nearest_neighbors(data2, support_points2)

    @test nearest2[1] == 1  # Exact match
    @test nearest2[2] == 2  # Exact match
  end

  @testset "Subsample by Support Points" begin
    Random.seed!(123)

    # Create data where we know the expected result
    data = [0.0 0.0; 0.1 0.1; 2.0 2.0; 2.1 2.1; 1.0 1.0]
    support_points = [0.05 0.05; 2.05 2.05]

    indices = subsample_by_support_points(data, support_points)

    @test length(indices) == 2
    @test length(unique(indices)) == 2  # No duplicates

    # Check that we get reasonable indices
    @test all(1 .<= indices .<= 5)

    # The algorithm should pick points closest to support points
    # Support point 1 (0.05, 0.05) should be closest to data points 1 or 2
    # Support point 2 (2.05, 2.05) should be closest to data points 3 or 4

    selected_points = data[indices, :]

    # At least one point should be close to each support point region
    distances_to_first_support = [norm(selected_points[i, :] - [0.05, 0.05]) for i = 1:2]
    distances_to_second_support = [norm(selected_points[i, :] - [2.05, 2.05]) for i = 1:2]

    @test minimum(distances_to_first_support) < 1.0  # At least one point near first support
    @test minimum(distances_to_second_support) < 1.0  # At least one point near second support
  end

  @testset "Subsample Indices - Main Function" begin
    Random.seed!(456)

    # Test basic functionality
    data = randn(20, 3)
    support_points = randn(5, 3)

    indices = subsample_indices(data, support_points)

    @test length(indices) == 5
    @test all(1 .<= indices .<= 20)
    @test length(unique(indices)) == 5  # All indices should be unique

    # Test dimension mismatch error
    data_wrong = randn(20, 2)
    support_points_wrong = randn(5, 3)

    @test_throws ArgumentError subsample_indices(data_wrong, support_points_wrong)

    # Test too many support points error
    data_small = randn(3, 2)
    support_points_too_many = randn(5, 2)

    @test_throws ArgumentError subsample_indices(data_small, support_points_too_many)
  end

  @testset "Edge Cases" begin
    # Test with single support point
    data = randn(10, 2)
    single_support = randn(1, 2)

    indices = subsample_indices(data, single_support)
    @test length(indices) == 1
    @test 1 <= indices[1] <= 10

    # Test with support points equal to data size
    small_data = randn(3, 2)
    many_support = randn(3, 2)

    indices_many = subsample_indices(small_data, many_support)
    @test length(indices_many) == 3
    @test Set(indices_many) == Set(1:3)  # Should select all data points
  end

  @testset "1D Data" begin
    # Test with 1D data
    data_1d = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1)
    support_1d = reshape([1.5, 4.5], 2, 1)

    indices_1d = subsample_indices(data_1d, support_1d)

    @test length(indices_1d) == 2
    @test all(1 .<= indices_1d .<= 5)
    @test length(unique(indices_1d)) == 2

    # The closest points to 1.5 and 4.5 should be selected
    # 1.5 is closest to 1.0 (index 1) or 2.0 (index 2)
    # 4.5 is closest to 4.0 (index 4) or 5.0 (index 5)
    selected_values = data_1d[indices_1d, 1]
    @test any(selected_values .<= 2.0)  # At least one small value
    @test any(selected_values .>= 4.0)  # At least one large value
  end

  @testset "Identical Points" begin
    # Test with identical data points
    data_identical = ones(5, 2)
    support_points_diff = [1.1 1.1; 0.9 0.9]

    indices_identical = subsample_indices(data_identical, support_points_diff)

    @test length(indices_identical) == 2
    @test all(1 .<= indices_identical .<= 5)
    @test length(unique(indices_identical)) == 2

    # Since all data points are identical, any selection is valid
    # The algorithm should still work without crashing
  end

  @testset "Precision Test" begin
    # Test with very close points to check numerical stability
    data_precise = [0.0 0.0; 1e-10 1e-10; 1.0 1.0]
    support_precise = [1e-11 1e-11]

    indices_precise = subsample_indices(data_precise, reshape(support_precise, 1, 2))

    @test length(indices_precise) == 1
    # Should select either point 1 or 2 (both very close to support point)
    @test indices_precise[1] in [1, 2]
  end
end
