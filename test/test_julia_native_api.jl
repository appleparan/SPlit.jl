using Test
using Random
using DataFrames
using Distances
using SPlit

@testset "Julia-native API Tests" begin

  @testset "Type System" begin
    # Test SupportPointSplitter construction
    @test SupportPointSplitter() isa SupportPointSplitter{Euclidean}
    @test SupportPointSplitter(Cityblock()) isa SupportPointSplitter{Cityblock}
    @test SupportPointSplitter(EnergyDistance(Euclidean())) isa
          SupportPointSplitter{EnergyDistance{Euclidean}}

    # Test parameter validation
    @test_throws ArgumentError SupportPointSplitter(ratio = 0.0)
    @test_throws ArgumentError SupportPointSplitter(ratio = 1.0)
    @test_throws ArgumentError SupportPointSplitter(max_iterations = 0)
    @test_throws ArgumentError SupportPointSplitter(tolerance = 0.0)
    @test_throws ArgumentError SupportPointSplitter(n_threads = 0)
    @test_throws ArgumentError SupportPointSplitter(kappa = 0)

    # Test property access
    splitter = SupportPointSplitter(Cityblock(); ratio = 0.3)
    @test ratio(splitter) == 0.3
    @test metric(splitter) isa Cityblock
  end

  @testset "Basic Splitting" begin
    Random.seed!(123)
    data = randn(30, 2)

    # Test different splitters
    euclidean_splitter = SupportPointSplitter(Euclidean(); ratio = 0.2, max_iterations = 5)
    result = datasplit(euclidean_splitter, data)

    @test result isa SplitResult
    @test length(train_indices(result)) == 24  # 80% of 30
    @test length(test_indices(result)) == 6    # 20% of 30
    @test length(train_indices(result)) + length(test_indices(result)) == 30

    # Test iterator interface
    train_idx, test_idx = result
    @test train_idx == train_indices(result)
    @test test_idx == test_indices(result)

    # Test no overlap
    @test isempty(intersect(train_indices(result), test_indices(result)))
    @test Set(union(train_indices(result), test_indices(result))) == Set(1:30)
  end

  @testset "Multiple Dispatch" begin
    Random.seed!(456)

    # Test Matrix input
    matrix_data = randn(20, 3)
    splitter = SupportPointSplitter(ratio = 0.25, max_iterations = 3)
    result_matrix = datasplit(splitter, matrix_data)
    @test length(test_indices(result_matrix)) == 5  # 25% of 20

    # Test Vector input
    vector_data = randn(20)
    result_vector = datasplit(splitter, vector_data)
    @test length(test_indices(result_vector)) == 5

    # Test DataFrame input
    df_data = DataFrame(x1 = randn(20), x2 = randn(20), x3 = randn(20))
    result_df = datasplit(splitter, df_data)
    @test length(test_indices(result_df)) == 5
  end

  @testset "Data Indexing Interface" begin
    Random.seed!(789)
    data = randn(25, 2)
    splitter = SupportPointSplitter(ratio = 0.2, max_iterations = 3)
    result = datasplit(splitter, data)

    # Test array indexing
    train_data = data[result, :train]
    test_data = data[result, :test]

    @test size(train_data, 1) == length(train_indices(result))
    @test size(test_data, 1) == length(test_indices(result))
    @test size(train_data, 2) == size(test_data, 2) == 2

    # Test DataFrame indexing
    df = DataFrame(data, :auto)
    train_df = df[result, :train]
    test_df = df[result, :test]

    @test nrow(train_df) == length(train_indices(result))
    @test nrow(test_df) == length(test_indices(result))
  end

  @testset "Quality Assessment" begin
    Random.seed!(111)
    data = randn(40, 3)
    splitter = SupportPointSplitter(ratio = 0.2, max_iterations = 5)

    # Test split with quality
    result = datasplit_with_quality(splitter, data)
    @test quality(result) isa Float64
    @test quality(result) >= 0  # Energy distance is non-negative
  end

  @testset "Comparison Interface" begin
    Random.seed!(222)
    data = randn(30, 2)

    # Test method comparison
    methods = [
      SupportPointSplitter(Euclidean(); ratio = 0.2, max_iterations = 3),
      SupportPointSplitter(Cityblock(); ratio = 0.2, max_iterations = 3),
    ]

    comparison = compare(methods, data; quality = true)
    @test comparison isa SplitComparison
    @test length(comparison.methods) == 2
    @test length(comparison.results) == 2

    # Test summary
    df = summary(comparison)
    @test df isa DataFrame
    @test nrow(df) == 2
    @test "Quality" in names(df)

    # Test quick compare
    quick_comp = quick_compare(data; ratio = 0.25, max_iterations = 3)
    @test quick_comp isa SplitComparison
    @test length(quick_comp.methods) == 3  # Default methods
  end

  @testset "Edge Cases" begin
    Random.seed!(333)

    # Test with very small dataset
    small_data = randn(10, 2)
    splitter = SupportPointSplitter(ratio = 0.2, max_iterations = 2)
    result = datasplit(splitter, small_data)
    @test length(test_indices(result)) == 2
    @test length(train_indices(result)) == 8

    # Test with single column
    single_col = randn(20, 1)
    result_single = datasplit(splitter, single_col)
    @test length(test_indices(result_single)) == 4
  end

  @testset "Backward Compatibility" begin
    Random.seed!(444)
    data = randn(30, 2)

    # Test that old API still works
    old_indices = split_data(data; split_ratio = 0.2, max_iterations = 3)
    @test length(old_indices) == 6  # 20% of 30

    # Test that new API gives similar results (structure-wise)
    splitter = SupportPointSplitter(ratio = 0.2, max_iterations = 3)
    new_result = datasplit(splitter, data)
    @test length(test_indices(new_result)) == length(old_indices)
  end

  @testset "Different Distance Metrics" begin
    Random.seed!(555)
    data = randn(25, 2)

    # Test different distance metrics
    metrics = [Euclidean(), Cityblock(), EnergyDistance(Euclidean())]

    for metric_type in metrics
      splitter = SupportPointSplitter(metric_type; ratio = 0.2, max_iterations = 3)
      result = datasplit(splitter, data)

      @test length(test_indices(result)) == 5
      @test length(train_indices(result)) == 20
      @test metric(splitter) isa typeof(metric_type)
    end
  end
end
