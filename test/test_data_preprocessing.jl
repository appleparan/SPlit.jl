using Test
using DataFrames
using CategoricalArrays
using Statistics
using Dates

include("../src/data_preprocessing.jl")

@testset "Data Preprocessing Tests" begin

  @testset "Helmert Contrasts" begin
    # Test with 3 levels
    levels = ["A", "B", "C"]
    contrasts = helmert_contrasts(levels)

    @test size(contrasts) == (3, 2)

    # Test first column
    @test contrasts[1, 1] ≈ 1.0
    @test contrasts[2, 1] ≈ -1.0
    @test contrasts[3, 1] ≈ 0.0

    # Test second column
    @test contrasts[1, 2] ≈ 0.5
    @test contrasts[2, 2] ≈ 0.5
    @test contrasts[3, 2] ≈ -1.0

    # Test with 1 level (should return empty matrix)
    single_level = ["A"]
    contrasts_single = helmert_contrasts(single_level)
    @test size(contrasts_single) == (1, 0)

    # Test with 2 levels
    two_levels = ["X", "Y"]
    contrasts_two = helmert_contrasts(two_levels)
    @test size(contrasts_two) == (2, 1)
    @test contrasts_two[1, 1] ≈ 1.0
    @test contrasts_two[2, 1] ≈ -1.0
  end

  @testset "Count Encoded Columns" begin
    # Test with Matrix
    matrix_data = randn(10, 3)
    @test count_encoded_columns(matrix_data) == 3

    # Test with DataFrame - numeric only
    df_numeric = DataFrame(x1 = randn(10), x2 = randn(10), x3 = randn(10))
    @test count_encoded_columns(df_numeric) == 3

    # Test with DataFrame - mixed types
    df_mixed = DataFrame(
      x1 = randn(10),
      cat1 = categorical(rand(["A", "B", "C"], 10)),
      x2 = randn(10),
      cat2 = categorical(rand(["X", "Y"], 10)),
    )
    # x1 (1) + cat1 (2) + x2 (1) + cat2 (1) = 5
    @test count_encoded_columns(df_mixed) == 5

    # Test with constant column
    df_with_constant = DataFrame(
      x1 = randn(10),
      constant_col = fill(1.0, 10),
      cat1 = categorical(rand(["A", "B"], 10)),
    )
    # x1 (1) + constant_col (0) + cat1 (1) = 2
    @test count_encoded_columns(df_with_constant) == 2
  end

  @testset "Format Data - Matrix Input" begin
    # Test with simple numeric matrix
    data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    formatted = format_data(data)

    @test size(formatted) == (3, 2)
    # Check standardization (mean ≈ 0, std ≈ 1)
    @test abs(mean(formatted[:, 1])) < 1e-10
    @test abs(mean(formatted[:, 2])) < 1e-10
    @test abs(std(formatted[:, 1]) - 1.0) < 1e-10
    @test abs(std(formatted[:, 2]) - 1.0) < 1e-10

    # Test with constant column (should be removed)
    data_with_constant = [1.0 1.0; 2.0 1.0; 3.0 1.0]
    formatted_no_constant = format_data(data_with_constant)
    @test size(formatted_no_constant) == (3, 1)

    # Test error with missing values
    data_with_missing = [1.0 missing; 2.0 3.0]
    @test_throws ArgumentError format_data(data_with_missing)
  end

  @testset "Format Data - DataFrame Input" begin
    # Test with numeric DataFrame
    df = DataFrame(x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0])
    formatted = format_data(df)

    @test size(formatted) == (3, 2)
    @test abs(mean(formatted[:, 1])) < 1e-10
    @test abs(std(formatted[:, 1]) - 1.0) < 1e-10

    # Test with categorical variables
    df_cat = DataFrame(
      x1 = [1.0, 2.0, 3.0, 4.0],
      cat1 = categorical(["A", "B", "C", "A"]),
      x2 = [5.0, 6.0, 7.0, 8.0],
    )
    formatted_cat = format_data(df_cat)

    # Should have x1 (1) + cat1 (2 Helmert contrasts) + x2 (1) = 4 columns
    @test size(formatted_cat) == (4, 4)

    # Test with single-level categorical (should be ignored)
    df_single_cat = DataFrame(
      x1 = [1.0, 2.0, 3.0],
      single_cat = categorical(["A", "A", "A"]),
      x2 = [4.0, 5.0, 6.0],
    )
    formatted_single = format_data(df_single_cat)
    @test size(formatted_single) == (3, 2)  # Only x1 and x2

    # Test error with invalid column type
    df_invalid = DataFrame(
      x1 = [1.0, 2.0, 3.0],
      invalid_col = [Date(2021, 1, 1), Date(2021, 1, 2), Date(2021, 1, 3)],
    )
    @test_throws Exception format_data(df_invalid)

    # Test with all constant columns
    df_all_constant = DataFrame(const1 = [1.0, 1.0, 1.0], const2 = [2.0, 2.0, 2.0])
    @test_throws ArgumentError format_data(df_all_constant)
  end

  @testset "Categorical Encoding" begin
    # Test encoding with known result
    data_col = ["A", "B", "C", "A", "B"]
    levels = ["A", "B", "C"]
    result_matrix = zeros(5, 2)

    n_cols = encode_categorical!(result_matrix, data_col, 1, levels)
    @test n_cols == 2

    # Check first few entries
    @test result_matrix[1, 1] ≈ 1.0  # A -> (1, 0.5)
    @test result_matrix[1, 2] ≈ 0.5
    @test result_matrix[2, 1] ≈ -1.0  # B -> (-1, 0.5)
    @test result_matrix[2, 2] ≈ 0.5
    @test result_matrix[3, 1] ≈ 0.0  # C -> (0, -1)
    @test result_matrix[3, 2] ≈ -1.0
  end

  @testset "Edge Cases" begin
    # Test with very small dataset
    small_data = reshape([1.0, 2.0], 2, 1)
    formatted_small = format_data(small_data)
    @test size(formatted_small) == (2, 1)

    # Test with single row (should work but might have numerical issues)
    single_row = reshape([1.0, 2.0, 3.0], 1, 3)
    @test_throws Exception format_data(single_row)  # Standard deviation would be 0
  end
end
