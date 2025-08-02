"""
Data preprocessing functions for SPlit.jl
"""

using DataFrames
using Statistics
using StatsAPI
using LinearAlgebra

"""
    helmert_contrasts(levels::Vector{T}) where T

Generate Helmert contrast matrix for categorical variables.

# Arguments
- `levels`: Vector of unique levels for the categorical variable

# Returns
- Matrix of Helmert contrasts with size (n_levels, n_levels-1)
"""
function helmert_contrasts(levels::Vector{T}) where {T}
  n_levels = length(levels)
  if n_levels <= 1
    return zeros(n_levels, 0)
  end

  contrasts = zeros(n_levels, n_levels - 1)

  for j = 1:(n_levels-1)
    # Positive values for first j levels
    for i = 1:j
      contrasts[i, j] = 1.0 / j
    end
    # Negative value for (j+1)th level
    contrasts[j+1, j] = -1.0
  end

  return contrasts
end

"""
    encode_categorical!(result_matrix::Matrix{Float64}, data_col::AbstractVector,
                       col_idx::Int, levels::Vector)

Encode a categorical column using Helmert contrasts and add to result matrix.

# Arguments
- `result_matrix`: Matrix to store encoded values
- `data_col`: Categorical data column
- `col_idx`: Starting column index in result matrix
- `levels`: Unique levels of the categorical variable

# Returns
- Number of columns added to the result matrix
"""
function encode_categorical!(
  result_matrix::Matrix{Float64},
  data_col::AbstractVector,
  col_idx::Int,
  levels::Vector,
)
  n_levels = length(levels)
  if n_levels <= 1
    return 0
  end

  contrasts = helmert_contrasts(levels)
  n_contrasts = size(contrasts, 2)

  # Create mapping from levels to indices
  level_to_idx = Dict(level => i for (i, level) in enumerate(levels))

  # Apply contrasts
  for (row_idx, value) in enumerate(data_col)
    level_idx = level_to_idx[value]
    for contrast_idx = 1:n_contrasts
      result_matrix[row_idx, col_idx+contrast_idx-1] = contrasts[level_idx, contrast_idx]
    end
  end

  return n_contrasts
end

"""
    count_encoded_columns(data)

Count total number of columns after categorical encoding.

# Arguments
- `data`: Input data (Matrix or DataFrame)

# Returns
- Number of columns after encoding
"""
function count_encoded_columns(data)
  if isa(data, Matrix)
    return size(data, 2)
  end

  total_cols = 0
  for col_name in names(data)
    col = data[!, col_name]
    if eltype(col) <: AbstractString || isa(col, CategoricalVector)
      # Categorical column
      unique_vals = unique(col)
      if length(unique_vals) > 1
        total_cols += length(unique_vals) - 1  # Helmert contrasts
      end
    elseif eltype(col) <: Number
      # Numeric column - check if constant
      if length(unique(col)) > 1
        total_cols += 1
      end
    else
      error("Dataset contains non-numeric non-categorical column: $(col_name)")
    end
  end

  return total_cols
end

"""
    format_data(data)

Format and preprocess data for SPlit algorithm.

This function:
1. Handles missing values (throws error if found)
2. Converts categorical variables to Helmert contrasts
3. Removes constant columns
4. Standardizes all columns to have mean 0 and variance 1

# Arguments
- `data`: Input dataset (Matrix or DataFrame)

# Returns
- Preprocessed matrix with standardized columns

# Throws
- `ArgumentError`: If dataset contains missing values
- `ArgumentError`: If dataset contains non-numeric non-categorical columns
"""
function format_data(data)
  # Check for missing values
  if any(ismissing, data)
    throw(ArgumentError("Dataset contains missing value(s)."))
  end

  n_rows = size(data, 1)

  if isa(data, Matrix)
    # Handle Matrix input - assume all numeric
    if !all(x -> isa(x, Number), data)
      throw(ArgumentError("Matrix input must contain only numeric values."))
    end

    # Remove constant columns
    result_cols = Vector{Vector{Float64}}()
    for j = 1:size(data, 2)
      col = data[:, j]
      if length(unique(col)) > 1  # Not constant
        push!(result_cols, Float64.(col))
      end
    end

    if isempty(result_cols)
      throw(ArgumentError("All columns are constant."))
    end

    result = hcat(result_cols...)
  else
    # Handle DataFrame input
    n_encoded_cols = count_encoded_columns(data)

    if n_encoded_cols == 0
      throw(ArgumentError("All columns are constant or invalid."))
    end

    result = Matrix{Float64}(undef, n_rows, n_encoded_cols)
    current_col = 1

    for col_name in names(data)
      col = data[!, col_name]

      if eltype(col) <: AbstractString || isa(col, CategoricalVector)
        # Handle categorical column
        unique_vals = unique(col)
        if length(unique_vals) > 1
          cols_added = encode_categorical!(result, col, current_col, unique_vals)
          current_col += cols_added
        end
      elseif eltype(col) <: Number
        # Handle numeric column
        if length(unique(col)) > 1  # Not constant
          result[:, current_col] = Float64.(col)
          current_col += 1
        end
      else
        throw(
          ArgumentError("Dataset contains non-numeric non-categorical column: $(col_name)"),
        )
      end
    end
  end

  # Standardize columns (mean 0, variance 1)
  result_standardized = similar(result)
  for j = 1:size(result, 2)
    col = result[:, j]
    μ = mean(col)
    σ = std(col)

    if σ > 0
      result_standardized[:, j] = (col .- μ) ./ σ
    else
      # Column has zero variance after processing - shouldn't happen
      # but handle gracefully
      result_standardized[:, j] = zeros(length(col))
    end
  end

  return result_standardized
end
