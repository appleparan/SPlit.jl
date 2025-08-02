using DataFrames, StatsBase, StatsModels

"""
    convert_categorical_to_numerical(df::DataFrame, column::Symbol, coding::AbstractContrasts = HelmertCoding())

Convert a categorical column in a DataFrame to numerical columns using the specified coding method.

Parameters:
- `df`: The input DataFrame
- `column`: The symbol representing the categorical column to be converted
- `coding`: The coding method to be used (default: HelmertCoding())

Returns:
- A original DataFrame with the categorical column replaced by numerical columns

Example:
    df = DataFrame(category = ["A", "B", "C", "A", "B"])
    convert_categorical_to_numerical(df, :category, DummyCoding())
"""
function convert_categorical_to_numerical(
  df::DataFrame,
  column::Symbol,
  coding::AbstractContrasts = HelmertCoding(),
)
  f = @eval @formula(0 ~ $(column))

  schema = StatsModels.schema(f, df, Dict(column => coding))
  mm = modelmatrix(schema, df)

  new_columns = [Symbol("$(column)_$i") for i in axes(mm, 2)]

  for (i, col) in enumerate(new_columns)
    df[!, col] = mm[:, i]
  end

  select!(df, Not(column))

  return df
end



"""
    preprocess(df::DataFrame)

Preprocess a DataFrame by converting categorical columns to numerical and standardizing all columns.

This function performs the following steps:
1. Iterates through each column of the input DataFrame.
2. Checks for missing values and raises an error if any are found.
3. Converts categorical columns to numerical using `convert_categorical_to_numerical`.
4. Standardizes all columns using `StatsBase.zscore`.

Parameters:
- `df`: The input DataFrame

Returns:
- A new DataFrame with preprocessed data

Throws:
- `ErrorException` if any missing values are found in the DataFrame

Example:
    df = DataFrame(
        A = [1, 2, 3, 4, 5],
        B = ["X", "Y", "Z", "X", "Y"],
        C = [1.1, 2.2, 3.3, 4.4, 5.5]
    )
    preprocessed_df = preprocess(df)
"""
function preprocess(df::DataFrame; standardize_function::Function = zscore)
  processed_df = copy(df)

  for col in names(processed_df)
    # Check for missing values
    if any(ismissing, processed_df[!, col])
      throw(ErrorException("Missing values found in column: $col"))
    end

    # Convert categorical columns to numerical
    if eltype(processed_df[!, col]) <: Union{CategoricalValue,AbstractString}
      processed_df = convert_categorical_to_numerical(processed_df, Symbol(col))
    end
  end

  # Standardize all columns using StatsBase.zscore
  for col in names(processed_df)
    processed_df[!, col] = standardize_function(processed_df[!, col])
  end

  return processed_df
end
