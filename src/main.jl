using Random
using Polynomials


"""
    SPlit(data::AbstractArray; split_ratio::Float64 = 0.2, kappa::Union{Nothing, Float64} = nothing, max_iterations::Int = 500, tolerance::Float64 = 1e-10, n_threads::Union{Nothing, Int} = nothing)

Split a dataset for training and testing.

`SPlit()` splits a dataset into two subsets for training and testing. 
The split is based on the support points computed using the `compute_sp()` function.
The support points are computed for continuous variables only. 
The categorical variables are handled by converting them to continuous variables using Helmert coding.
The support points are then used to perform a nearest neighbor subsampling using the `subsample()` function.

# Arguments
- `data::AbstractArray`: The dataset including both the predictors and response(s); should not contain missing values, and only numeric and/or factor columns are allowed.
- `split_ratio::Float64`: The ratio in which the dataset is to be split; should be in (0, 1), e.g., for an 80-20 split, the `split_ratio` is either 0.8 or 0.2.
- `kappa::Union{Nothing, Float64}`: If provided, stochastic majorization-minimization is used for computing support points using a random sample from the dataset of size `ceil(kappa * split_ratio * size(data, 1))` in every iteration.
- `max_iterations::Int`: The maximum number of iterations before the tolerance level is reached during support points optimization.
- `tolerance::Float64`: The tolerance level for support points optimization; measured in terms of the maximum point-wise difference in distance between successive solutions.
- `n_threads::Union{Nothing, Int}`: Number of threads to be used for parallel computation; if not supplied, `n_threads` defaults to the maximum available threads.

# Returns
Train/Test dataset indices of the original dataset.

# Details
Support points are defined only for continuous variables. The categorical variables are handled as follows. `SPlit()` will automatically convert a nominal categorical variable with *m* levels to *m - 1* continuous variables using Helmert coding. Ordinal categorical variables should be converted to numerical columns using a scoring method before using `SPlit()`.
For example, if the three levels of an ordinal variable are poor, good, and excellent, then the user may choose 1, 2, and 5 to represent the three levels. These values depend on the problem and data collection method, and therefore, `SPlit()` will not do it automatically. The columns of the resulting numeric dataset are then standardized to have mean zero and variance one.
`SPlit()` then computes the support points and calls the provided `subsample()` function to perform a nearest neighbor subsampling. The indices of this subsample are returned.
`SPlit` can be time-consuming for large datasets. The computational time can be reduced by using the stochastic majorization-minimization technique with a trade-off in the quality of the split. For example, setting `kappa = 2` will use a random sample twice the size of the smaller subset in the split, instead of using the whole dataset in every iteration of the support points optimization. Another option for large datasets is to use data twinning (Vakayil and Joseph, 2022) implemented in the `R` package [twinning](https://CRAN.R-project.org/package=twinning). `Twinning` is extremely fast, but for small datasets, the results may not be as good as `SPlit`.


## Examples
```julia
# 1. An 80-20 split of a numeric dataset
using Random, Statistics, Plots

Random.seed!(123)
X = randn(100)
Y = randn(100) .+ X.^2
data = hcat(X, Y)
SPlitIndices = SPlit(data, tolerance=1e-6, nThreads=2)
dataTest = data[SPlitIndices, :]
dataTrain = data[setdiff(1:size(data, 1), SPlitIndices), :]
scatter(data[:, 1], data[:, 2], title="SPlit testing set")
scatter!(dataTest[:, 1], dataTest[:, 2], color=:green, markersize=8)

# 2. An 80-20 split of the iris dataset
using RDatasets
iris = dataset("datasets", "iris")
SPlitIndices = SPlit(Matrix(iris[:, 1:4]), nThreads=2)
irisTest = iris[SPlitIndices, :]
irisTrain = iris[setdiff(1:size(iris, 1), SPlitIndices), :]
```

## References

* Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. Technometrics, 1-11. doi:10.1080/00401706.2021.1921037.

* Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal. https://doi.org/10.1002/sam.11574.

* Mak, S., & Joseph, V. R. (2018). Support points. The Annals of Statistics, 46(6A), 2562-2592.

"""
function SPlit(
  data::AbstractArray;
  split_ratio::Float64 = 0.2,
  kappa::Union{Nothing,Float64} = nothing,
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  n_threads::Union{Nothing,Int} = nothing,
)
  # Validate split_ratio
  if split_ratio <= 0 || split_ratio >= 1
    error("split_ratio should be in (0, 1).")
  end

  # Convert data into DataFrames
  if eltype(data) <: Number
    df = DataFrame(data)
  end

  # Preprocess the data
  df_preprocessed = preprocess(df)

  # Determine the size of the subset for splitting
  # If split_ratio is nothing, use optimal split ratio
  if isnothing(split_ratio)
    split_ratio = splitratio(df_preprocessed, df_preprocessed[:, end])
  end
  n_subset = round(Int, min(split_ratio, 1 - split_ratio) * size(df_preprocessed, 1))

  # Determine kappa value
  if isnothing(kappa)
    kappa = size(df_preprocessed, 1)
  else
    if kappa <= 0
      error("kappa should be positive.")
    end
    kappa = min(size(formatted_data, 1), ceil(Int, kappa * n_subset))
  end

  # Compute support points
  support_points = compute_sp(
    n_subset,
    size(df_preprocessed, 2);
    dist_samp = df_preprocessed,
    num_subsamp = kappa,
    iter_max = max_iterations,
    tol = tolerance,
    nThreads = n_threads,
  )

  # Perform nearest neighbor subsampling and return the indices
  return subsample(formatted_data, support_points)
end

"""
Optimal splitting ratio.

`splitratio()` finds the optimal splitting ratio by assuming a polynomial regression model with interactions can approximate the true model. The number of parameters in the model is estimated from the full data using stepwise regression. A simpler solution is to choose the number of parameters to be the square root of the number of unique rows in the input matrix of the dataset. Please see Joseph (2022) for details.

# Arguments
- `xs`: Input matrix.
- `ys`: Response (output variable).
- `method`: This could be `"simple"` or `"regression"`. The default method `"simple"` uses the square root of the number of unique rows in `x` as the number of parameters, whereas `"regression"` estimates the number of parameters using stepwise regression. The `"regression"` method works only with continuous output variables.
- `degree`: Specifies the degree of the polynomial to be fitted, which is needed only if `method="regression"` is used. Default is 2.

# Returns
Splitting ratio, which is the fraction of the dataset to be used for testing.

# Examples
```julia
using Random

Random.seed!(123)
X = randn(100)
Y = randn(100) .+ X.^2
splitratio(X, Y)
splitratio(X, Y, method="regression")

# References
Joseph, V. R. (2022). Optimal Ratio for Data Splitting. Statistical Analysis & Data Mining: The ASA Data Science Journal, to appear
"""
function splitratio(xs, ys; method = "simple", degree = 2)
  if method == "regression"
    xs = convert(Matrix{Float64}, x)
    poly = fit(xs, ys, degree)
    ys_pred = poly.(x)

    # Get coefficients of the polynomial
    coeffs = coefficients(poly)

    # Count the number of significant coefficients (nonzero coefficients)
    threshold = 1e-8
    significant_coeffs = sum(abs.(coeffs) .> threshold)
    p = length(significant_coeffs)
  elseif method == "simple" || eltype(ys) <: Number
    p = sqrt(size(unique(xs, dims = 1))[1])
  end

  γ = 1 / (sqrt(p) + 1)
  return γ
end
