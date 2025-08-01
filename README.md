# SPlit.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://appleparan.github.io/SPlit.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://appleparan.github.io/SPlit.jl/dev)
[![Build Status](https://github.com/appleparan/SPlit.jl/workflows/Test/badge.svg)](https://github.com/appleparan/SPlit.jl/actions)
[![Test workflow status](https://github.com/appleparan/SPlit.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/appleparan/SPlit.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/appleparan/SPlit.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Docs.yml?query=branch%3Amain)

[![Coverage](https://codecov.io/gh/appleparan/SPlit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/appleparan/SPlit.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/appleparan/SPlit.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)

A Julia implementation of the optimal data splitting procedure described in [Joseph and Vakayil (2021)](https://doi.org/10.1080/00401706.2021.1921037). This is a port of the original [SPlit R package](https://CRAN.R-project.org/package=SPlit).

## Overview

SPlit provides an optimal method for splitting datasets into training and testing sets based on the method of support points. Unlike traditional random splitting, SPlit ensures that both subsets are representative of the original data distribution, leading to more reliable model evaluation and better generalization performance.

### Key Features

- **Model-independent**: Works with both regression and classification problems
- **Optimal splitting**: Based on support points theory for optimal data representation
- **Categorical support**: Handles both numeric and categorical variables
- **Flexible ratios**: Supports any split ratio between 0 and 1
- **Performance optimized**: Multi-threaded implementation for large datasets
- **Automatic ratio selection**: Includes functionality to determine optimal split ratios

## Installation

```julia
using Pkg
Pkg.add("SPlit")
```

Or from the Julia REPL:

```julia
] add SPlit
```

## Quick Start

```julia
using SPlit
using Random

# Generate sample data
Random.seed!(123)
n = 100
X = randn(n)
Y = randn(n) .+ X.^2
data = hcat(X, Y)

# Split data with default 80-20 ratio
test_indices = split_data(data, split_ratio=0.2)
train_data = data[setdiff(1:n, test_indices), :]
test_data = data[test_indices, :]

println("Training set size: ", size(train_data, 1))
println("Test set size: ", size(test_data, 1))
```

## Main Functions

### `split_data(data; split_ratio=0.2, kappa=nothing, max_iterations=500, tolerance=1e-10, n_threads=Threads.nthreads())`

Split a dataset optimally for training and testing.

**Arguments:**
- `data`: Dataset matrix or DataFrame (numeric and categorical columns supported)
- `split_ratio`: Fraction for the smaller subset (default: 0.2 for 80-20 split)
- `kappa`: Subsample size for stochastic optimization (optional)
- `max_iterations`: Maximum iterations for support points optimization
- `tolerance`: Convergence tolerance
- `n_threads`: Number of threads for parallel computation

**Returns:** Vector of indices for the smaller subset.

### `optimal_split_ratio(x, y; method="simple", degree=2)`

Determine the optimal splitting ratio for your dataset.

**Arguments:**
- `x`: Input matrix
- `y`: Response variable
- `method`: "simple" (uses √n rule) or "regression" (stepwise regression)
- `degree`: Polynomial degree for regression method

**Returns:** Optimal split ratio for testing set.

## Examples

### Example 1: Numeric Data

```julia
using SPlit, Random
Random.seed!(42)

# Create sample data
n = 200
X1 = randn(n)
X2 = randn(n)
Y = X1.^2 + X2 + 0.5 * randn(n)

data = hcat(X1, X2, Y)

# Optimal split
test_indices = split_data(data, split_ratio=0.25)
train_data = data[setdiff(1:n, test_indices), :]
test_data = data[test_indices, :]
```

### Example 2: Mixed Data Types (with DataFrames)

```julia
using SPlit, DataFrames, CategoricalArrays

# Create mixed-type dataset
df = DataFrame(
    x1 = randn(100),
    x2 = randn(100),
    category = categorical(rand(["A", "B", "C"], 100)),
    y = randn(100)
)

# Split the data
test_indices = split_data(df, split_ratio=0.3)
train_df = df[setdiff(1:nrow(df), test_indices), :]
test_df = df[test_indices, :]
```

### Example 3: Finding Optimal Split Ratio

```julia
using SPlit

# Generate data
X = randn(100, 3)
Y = X[:, 1] + X[:, 2]^2 + 0.1 * randn(100)

# Find optimal ratio
optimal_ratio = optimal_split_ratio(X, Y)
println("Optimal test ratio: ", optimal_ratio)

# Use the optimal ratio
test_indices = split_data(hcat(X, Y), split_ratio=optimal_ratio)
```

## Algorithm Details

SPlit uses the method of support points to create optimal data splits:

1. **Data preprocessing**: Categorical variables are encoded using Helmert contrasts, and all variables are standardized
2. **Support points computation**: Iteratively optimizes support points to minimize energy distance
3. **Nearest neighbor assignment**: Each data point is assigned to its nearest support point
4. **Subset selection**: Returns indices of the smaller subset based on support point assignments

For large datasets, stochastic optimization can be enabled using the `kappa` parameter to improve computational efficiency.

## Performance Considerations

- **Large datasets**: Use `kappa` parameter for stochastic optimization
- **Parallel processing**: Automatically uses available CPU threads
- **Memory usage**: Efficient implementation with minimal memory overhead
- **Alternative for very large data**: Consider the [Twinning.jl](https://github.com/example/Twinning.jl) package for extremely large datasets

## References

1. Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502. [DOI](https://doi.org/10.1080/00401706.2021.1921037)

2. Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.

3. Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining: The ASA Data Science Journal*, 15(4), 537-546.

## How to Cite

If you use SPlit.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/appleparan/SPlit.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://appleparan.github.io/SPlit.jl/dev/90-contributing/).

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
