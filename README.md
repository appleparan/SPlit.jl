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

A **Julia-native implementation** of optimal data splitting using support points, based on [Joseph and Vakayil (2021)](https://doi.org/10.1080/00401706.2021.1921037).

**✨ Now featuring a modern Julia API with type safety, multiple dispatch, and energy distance support!**

## Overview

SPlit provides an optimal method for splitting datasets into training and testing sets based on the method of support points. Unlike traditional random splitting, SPlit ensures that both subsets are representative of the original data distribution, leading to more reliable model evaluation and better generalization performance.

### Key Features

#### 🚀 **Modern Julia API**

- **Type-safe splitting**: Parameterized types with compile-time dispatch
- **Multiple dispatch**: Specialized methods for Matrix, DataFrame, Vector inputs
- **Composable design**: Iterator and indexing protocols for seamless integration

#### 🎯 **Advanced Distance Metrics**

- **Energy distance**: Superior splits for complex data distributions
- **Flexible metrics**: Support for any `Distances.jl` metric (Euclidean, Manhattan, etc.)
- **Quality assessment**: Built-in split quality evaluation and comparison

#### 💪 **Robust Implementation**

- **Model-independent**: Works with both regression and classification problems
- **Categorical support**: Automatic Helmert contrast encoding
- **Performance optimized**: Multi-threaded with stochastic optimization for large data
- **Backward compatible**: Full R-style API preserved for migration

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

### 🌟 **Modern Julia API** (Recommended)

```julia
using SPlit, Distances
using Random

# Generate sample data
Random.seed!(123)
data = randn(100, 3)

# Create a splitter with energy distance for better complex data handling
splitter = SupportPointSplitter(EnergyDistance(Euclidean()); ratio=0.2)

# Split the data
result = datasplit(splitter, data)

# Access train/test data using intuitive indexing
train_data = data[result, :train]
test_data = data[result, :test]

# Or use iterator interface
train_indices, test_indices = result

println("Training: $(size(train_data))")
println("Test: $(size(test_data))")
```

### 📚 **Legacy API** (For R users)

```julia
using SPlit

# Traditional function-style API (still fully supported)
test_indices = split_data(data; split_ratio=0.2)
train_data = data[setdiff(1:size(data,1), test_indices), :]
test_data = data[test_indices, :]
```

## API Reference

### 🎯 **Modern Julia API**

#### Core Types

```julia
# Create a splitter with your preferred distance metric
splitter = SupportPointSplitter(
    metric;                    # Distance metric (Euclidean(), Cityblock(), EnergyDistance(), etc.)
    ratio = 0.2,              # Split ratio for test set
    max_iterations = 500,     # Maximum optimization iterations
    tolerance = 1e-10,        # Convergence tolerance
    n_threads = nthreads(),   # Parallel threads
    kappa = nothing,          # Stochastic subsample size
    rng = GLOBAL_RNG         # Random number generator
)

# Split any data type
result = datasplit(splitter, data)  # Matrix, DataFrame, or Vector
```

#### Data Access

```julia
# Intuitive indexing interface
train_data = data[result, :train]
test_data = data[result, :test]

# Iterator protocol
train_indices, test_indices = result

# Property access
train_indices(result)
test_indices(result)
quality(result)          # If computed with datasplit_with_quality
```

#### Method Comparison

```julia
# Compare multiple distance metrics
methods = [
    SupportPointSplitter(Euclidean()),
    SupportPointSplitter(Cityblock()),
    SupportPointSplitter(EnergyDistance(Euclidean()))
]

comparison = compare(methods, data)
summary(comparison)      # DataFrame with quality metrics
best_method, best_result = best(comparison)

# Quick comparison
quick_comparison = quick_compare(data; ratio=0.2)
```

### 📚 **Legacy API** (Backward Compatible)

#### `split_data(data; split_ratio=0.2, metric=Euclidean(), ...)`

Traditional function-style interface with all original R-package functionality.

#### `optimal_split_ratio(x, y; method="simple")`

Determine optimal splitting ratio using √n rule or regression methods.

## Examples

### Example 1: Basic Usage with Energy Distance

```julia
using SPlit, Distances, Random
Random.seed!(42)

# Generate complex data with nonlinear relationships
n = 200
X = randn(n, 3)
Y = X[:, 1].^2 + sin.(X[:, 2]) + X[:, 3] + 0.1 * randn(n)
data = hcat(X, Y)

# Use Energy Distance for better handling of complex distributions
splitter = SupportPointSplitter(EnergyDistance(Euclidean()); ratio=0.25)
result = datasplit(splitter, data)

# Clean data access
train_data = data[result, :train]
test_data = data[result, :test]

println("Train: $(size(train_data)), Test: $(size(test_data))")
```

### Example 2: Method Comparison & Quality Assessment

```julia
using SPlit, DataFrames, CategoricalArrays

# Mixed-type dataset
df = DataFrame(
    x1 = randn(150),
    x2 = randn(150),
    category = categorical(rand(["A", "B", "C"], 150)),
    target = randn(150)
)

# Compare different distance metrics
methods = [
    SupportPointSplitter(Euclidean(); ratio=0.2),
    SupportPointSplitter(Cityblock(); ratio=0.2),
    SupportPointSplitter(EnergyDistance(Euclidean()); ratio=0.2)
]

comparison = compare(methods, df; quality=true)
println(summary(comparison))

# Select best method by quality
best_method, best_result = best(comparison; by=:Quality)
train_df = df[best_result, :train]
test_df = df[best_result, :test]
```

### Example 3: High-Performance for Large Data

```julia
using SPlit

# Large dataset optimization
large_data = randn(10_000, 20)

# Use stochastic optimization for speed
splitter = SupportPointSplitter(
    Euclidean();
    ratio = 0.15,
    kappa = 1000,           # Stochastic subsample size
    max_iterations = 100,   # Fewer iterations for speed
    n_threads = 4          # Parallel processing
)

result = datasplit(splitter, large_data)
println("Large data split: $(length(train_indices(result))) train, $(length(test_indices(result))) test")
```

### Example 4: Finding Optimal Split Ratio

```julia
using SPlit

# Generate data
X = randn(100, 3)
Y = X[:, 1] + X[:, 2].^2 + 0.1 * randn(100)

# Find optimal ratio using traditional method
optimal_ratio = optimal_split_ratio(X, Y)
println("Optimal test ratio: $optimal_ratio")

# Apply with modern API
splitter = SupportPointSplitter(ratio=optimal_ratio)
result = datasplit(splitter, hcat(X, Y))
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
