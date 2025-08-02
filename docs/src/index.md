```@meta
CurrentModule = SPlit
```

# SPlit.jl

**Julia-native implementation** of optimal data splitting using support points, based on [Joseph and Vakayil (2021)](https://arxiv.org/abs/2012.10945).

✨ **Now featuring a modern Julia API with type safety, multiple dispatch, and energy distance support!**

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

## Quick Start

### Modern Julia API (Recommended)

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

### Legacy API (For R users)

```julia
using SPlit

# Traditional function-style API (still fully supported)
test_indices = split_data(data; split_ratio=0.2)
train_data = data[setdiff(1:size(data,1), test_indices), :]
test_data = data[test_indices, :]
```

## API Reference

See the [Reference](@ref reference) section for complete API documentation.

## Examples

### Method Comparison & Quality Assessment

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

## Algorithm Details

SPlit uses the method of support points to create optimal data splits:

1. **Data preprocessing**: Categorical variables are encoded using Helmert contrasts, and all variables are standardized
2. **Support points computation**: Iteratively optimizes support points to minimize energy distance
3. **Nearest neighbor assignment**: Each data point is assigned to its nearest support point
4. **Subset selection**: Returns indices of the smaller subset based on support point assignments

For large datasets, stochastic optimization can be enabled using the `kappa` parameter to improve computational efficiency.

## References

1. Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502. [DOI](https://arxiv.org/abs/2012.10945)

2. Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.

3. Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining: The ASA Data Science Journal*, 15(4), 537-546.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
