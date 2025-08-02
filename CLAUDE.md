# SPlit.jl Development Context

This document provides context for Claude Code when working on the SPlit.jl project.

## Project Overview

SPlit.jl is a Julia port of the original SPlit R package, implementing optimal data splitting procedures based on the method of support points as described in Joseph and Vakayil (2021).

### Original R Package Details

- **Package**: SPlit
- **Version**: 1.2 (2022-03-22)
- **Authors**: Akhil Vakayil, Roshan Joseph, Simon Mak
- **License**: GPL (>= 2)
- **CRAN**: <https://CRAN.R-project.org/package=SPlit>

## Core Functionality

### Main Functions (R → Julia mapping)

1. **`SPlit()` → `split_data()`**: Main splitting function
2. **`splitratio()` → `optimal_split_ratio()`**: Optimal ratio determination
3. **`subsample()` → `subsample()`**: Nearest neighbor subsampling (C++ → Julia)
4. **`sp_cpp()` → `compute_support_points()`**: Support points computation (C++ → Julia)

### Key Parameters

- `splitRatio`/`split_ratio`: Split proportion (default 0.2)
- `kappa`: Stochastic optimization subsample size
- `maxIterations`/`max_iterations`: Max iterations (default 500)
- `tolerance`: Convergence tolerance (default 1e-10)
- `nThreads`/`n_threads`: Parallel threads

## Implementation Status

### Current Structure

```text
src/
├── SPlit.jl              # Main module file
├── energy_distance.jl    # Energy distance calculations
├── support_points.jl     # Support points computation (TODO)
├── main.jl              # Main splitting functions (TODO)
├── convert.jl           # Data preprocessing (TODO)
└── init.jl              # Initialization functions (TODO)
```

### Dependencies

- DataFrames.jl: DataFrame support
- Distances.jl: Distance calculations
- LinearAlgebra.jl: Matrix operations
- Statistics.jl: Statistical functions
- StatsBase.jl: Statistical utilities
- StatsModels.jl: Model handling
- Polynomials.jl: Polynomial fitting for optimal ratios
- Random.jl: Random number generation

## Algorithm Implementation Notes

### Data Preprocessing (`data_format` in R)

1. Handle missing values (error if found)
2. Convert factors to Helmert contrasts
3. Remove constant columns
4. Standardize all columns (mean 0, variance 1)

### Support Points Computation (`compute_sp` + `sp_cpp` in R)

1. Initialize with jittered random sample
2. Apply bounds constraints
3. Use stochastic majorization-minimization if `kappa < n`
4. Parallel optimization with configurable threads
5. Convergence based on point-wise distance tolerance

### Nearest Neighbor Subsampling (`subsample` in R)

- C++ implementation using nanoflann library for k-d tree
- Find nearest support point for each data point
- Return indices of smaller subset

## Testing Strategy

### Test Cases to Implement

1. **Basic functionality**: Simple numeric data splitting
2. **Categorical data**: Factor handling with Helmert contrasts
3. **Mixed data types**: Numeric + categorical combinations
4. **Edge cases**: Single column, constant columns, duplicates
5. **Large datasets**: Stochastic optimization with `kappa`
6. **Optimal ratios**: Both "simple" and "regression" methods
7. **Parallel execution**: Multi-threading verification

### Reference Results

- Compare against R package outputs for identical datasets
- Verify split quality using energy distance metrics
- Check convergence behavior and iteration counts

## Development Commands

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"

# Build documentation
julia --project=docs/ docs/make.jl

# Format code
julia --project=. -e "using JuliaFormatter; format(\".\")"

# Benchmark against R
# (implement comparative benchmarking scripts)
```

## Key References

1. **Primary Paper**: Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502.

2. **Support Points Theory**: Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.

3. **Optimal Ratios**: Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining*, 15(4), 537-546.

## Performance Considerations

### Optimization Targets

- **Memory efficiency**: Minimize allocations in inner loops
- **Parallel scaling**: Effective multi-threading for large datasets
- **Numerical stability**: Proper handling of edge cases and floating-point precision
- **Stochastic optimization**: Balanced quality vs. speed trade-offs

### Known Challenges

1. **Helmert contrasts**: Efficient categorical variable encoding
2. **k-d tree implementation**: Fast nearest neighbor search (consider NearestNeighbors.jl)
3. **Convergence criteria**: Robust stopping conditions
4. **Memory management**: Large matrix operations with minimal copying

## Code Style Guidelines

- Follow Julia conventions and existing codebase patterns
- Use descriptive variable names matching mathematical notation when possible
- Comprehensive docstrings with parameter descriptions and examples
- Type annotations for public APIs
- Consistent error handling with informative messages

## Integration Notes

- Maintain compatibility with DataFrames.jl ecosystem
- Support both Matrix and DataFrame inputs
- Provide clear migration path from R package usage
- Consider MLJ.jl integration for machine learning workflows
