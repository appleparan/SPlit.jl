# SPlit.jl Development Context

This document provides context for Claude Code when working on the SPlit.jl project.

## Project Overview

SPlit.jl implements optimal data splitting via the method of support points,
based on three papers:

- Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data
  Splitting. *Technometrics*, 63(4), 492-502.
- Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of
  Statistics*, 46(6A), 2562-2592.
- Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical
  Analysis and Data Mining*, 15(4), 537-546.

Given a dataset, SPlit computes support points — the sample of a chosen size
that minimizes the energy distance to the full data — and maps each support
point to its nearest unclaimed data row via a k-d tree. The claimed rows form
the smaller subset, the rest form the larger one; both track the population
distribution more closely than a random split would.

## Public API

Exported from `SPlit`, grouped by role:

### Splitting

- `SplitKernel`, `EnergyKernel`: discrepancy kernel selecting what the
  optimizer minimizes; `EnergyKernel` is the kernel whose maximum mean
  discrepancy is the energy distance (Mak & Joseph, 2018).
- `SupportPointSplitter`: split configuration — `kernel`, `ratio`, `kappa`
  (stochastic subsample size), `max_iterations`, `tolerance`, `n_threads`,
  `rng`, `verbose`.
- `SplitResult`: index partition plus convergence report (`train_indices`,
  `test_indices`, `converged`, `iterations`); iterable as `train, test =
  result` and indexable as `data[result, :train]`.
- `datasplit(splitter, data)`: run a split on a `Matrix`, `DataFrame`, or
  `Vector`, returning a `SplitResult`.
- `train_indices(result)`, `test_indices(result)`: accessors.

### Quality diagnostics

- `energydistance(X, Y)`: energy distance between two samples — exact by
  default, subsampled-estimator for large inputs.
- `splitquality(data, result)`: energy distance between the train and test
  rows of a `SplitResult`; lower is better.

### Optimal ratio

- `optimal_split_ratio(x, y; method = :simple)`: the test-set fraction
  `γ = 1 / (√p + 1)` from Joseph (2022, Eq. 11), `p` being the number of
  model parameters.

### Comparison

- `compare(methods, data)`: run multiple `SupportPointSplitter`
  configurations on the same data, each scored by `splitquality`.
- `SplitComparison`: result container, convertible to a `DataFrame`.
- `best(comparison)`: the method/result pair with the lowest energy
  distance.

## Implementation Status

### Current Structure

```text
src/
├── SPlit.jl              # Module entry point: includes source files, defines exports
├── kernels.jl             # SplitKernel hierarchy (EnergyKernel)
├── preprocessing.jl       # Helmert encoding, constant-column removal, standardization
├── optimizer.jl           # Support-point MM optimization (full-data and stochastic)
├── kdtree_selection.jl    # Sequential nearest-neighbor row selection via k-d tree
├── splitter.jl            # SupportPointSplitter, SplitResult, datasplit
├── quality.jl             # energydistance / splitquality diagnostics
├── ratio.jl               # optimal_split_ratio (Joseph 2022)
└── comparison.jl          # compare / SplitComparison / best
```

### Dependencies

- CategoricalArrays.jl: detecting and iterating categorical columns for
  Helmert encoding
- DataFrames.jl: DataFrame input/output support
- Distances.jl: pairwise Euclidean distances for the energy distance
- LinearAlgebra.jl: matrix operations in the optimizer
- NearestNeighbors.jl: k-d tree for nearest-neighbor row selection
- Random.jl: RNG threading through splitting and optimization
- Statistics.jl: mean/std for standardization and quality estimates
- StatsBase.jl: sampling without replacement for stochastic optimization and
  the subsampled energy-distance estimator

## Algorithm Implementation Notes

### Data preprocessing (`preprocess`)

1. Error on missing values.
2. Encode categorical columns with Helmert contrasts.
3. Drop constant columns.
4. Standardize all remaining columns to mean 0, variance 1.

### Support-point computation (`support_points`)

1. Initialize with a jittered random sample from the data, clamped to the
   data's bounding box.
2. Apply the kernel's closed-form majorization-minimization update, which
   decreases the energy-distance objective monotonically (Mak & Joseph,
   2018).
3. When `kappa < n`, use the stochastic variant of Joseph & Vakayil (2021):
   resample `kappa` rows per iteration and stabilize the update with
   running averages.
4. Converge when the point-wise update falls below `tolerance`, or stop at
   `max_iterations`.

### Nearest-neighbor selection (`select_nearest`)

- Build a k-d tree (NearestNeighbors.jl) over the data rows.
- Each support point, in order, claims its nearest not-yet-claimed row;
  when every returned neighbor is already claimed, the k-nearest-neighbor
  query doubles its `k` and retries.

## Test Suite

- `test/test_preprocessing.jl`: Helmert encoding, constant-column removal,
  standardization, missing-value handling.
- `test/test_quality.jl`: `energydistance` and `splitquality`, exact vs.
  subsampled-estimator agreement.
- `test/test_optimizer.jl`: MM update monotone descent of the energy
  objective, full-data and stochastic modes.
- `test/test_kdtree_selection.jl`: k-d tree and brute-force nearest-neighbor
  selection equivalence.
- `test/test_splitter.jl`: `SupportPointSplitter`/`datasplit`/`SplitResult`
  construction, indexing, iteration, and reproducibility under a fixed
  `rng`.
- `test/test_ratio.jl`: `optimal_split_ratio`, including `γ = 1/(√p+1)` spot
  checks against Joseph (2022, Eq. 11).
- `test/test_comparison.jl`: `compare`/`SplitComparison`/`best`.
- `test/test_properties.jl`: integration properties spanning the whole
  public API — support-point splits achieving lower energy distance than
  random splits, and stochastic optimization staying within a bounded
  factor of full-data optimization quality.

Correctness is verified against the properties the three papers establish,
not against any external reference implementation.

## Development Commands

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"

# Build documentation
julia --project=docs/ docs/make.jl

# Format code (JuliaFormatter runs via pre-commit; it is not a package dependency)
pre-commit run julia-formatter -a
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

## Code Style Guidelines

- Use descriptive variable names matching mathematical notation when possible
- Comprehensive docstrings with parameter descriptions and examples on public API
- Type annotations on public API

## Integration Notes

- Maintain compatibility with the DataFrames.jl ecosystem; support Matrix,
  DataFrame, and Vector inputs
- Consider MLJ.jl integration for machine learning workflows
