# SPlit.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://liam.kim/SPlit.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://liam.kim/SPlit.jl/dev)
[![Test workflow status](https://github.com/appleparan/SPlit.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/appleparan/SPlit.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/appleparan/SPlit.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Docs.yml?query=branch%3Amain)

[![Coverage](https://codecov.io/gh/appleparan/SPlit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/appleparan/SPlit.jl)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

A Julia implementation of optimal data splitting via support points, based on
[Joseph and Vakayil (2022)](https://arxiv.org/abs/2012.10945).

## Overview

SPlit.jl splits a dataset into training and test sets so that both subsets
represent the original data distribution as closely as possible. It does so
by computing *support points*, the sample of a given size that minimizes
the energy distance to the full data (Mak & Joseph, 2018), and mapping each
support point to its nearest unclaimed data row (Joseph & Vakayil, 2022).
Unlike random splitting, this makes both the train and test distributions
close to the population distribution, which improves the reliability of
model evaluation.

The package also implements the optimal split ratio result of Joseph (2022):
for a linear model with `p` parameters, the test fraction that minimizes
the variance of the fitted model is `γ = 1 / (√p + 1)`.

## Installation

```julia
using Pkg
Pkg.add("SPlit")
```

Or from the Julia REPL:

```julia
] add SPlit
```

## Quick start

```julia
using SPlit, Random

data = randn(MersenneTwister(1), 1_000, 3)
splitter = SupportPointSplitter(ratio = 0.2, rng = MersenneTwister(2))
result = datasplit(splitter, data)

train = data[result, :train]
test = data[result, :test]
splitquality(data, result)                 # energy distance, lower is better
optimal_split_ratio(data[:, 1:2], data[:, 3])
```

## Python

The same implementation is available from Python as
[`splitiq`](splitiq/README.md), a `juliacall` wrapper that keeps Julia as the
only implementation:

```python
import splitiq
result = splitiq.datasplit(X, ratio=0.2, seed=2)
train, test = result.apply(X)
```

## API reference

### Splitting

`SupportPointSplitter` configures a split: which `SplitKernel` to optimize
under (`EnergyKernel`, the kernel whose maximum mean discrepancy is the
energy distance of Mak & Joseph, 2018), the test `ratio`, an optional `kappa`
for stochastic majorization-minimization on large datasets, iteration and
tolerance limits, and the `rng` that drives every random choice. `datasplit`
runs it on a `Matrix`, `DataFrame`, or `Vector` and returns a `SplitResult`,
whose `train_indices` and `test_indices` (also reachable via `data[result,
:train]`/`data[result, :test]` indexing or `train, test = result`
destructuring) partition the input rows; `result.selected` names the side
(`:test` or `:train`) holding the chosen rows. `selectrows(splitter, data,
n; reference = ...)` returns just the `n` chosen row indices, without
building a train/test partition. `datasplit`'s `reference` keyword (and
`selectrows`'s) makes the chosen side approximate a second dataset instead
of `data`'s own distribution, with candidates still drawn from `data`.

- `GaussianKernel(bandwidth = :median)`: support points minimize the squared
  maximum mean discrepancy (Gretton et al., 2012) via projected gradient
  descent with Armijo backtracking, instead of the energy-distance MM step.
  It has no `kappa` mode; the resolved bandwidth is stored in
  `result.method.kernel`.
- `HerdingSplitter`: builds the smaller subset directly by greedy kernel
  herding (Chen, Welling & Smola, 2010) instead of computing support
  points, under either `EnergyKernel` or `GaussianKernel`. It is
  deterministic given the data and a numeric kernel, with an exact
  (`O(N²)`) data term at every dataset size.
- `TwinningSplitter`: partitions by sequential nearest-neighbor twinning
  (Vakayil & Joseph, 2022) under the energy distance, with no kernel or
  optimizer options. It is deterministic by default.
- `KernelThinningSplitter`: generalized kernel thinning with the target
  kernel (Dwivedi & Mackey, 2022, 2024) under `EnergyKernel` or
  `GaussianKernel`: KT-SPLIT halves a shuffled sequence of rows by
  randomized kernel halving, and KT-SWAP keeps the candidate closest to
  the target measure and refines it by single-row swaps. It carries a
  high-probability MMD guarantee and selects at most half of the rows.
- `multiplet(splitter, data, k; strategy = :sequential)`: splits `data`
  into `k` distribution-balanced folds instead of one train/test pair,
  using any splitter under the `:sequential`, `:halving`, or `:single`
  strategy.

```julia
herding = HerdingSplitter(kernel = EnergyKernel(), rng = MersenneTwister(2))
result = datasplit(herding, data)
```

### Quality diagnostics

`energydistance(X, Y)` computes the energy distance between two samples
(exactly, or via a `DiscrepancyEstimator` for large inputs). `mmd(X, Y,
kernel)` generalizes this to any `SplitKernel` (the squared maximum mean
discrepancy). `splitquality(data, result; kernel = EnergyKernel())` applies
the matching statistic to the train/test partition of a `SplitResult`,
switching to an estimator automatically once the row count crosses a
threshold. Lower values mean the two sides are closer in distribution.

```julia
gauss = SupportPointSplitter(kernel = GaussianKernel(), rng = MersenneTwister(3))
result = datasplit(gauss, data)
splitquality(data, result; kernel = result.method.kernel)   # MMD under the fitted kernel

energydistance(X, Y; estimator = RandomSlices(256))
mmd(X, Y, GaussianKernel(1.0); estimator = RandomFeatures(2048))
```

### Optimal ratio

`optimal_split_ratio(x, y; method = :simple)` returns the test-set fraction
`γ = 1 / (√p + 1)` from Joseph (2022, Eq. 11), where `p` is the number of
model parameters (predictor columns after preprocessing, plus the
intercept).

### Comparison

`compare(methods, data)` runs several splitter configurations
(any splitter: `SupportPointSplitter`, `HerdingSplitter`,
`TwinningSplitter`, or `KernelThinningSplitter`) on the same data and scores
each with `splitquality`, returning a
`SplitComparison` (convertible to a `DataFrame`); `best(comparison)` returns
the method/result pair with the lowest energy distance.

### Benchmarks

See the [Benchmarks](docs/src/20-benchmarks.md)
page for how the four splitters compare across
kernels and dataset sizes.

## Algorithm details

1. Preprocessing. Categorical columns are Helmert-encoded, constant columns
   are dropped, and every remaining column is standardized to mean 0 and
   variance 1.
2. Support-point computation. The kernel's majorization-minimization update
   moves a candidate point set, sweep by sweep, to minimize its energy
   distance to the data (Mak & Joseph, 2018). For large `n`, `kappa`
   switches to the stochastic variant that resamples rows each iteration.
3. Nearest-neighbor assignment. Each support point claims its nearest
   unclaimed data row via a k-d tree (Joseph & Vakayil, 2022).
4. Partitioning. The claimed rows form the smaller subset and the rest form
   the larger one; `ratio` decides which of the two is the test set.

## References

1. Joseph, V. R., & Vakayil, A. (2022). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 64(2), 166-176. [DOI](https://doi.org/10.1080/00401706.2021.1921037)

2. Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.

3. Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining: The ASA Data Science Journal*, 15(4), 531-538.

4. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. *Journal of Machine Learning Research*, 13, 723-773.

5. Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel Herding. *UAI*, 109-116.

6. Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NIPS*, 20.

7. Vakayil, A., & Joseph, V. R. (2022). Data Twinning. *Statistical Analysis and Data Mining*, 15(5), 598-610.

8. Dwivedi, R., & Mackey, L. (2022). Generalized Kernel Thinning. *ICLR*.

9. Dwivedi, R., & Mackey, L. (2024). Kernel Thinning. *Journal of Machine Learning Research*, 25(152), 1-77.

## How to cite

If you use SPlit.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/appleparan/SPlit.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first take a look at our [contributing guide directly on GitHub](docs/src/90-contributing.md).

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
