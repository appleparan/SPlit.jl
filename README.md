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

A Julia implementation of optimal data splitting via support points, based on
[Joseph and Vakayil (2021)](https://arxiv.org/abs/2012.10945).

## Overview

SPlit.jl splits a dataset into training and test sets so that both subsets
represent the original data distribution as closely as possible. It does so
by computing *support points* — the sample of a given size that minimizes
the energy distance to the full data (Mak & Joseph, 2018) — and mapping each
support point to its nearest unclaimed data row (Joseph & Vakayil, 2021).
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

## Quick Start

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

## API Reference

### Splitting

`SupportPointSplitter` configures a split: which `SplitKernel` to optimize
under (`EnergyKernel`, the kernel whose maximum mean discrepancy is the
energy distance of Mak & Joseph, 2018), the test `ratio`, an optional `kappa`
for stochastic majorization-minimization on large datasets, iteration and
tolerance limits, and the `rng` that drives every random choice. `datasplit`
runs it on a `Matrix`, `DataFrame`, or `Vector` and returns a `SplitResult`,
whose `train_indices` and `test_indices` (also reachable via `data[result,
:train]`/`data[result, :test]` indexing or `train, test = result`
destructuring) partition the input rows.

### Quality diagnostics

`energydistance(X, Y)` computes the energy distance between two samples
(exactly, or via random subsampling for large inputs). `splitquality(data,
result)` applies it to the train/test partition of a `SplitResult`, switching
to the subsampled estimator automatically once the row count crosses a
threshold — lower values indicate a split whose two sides are more alike in
distribution.

### Optimal ratio

`optimal_split_ratio(x, y; method = :simple)` returns the test-set fraction
`γ = 1 / (√p + 1)` from Joseph (2022, Eq. 11), where `p` is the number of
model parameters (predictor columns after preprocessing, plus the
intercept).

### Comparison

`compare(methods, data)` runs several `SupportPointSplitter` configurations
on the same data and scores each with `splitquality`, returning a
`SplitComparison` (convertible to a `DataFrame`); `best(comparison)` returns
the method/result pair with the lowest energy distance.

## Algorithm Details

1. **Preprocessing**: categorical columns are Helmert-encoded, constant
   columns are dropped, and every remaining column is standardized to mean 0
   and variance 1.
2. **Support-point computation**: the kernel's majorization-minimization
   update iteratively moves a candidate point set to minimize its energy
   distance to the data (Mak & Joseph, 2018); `kappa` switches to the
   stochastic variant that resamples rows each iteration for large `n`.
3. **Nearest-neighbor assignment**: each support point claims its nearest
   not-yet-claimed data row via a k-d tree (Joseph & Vakayil, 2021).
4. **Partitioning**: the claimed rows form the smaller subset; the rest form
   the larger one, split into train/test according to `ratio`.

## References

1. Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502. [DOI](https://arxiv.org/abs/2012.10945)

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
