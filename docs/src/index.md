```@meta
CurrentModule = SPlit
```

# SPlit.jl

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

## Kernels and splitters

`SupportPointSplitter` accepts any `SplitKernel`. The default,
`EnergyKernel`, minimizes the energy distance of Mak & Joseph (2018).
`GaussianKernel` minimizes the squared maximum mean discrepancy (MMD²,
Gretton et al., 2012) instead, by projected gradient descent with Armijo
backtracking; see [Methods](@ref methods) for both objectives.

```julia
gauss = SupportPointSplitter(kernel = GaussianKernel(), rng = MersenneTwister(3))
result = datasplit(gauss, data)
result.method.kernel            # GaussianKernel with the resolved bandwidth
splitquality(data, result; kernel = result.method.kernel)   # MMD under the fitted kernel
```

`HerdingSplitter` builds the smaller subset directly by greedy kernel
herding (Chen, Welling & Smola, 2010) instead of computing support points:

```julia
herd = HerdingSplitter(kernel = GaussianKernel())   # deterministic, optimizer-free
result = datasplit(herd, data)
```

See [Benchmarks](@ref benchmarks) for how the two splitters compare across
kernels and dataset sizes.

## API Reference

See the [Reference](@ref reference) section for complete API documentation.

## Algorithm Details

1. **Preprocessing**: categorical columns are Helmert-encoded, constant
   columns are dropped, and every remaining column is standardized to mean 0
   and variance 1.
2. **Support-point computation**: the kernel's majorization-minimization
   update iteratively moves a candidate point set to minimize its energy
   distance to the data (Mak & Joseph, 2018); `kappa` switches to the
   stochastic variant that resamples rows each iteration for large `n` —
   or, for `GaussianKernel`, projected gradient descent on the squared MMD;
   see [Methods](@ref methods).
3. **Nearest-neighbor assignment**: each support point claims its nearest
   not-yet-claimed data row via a k-d tree (Joseph & Vakayil, 2021).
4. **Partitioning**: the claimed rows form the smaller subset; the rest form
   the larger one, split into train/test according to `ratio`.

## References

1. Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502. [DOI](https://arxiv.org/abs/2012.10945)

2. Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.

3. Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining: The ASA Data Science Journal*, 15(4), 537-546.

4. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. *Journal of Machine Learning Research*, 13, 723-773.

5. Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel Herding. *UAI*, 109-116.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
