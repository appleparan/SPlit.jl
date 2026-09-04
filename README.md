# SPlit.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://liam.kim/SPlit.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://liam.kim/SPlit.jl/dev)
[![Test workflow status](https://github.com/appleparan/SPlit.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/appleparan/SPlit.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/appleparan/SPlit.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/Docs.yml?query=branch%3Amain)

[![Python CI](https://github.com/appleparan/SPlit.jl/actions/workflows/PythonCI.yml/badge.svg?branch=main)](https://github.com/appleparan/SPlit.jl/actions/workflows/PythonCI.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/splitiq.svg)](https://pypi.org/project/splitiq/)

[![Coverage](https://codecov.io/gh/appleparan/SPlit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/appleparan/SPlit.jl)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

Distribution-preserving subset selection for tabular data and embeddings,
grown from SPlit ([Joseph and Vakayil, 2022](https://arxiv.org/abs/2012.10945)):
optimal train/test splits, k-fold multiplets, and training-data selection by
support points, kernel herding, twinning, and kernel thinning.

The repository ships two packages with the same features and the same
version number: `SPlit.jl` for Julia, registered in the General registry as
`SPlit`, and [`splitiq`](splitiq/README.md) for Python, a thin `juliacall`
wrapper in which every computation still runs in Julia.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/intuition/split-overview-dark.svg">
  <img alt="The same 240 rows split three ways: all rows, a random 20% test draw that lands unevenly across the clusters, and the SPlit test rows spread through every cluster in proportion" src="docs/src/assets/intuition/split-overview-light.svg">
</picture>

The figure is a real run on 240 two-dimensional rows: a random 20% test draw
lands unevenly across the three clusters, while the rows `datasplit` holds out
follow the shape of the data, and the energy distance between the two sides
drops by more than an order of magnitude.

## Overview

SPlit.jl chooses rows whose empirical distribution stays as close as possible
to a target measure, under the energy distance or the maximum mean
discrepancy. Three choices make up a call:

- **What you get.** `datasplit` returns a train/test partition, `multiplet`
  returns `k` distribution-balanced folds, and `selectrows` (Python:
  `select_rows`) returns just the `n` chosen row indices.
- **What you match.** The data's own distribution by default; `weights`
  matches a quality-weighted version of it, `reference` matches a separate
  target sample while still drawing rows from the data.
- **How rows are chosen.** `SupportPointSplitter`, `HerdingSplitter`,
  `TwinningSplitter`, or `KernelThinningSplitter` (Python:
  `method='support_points'`, `'herding'`, `'twinning'`, `'kernel_thinning'`).

The original method is SPlit: it computes *support points*, the sample of a
given size that minimizes the energy distance to the full data (Mak & Joseph,
2018), and maps each support point to its nearest unclaimed data row (Joseph
& Vakayil, 2022). Unlike random splitting, this makes both the train and test
distributions close to the population distribution, which improves the
reliability of model evaluation. The other three splitters select rows
directly, without the continuous optimization step.

The rows need not be tabular: `standardize = false` passes a numeric matrix
through unchanged, which is what the
[LLM data-selection guide](docs/src/40-llm-data-selection.md) uses on
embedding matrices.

The package also implements the optimal split ratio result of Joseph (2022):
for a linear model with `p` parameters, the test fraction that minimizes
the variance of the fitted model is `γ = 1 / (√p + 1)`.

## Installation

### Julia

```julia
using Pkg
Pkg.add("SPlit")
```

Or from the Julia REPL:

```julia
] add SPlit
```

### Python

```bash
uv add splitiq                 # or: uv add "splitiq[pandas]" for DataFrame input
```

`pip install splitiq` works too; Python 3.12+ is required. Julia itself is
installed automatically by `juliapkg` on the first call to a `splitiq`
function if none is found on the system (one-time, a few minutes); see
[`splitiq/README.md`](splitiq/README.md) for details.

## Quick start

### Julia

```julia
using SPlit, Random

data = randn(MersenneTwister(1), 1_000, 3)
splitter = SupportPointSplitter(ratio = 0.2, rng = MersenneTwister(2))
result = datasplit(splitter, data)

train = data[result, :train]
test = data[result, :test]
splitquality(data, result)                 # energy distance, lower is better
optimal_split_ratio(data[:, 1:2], data[:, 3])

idx = selectrows(HerdingSplitter(), data, 100)          # 100 rows, no partition
folds = multiplet(TwinningSplitter(), data, 5)          # 5 balanced folds
```

### Python

```python
import numpy as np
import splitiq

X = np.random.default_rng(1).standard_normal((1_000, 3))
result = splitiq.datasplit(X, ratio=0.2, seed=2)

train, test = result.apply(X)          # or X[result.train_indices], X[result.test_indices]
splitiq.splitquality(X, result)        # energy distance, lower is better
splitiq.optimal_split_ratio(X[:, :2], X[:, 2])

idx = splitiq.select_rows(X, 100, method='herding')       # 100 rows, no partition
folds = splitiq.multiplet(X, 5, method='twinning')        # 5 balanced folds
```

pandas DataFrames are accepted; `category` and string columns are
Helmert-encoded exactly as a Julia `DataFrame` would be; indices are 0-based.

## API reference

The `splitiq` Python package exposes the same operations under the names in
this table; the remaining subsections document the Julia API in prose.

### Julia and Python names

| Operation | Julia | Python |
|---|---|---|
| Train/test split | `datasplit(splitter, data)` | `datasplit(data, ratio, method=...)` |
| Row selection | `selectrows(splitter, data, n)` | `select_rows(data, n, method=...)` |
| k-fold multiplets | `multiplet(splitter, data, k; strategy)` | `multiplet(data, k, strategy=..., method=...)` |
| Split quality | `splitquality(data, result; kernel)` | `splitquality(data, result, kernel=...)` |
| Compare splitters | `compare(methods, data)`, `best` | `compare(data, methods)`, `SplitComparison.best()` |
| Discrepancies | `energydistance`, `mmd` | `energydistance`, `mmd` |
| Estimators | `Exact`, `Subsample`, `RandomSlices`, `RandomFeatures` | same names |
| Optimal ratio | `optimal_split_ratio(x, y)` | `optimal_split_ratio(x, y)` |
| Method | `SupportPointSplitter`, `HerdingSplitter`, `TwinningSplitter`, `KernelThinningSplitter` | `method='support_points'`, `'herding'`, `'twinning'`, `'kernel_thinning'` |
| Kernel | `EnergyKernel()`, `GaussianKernel(bandwidth)` | `kernel='energy'`, `kernel='gaussian'`, `bandwidth=...` |
| Randomness | `rng = Xoshiro(seed)` on the splitter | `seed=<int>` |
| Indices | 1-based `Vector{Int}` | 0-based numpy arrays |

The keyword arguments `weights`, `reference`, `reference_weights`,
`standardize`, `kappa`, `delta`, `compress`, `start`, and `n_threads` keep
their names in both languages.

### Splitting, selection, and folds

`datasplit` runs a splitter on a `Matrix`, `DataFrame`, or `Vector` and
returns a `SplitResult`, whose `train_indices` and `test_indices` (also
reachable via `data[result, :train]`/`data[result, :test]` indexing or
`train, test = result` destructuring) partition the input rows;
`result.selected` names the side (`:test` or `:train`) holding the chosen
rows. `selectrows(splitter, data, n)` returns just the `n` chosen row
indices, without building a train/test partition.
`multiplet(splitter, data, k; strategy = :sequential)` returns `k`
distribution-balanced folds instead of one train/test pair, under the
`:sequential`, `:halving`, or `:single` strategy.

All three take the same measure and preprocessing keywords: `weights` (a
non-negative score per row, so the selection matches the weighted data),
`reference`/`reference_weights` (make the chosen side approximate a second
dataset instead of `data`'s own distribution, with candidates still drawn
from `data`), and `standardize = false` (use a numeric matrix as it is, with
no encoding, constant-column removal, or scaling — the mode for embeddings;
see the [LLM data-selection guide](docs/src/40-llm-data-selection.md)).

### Selection methods

`SupportPointSplitter` configures a split: which `SplitKernel` to optimize
under (`EnergyKernel`, the kernel whose maximum mean discrepancy is the
energy distance of Mak & Joseph, 2018), the test `ratio`, an optional `kappa`
for stochastic majorization-minimization on large datasets, iteration and
tolerance limits, and the `rng` that drives every random choice.

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
  high-probability MMD guarantee; above half of the rows the selection is the complement of a kernel-thinning selection of the other side.
  `compress = :auto` (the default) switches to Compress++ (Shetty, Dwivedi
  & Mackey, 2022) when `n` is a small fraction of `N` and the target
  measure is the data itself, which removes both `O(N²)` terms;
  `:always`/`:never` force either path.

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
kernels and dataset sizes, and the
[LLM data-selection guide](docs/src/40-llm-data-selection.md) for the
decision table on embedding matrices.

## Algorithm details

1. Preprocessing. Categorical columns are Helmert-encoded, constant columns
   are dropped, and every remaining column is standardized to mean 0 and
   variance 1. `standardize = false` skips this step and uses the numeric
   matrix as it is.
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

10. Shetty, A., Dwivedi, R., & Mackey, L. (2022). Distribution Compression in Near-Linear Time. *ICLR*.

## How to cite

If you use SPlit.jl or splitiq in your work, please cite the software. The
same metadata lives in [CITATION.cff](CITATION.cff), which GitHub renders
under "Cite this repository":

```bibtex
@software{kim2026splitjl,
  author  = {Kim, Jongsu Liam},
  title   = {SPlit.jl: Distribution-preserving subset selection for tabular data and embeddings},
  year    = {2026},
  version = {0.5.2},
  url     = {https://github.com/appleparan/SPlit.jl},
  license = {Apache-2.0}
}
```

Please also cite the papers behind the method you used, listed under
[References](#references); the original splitting method is Joseph &
Vakayil (2022).

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
