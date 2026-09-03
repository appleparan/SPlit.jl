# [Design experiments](@id design-experiments)

These are measurements that fixed decisions in the code: `splitquality`'s
automatic estimator fallback, and kernel herding's exact (not approximated)
data term. They are not the splitter comparison, which is on the
[Benchmarks](@ref benchmarks) page.

## [Estimator selection](@id estimator-selection)

Selection experiment for the `DiscrepancyEstimator` `splitquality` falls
back to above `exact_threshold`: on the four datasets from the
[Benchmarks](@ref benchmarks) page at N = 10,000, absolute error against the
exact value and wall time of every candidate, measured on the split from
`support points · energy`, `herding · energy` (energy distance) and
`herding · gaussian` (MMD), over 5 rng seeds. Full table:
[`assets/benchmarks/estimators.md`](assets/benchmarks/estimators.md).
Reproduce with:

```sh
julia -t auto --project=benchmark benchmark/estimators.jl
```

| estimator | max abs error (worst over rows) | mean time (s) (over rows) |
|---|---:|---:|
| `Subsample(2000, 8)` (EnergyKernel) | 0.00197 | 0.208 |
| `RandomSlices(64)` | 0.00014 | 0.0189 |
| `RandomSlices(256)` | 7.65e-5 | 0.0835 |
| `RandomSlices(1024)` | 4.14e-5 | 0.331 |
| `Subsample(2000, 8)` (GaussianKernel) | 0.000175 | 0.298 |
| `RandomFeatures(512)` | 5.36e-7 | 0.0658 |
| `RandomFeatures(2048)` | 3.85e-7 | 0.325 |

Rule: an estimator becomes the automatic fallback if, at no more than
`Subsample(2000, 8)`'s mean wall time, its worst-case max error over every
row is at most one third of `Subsample(2000, 8)`'s; otherwise
`Subsample(2000, 8)` stays the fallback.

Decision: `RandomSlices(64)` for `EnergyKernel` and `RandomFeatures(512)`
for `GaussianKernel`. Worst-case max error is 14× lower for
`RandomSlices(64)` (0.00197 to 0.00014) at 9% of `Subsample(2000, 8)`'s mean
time, and 330× lower for `RandomFeatures(512)` (0.000175 to 5.36e-7) at 22%
of its time. `ENERGY_FALLBACK` and `GAUSSIAN_FALLBACK` in `quality.jl` are
set accordingly.

## [Approximate herding data terms (rejected)](@id herding-estimators-rejected)

Kernel herding's data term (`mean_l k(x_i, x_l)`, Chen, Welling & Smola
2010, Eq. 8) was tried with `RandomSlices`/`RandomFeatures` approximations
and rejected: all candidate rows share the same random directions or
features, so the estimator's noise is correlated across rows, and greedy
`argmax` selection tracks that noise rather than averaging it out. In the
table below the smallest budgets (k = 64 and 256, D = 512) select subsets
*worse than a random subset*. Larger budgets beat random but stay 7-35×
from exact herding, and only k = 8192 and D = 32768 come within about 3.5×.
At that budget the estimator's own cost
(`O(kN log N)` for slices, `O(NDp)` for Fourier features) matches the exact
`O(N²)` data term for `N` around 10⁵. `RandomSlices`/`RandomFeatures` remain
available for `energydistance`/`mmd` quality diagnostics only;
`HerdingSplitter`'s data term is exact only. N = 1500, p = 3, n = 300, 3 rng
seeds per row. Full table:
[`assets/benchmarks/herding_estimators.md`](assets/benchmarks/herding_estimators.md).
Reproduce with:

```sh
julia -t auto --project=benchmark benchmark/herding_estimators.jl
```

| kernel | estimator | selected-subset discrepancy (3 seeds) | exact herding | random | ratio to exact |
|---|---|---:|---:|---:|---:|
| EnergyKernel | RandomSlices(64) | 0.0255, 0.0561, 0.0692 (mean 0.0503) | 0.000643 | 0.00713 | 78.2× |
| EnergyKernel | RandomSlices(256) | 0.0181, 0.0222, 0.0262 (mean 0.0222) | 0.000643 | 0.00713 | 34.5× |
| EnergyKernel | RandomSlices(2048) | 0.00467, 0.00486, 0.0108 (mean 0.00679) | 0.000643 | 0.00713 | 10.6× |
| EnergyKernel | RandomSlices(8192) | 0.00138, 0.00252, 0.00228 (mean 0.00206) | 0.000643 | 0.00713 | 3.2× |
| GaussianKernel | RandomFeatures(512) | 0.00448, 0.00385, 0.00407 (mean 0.00413) | 5.88e-5 | 0.00268 | 70.3× |
| GaussianKernel | RandomFeatures(2048) | 0.0019, 0.00168, 0.00171 (mean 0.00177) | 5.88e-5 | 0.00268 | 30.0× |
| GaussianKernel | RandomFeatures(8192) | 0.000474, 0.000503, 0.000343 (mean 0.00044) | 5.88e-5 | 0.00268 | 7.48× |
| GaussianKernel | RandomFeatures(32768) | 0.000203, 0.000225, 0.000187 (mean 0.000205) | 5.88e-5 | 0.00268 | 3.48× |

## [Weighted `kappa` subsampling](@id weighted-kappa)

With sample weights, the stochastic MM can draw its `kappa` rows in two
ways: uniformly, rescaling the drawn weights to mean one within the
subsample (`:uniform`), or in proportion to the weights, treating the
subsample as uniform (`:proportional`). Both are implemented behind the
internal `_subsampling` keyword of `support_points`; the default is
`:uniform`. Measured on `normal-10d` and `uniform-5d` at N = 10,000 with
log-normal weights and a 10:1 two-cluster profile, `kappa` ∈ {500, 2000},
five rng seeds each; the score is the weighted energy distance between the
selected rows and the full data under the weights. At `kappa` = 500 the
mean score was 0.0217 (se 0.00086) for `:uniform` and 0.0226 (se 0.0012)
for `:proportional`; `:uniform` had the lower mean but the two rules were
within one combined standard error of each other, so `:uniform` stays the
default for its simplicity (sampling without replacement in proportion to
`w` also does not give inclusion probabilities exactly proportional to
`w`, so `:proportional` is only an approximation of the weighted
distribution in the first place). Full table:
[`assets/benchmarks/weighted_kappa.md`](assets/benchmarks/weighted_kappa.md).
Reproduce with:

```sh
julia -t auto --project=benchmark benchmark/weighted_kappa.jl
```

## [Nearest-neighbor structure for twinning](@id twinning-trees)

`TwinningSplitter` answers two nearest-neighbor queries per group. A k-d
tree is the paper's choice, but its pruning weakens as the dimension
grows, so the rows can also be scanned by brute force. Measured on
standard-normal data at N = 1,000, 10,000, and 100,000 and `ratio = 0.2`,
serial (Julia 1.10.12, AMD Ryzen 7 7800X3D): minimum of three runs for
N ≤ 10,000, a single run at N = 100,000.

| N | p | k-d tree (s) | brute force (s) | brute / k-d |
|---:|---:|---:|---:|---:|
| 1000 | 2 | 0.000326 | 0.00127 | 3.9 |
| 1000 | 10 | 0.00243 | 0.00186 | 0.768 |
| 1000 | 50 | 0.00719 | 0.00418 | 0.581 |
| 1000 | 200 | 0.0296 | 0.0166 | 0.561 |
| 1000 | 768 | 0.236 | 0.0341 | 0.145 |
| 10000 | 2 | 0.0055 | 0.101 | 18.4 |
| 10000 | 10 | 0.114 | 0.157 | 1.37 |
| 10000 | 50 | 0.656 | 0.306 | 0.466 |
| 10000 | 200 | 1.38 | 1.11 | 0.804 |
| 10000 | 768 | 7.38 | 3.71 | 0.503 |
| 100000 | 2 | 0.0693 | 15.2 | 220.0 |
| 100000 | 10 | 4.15 | 19.7 | 4.75 |
| 100000 | 50 | 71.6 | 37.7 | 0.527 |
| 100000 | 200 | 328.0 | 293.0 | 0.893 |
| 100000 | 768 | 999.0 | 833.0 | 0.834 |

Brute force is faster at every measured p ≥ 50 for every N (1.9-2.1x at
p = 50; 1.1-1.8x at p = 200; 1.2-6.9x at p = 768), while the k-d tree wins
at p ≤ 10 once N ≥ 10,000 and by a widening margin as N grows (18x at
N = 10⁴ and 220x at N = 10⁵ for p = 2; 1.4x and 4.8x for p = 10). At
N = 1,000 brute force is marginally faster at p = 10 too, but both take
about 2 ms there, so the threshold is set by the larger sizes. The
crossover therefore does not move with N in the measured range, so
`TWINNING_BRUTE_FORCE_DIMENSION` stays 50. This settles the roadmap's open
question on high-dimensional nearest neighbors for M3. Reproduce with:

```sh
julia --project=benchmark benchmark/twinning_trees.jl
```
