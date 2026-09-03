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
standard-normal data at N = 10,000 and `ratio = 0.2`, minimum of three
runs, serial (Julia 1.10.12, AMD Ryzen 7 7800X3D):

| p | k-d tree (s) | brute force (s) | brute / k-d |
|---:|---:|---:|---:|
| 2 | 0.0051 | 0.113 | 22.1 |
| 10 | 0.122 | 0.172 | 1.41 |
| 50 | 0.662 | 0.326 | 0.492 |
| 200 | 1.5 | 1.16 | 0.776 |
| 768 | 8.53 | 4.25 | 0.499 |

Brute force first reaches the 1.5x-faster bar at p = 50 (2.03x faster; it
stays faster at p = 200 and p = 768 too, at 1.29x and 2.00x), so
`TWINNING_BRUTE_FORCE_DIMENSION = 50`. This settles the roadmap's open
question on high-dimensional nearest neighbors for M3. Reproduce with:

```sh
julia --project=benchmark benchmark/twinning_trees.jl
```
