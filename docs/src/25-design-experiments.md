# [Design experiments](@id design-experiments)

These are measurements that fixed decisions in the code: `splitquality`'s
automatic estimator fallback, kernel herding's exact (not approximated)
data term, the weighted `kappa` rule, twinning's nearest-neighbor
structure, and kernel thinning's Compress++ cost rule. They are not the
splitter comparison, which is on the [Benchmarks](@ref benchmarks) page.

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

The brute-force structure above was `BruteTree` (NearestNeighbors.jl); it
was replaced by the plain-matrix search on 2026-09-05, see
[Matrix brute-force search](@ref matrix-brute-force).

## [Matrix brute-force search](@id matrix-brute-force)

`BruteTree` and `KDTree` (NearestNeighbors.jl) both specialize their search
code on `SVector{p, Float64}`, so compilation is per-width: first call
22 s at p = 1,536, 110 s at 3,072, over 7 minutes at 6,144, and a compiler
failure at 12,288 (measured in the time-series example, 2026-09-05). The
search itself takes well under a second there. `MatrixSearch` keeps the
data as a plain `Matrix{Float64}` (columns as points) and answers queries
with explicit `@inbounds @simd` distance loops, so it compiles once
regardless of width.

Measured on standard-normal data through `SPlit.preprocess`, serial
(Julia 1.10.12, AMD Ryzen 7 7800X3D): minimum of repeats where noted,
single runs at N = 100,000.

### Twinning: search structure wall time

First call: one warm-up per `p`, on a 500-row/100-group slice, in this
process.

| p | k-d tree first call (s) | brute tree first call (s) | matrix first call (s) |
|---:|---:|---:|---:|
| 50 | 0.72 | 0.151 | 0.286 |
| 200 | 1.6 | 0.179 | 0.00146 |
| 768 | 13.5 | 0.188 | 0.0253 |

| N | p | k-d tree (s) | brute tree (s) | matrix (s) | brute/matrix | kdtree/matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 50 | 0.0081 | 0.00445 | 0.00353 | 1.26 | 2.3 |
| 10000 | 50 | 0.632 | 0.314 | 0.323 | 0.973 | 1.95 |
| 1000 | 200 | 0.0279 | 0.0114 | 0.00555 | 2.06 | 5.02 |
| 10000 | 200 | 1.45 | 1.09 | 0.663 | 1.65 | 2.19 |
| 1000 | 768 | 0.235 | 0.0359 | 0.0158 | 2.27 | 14.9 |
| 10000 | 768 | 8.89 | 4.15 | 2.17 | 1.91 | 4.09 |
| 100000 | 50 | 70.0 | 39.6 | 34.8 | 1.14 | 2.01 |

Versus `BruteTree`, `MatrixSearch` is within 3% or faster everywhere, and
1.65-2.3x faster at p ≥ 200; its first call is not width-specific (0.29 s
once, then 0.0015-0.025 s at the next widths, against 0.15-0.19 s for
`BruteTree` and 0.72-13.5 s for the k-d tree). The threshold is unchanged:
the matrix search is still 1.95-2.3x faster than the k-d tree at p = 50
for every N, so `TWINNING_BRUTE_FORCE_DIMENSION` stays 50. `:brute_tree`
stays available as an explicit, never-default `search` option so this
benchmark stays reproducible against the structure it replaces.

An earlier run of this same comparison found `MatrixSearch` 13% slower
than `BruteTree` at N = 10,000/p = 50 (brute/matrix = 0.884), which raised
a decision-rule concern; the cause was per-query sorting scratch in
`_knn`, and replacing it with k independent argmin passes closed the gap
(brute/matrix = 0.973, the value in the table above).

### `select_nearest`: search structure wall time

First call: one warm-up per row, on a 500-row/100-point slice, in this
process. Query points are data rows plus N(0, 0.1) noise.

| N | p | k-d tree first call (s) | matrix first call (s) | k-d tree (s) | matrix (s) | kdtree/matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 2 | 0.00499 | 7.71e-5 | 0.00262 | 0.0305 | 0.0857 |
| 10000 | 10 | 0.335 | 0.000231 | 0.00793 | 0.0921 | 0.0861 |
| 10000 | 50 | 0.167 | 0.000546 | 0.0411 | 0.221 | 0.185 |
| 10000 | 200 | 0.321 | 0.000924 | 0.363 | 0.429 | 0.847 |
| 10000 | 768 | 1.62 | 0.00345 | 4.95 | 1.73 | 2.86 |
| 100000 | 10 | – | – | 0.173 | 9.4 | 0.0184 |
| 100000 | 50 | – | – | 2.32 | 22.9 | 0.102 |

First-call columns show "–" for the 100000-row rows: their widths (10, 50)
already ran, and compiled, earlier in this process at the 10000-row rows
above, so no genuine first call remains to measure there.

At N = 10,000 with 2,000 query points, the matrix search is within about
15% of the k-d tree at p = 200 (0.429 s vs 0.363 s) and 2.86x faster at
p = 768; the k-d tree is 5-12x faster at p ≤ 50 and 10-54x faster at
N = 100,000 for p ≤ 50. The k-d tree's first call is width-specific
(0.32 s at p = 200, 1.62 s at p = 768), the matrix search's is not.
`NEAREST_BRUTE_FORCE_DIMENSION` stays 200: that is set by the crossover
falling between p = 200 and p = 768 together with the k-d tree's first-call
cost growing sharply over that same range, not because the matrix search
already matches the k-d tree at 200.

### First call at extreme width

`:matrix` only — the widths `BruteTree`/`KDTree` could not compile
(N = 200, n = 20). Each width runs `selectrows` in a fresh Julia process,
so both columns are genuine first calls and include Julia startup, package
load, and compilation, not just the search structure's own compile time:

| p | twinning first call (s) | select_nearest first call (s) |
|---:|---:|---:|
| 3072 | 1.18 | 1.84 |
| 6144 | 1.27 | 1.82 |
| 12288 | 1.35 | 1.85 |

Against 110 s at 3,072, over 7 minutes at 6,144, and a compiler error at
12,288 for the static-vector structures (measured in the time-series
example, 2026-09-05), `MatrixSearch`'s fresh-process first call is flat at
1.18-1.35 s for twinning and 1.82-1.85 s for `select_nearest` across all
three widths.

Below both thresholds, results are unchanged (the k-d tree paths are
untouched); above them, the matrix search and the tree it replaces agree
except on exact ties, which continuous data does not produce. Reproduce
with:

```sh
julia --project=benchmark benchmark/brute_force.jl
```

## [Compress++ cost rule](@id compress-rule)

`KernelThinningSplitter(compress = :auto)` runs [Compress++](@ref compress)
when the estimated kernel-evaluation count `4^g N (4 log₄ N + 1)` is below
plain kernel thinning's `1.5 N²`, with `g = max(4, ⌈log₂(2n/√N)⌉)`. The
estimate was checked against wall time and quality on standard-normal
matrices passed with `standardize = false` (the embedding path): N = 10,000
at p = 10 and 384, N = 100,000 at p = 10, and n/N from 1% to 20%, on
16 threads (Julia 1.10.12, AMD Ryzen 7 7800X3D). Each cell runs three
splitter seeds; the time is the minimum over them and ED, the energy
distance between the selected rows and the full data, is their mean.
`plain` is `compress = :never`, `compress++` is `:always`, `auto fires` is
what `:auto` would choose. ED is exact when N + n ≤ 20,000 and otherwise
`RandomSlices(64)` (the `splitquality` fallback, with the same slices in
every column); `ED random` is the mean over three uniform random subsets.

| N | p | n | n/N | auto fires | g | plain (s) | compress++ (s) | plain / compress++ | ED plain | ED compress++ | ED random |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 10 | 100 | 0.01 | yes | 4 | 1.12 | 0.826 | 1.36 | 0.0139 | 0.0146 | 0.0371 |
| 10000 | 10 | 500 | 0.05 | yes | 4 | 1.54 | 0.804 | 1.92 | 0.00222 | 0.00237 | 0.00792 |
| 10000 | 10 | 1000 | 0.1 | no | 5 | 1.84 | 1.62 | 1.14 | 0.000978 | 0.00102 | 0.00415 |
| 10000 | 10 | 2000 | 0.2 | no | 6 | 1.66 | 2.27 | 0.732 | 0.000406 | 0.000414 | 0.00172 |
| 10000 | 384 | 100 | 0.01 | yes | 4 | 10.9 | 11.2 | 0.975 | 0.156 | 0.163 | 0.268 |
| 10000 | 384 | 500 | 0.05 | yes | 4 | 15.3 | 11.6 | 1.31 | 0.027 | 0.0278 | 0.0506 |
| 10000 | 384 | 1000 | 0.1 | no | 5 | 16.7 | 22.0 | 0.761 | 0.0126 | 0.0127 | 0.0248 |
| 10000 | 384 | 2000 | 0.2 | no | 6 | 16.1 | 24.6 | 0.654 | 0.00554 | 0.00556 | 0.0113 |
| 100000 | 10 | 1000 | 0.01 | yes | 4 | 44.2 | 6.39 | 6.92 | 0.00104 | 0.00112 | 0.00456 |
| 100000 | 10 | 5000 | 0.05 | yes | 5 | 47.5 | 17.3 | 2.75 | 0.000163 | 0.000189 | 0.000817 |
| 100000 | 10 | 10000 | 0.1 | yes | 6 | 44.9 | 31.7 | 1.42 | 7.28e-5 | 8.43e-5 | 0.000404 |
| 100000 | 10 | 20000 | 0.2 | no | 7 | 47.2 | 49.6 | 0.952 | 3.01e-5 | 3.4e-5 | 0.000169 |

The rule sorts the cells correctly up to two exceptions. Where it fires
(seven cells) Compress++ is 1.3-6.9x faster in six and 2.8% slower in the
seventh (N = 10⁴, p = 384, 1%); where it does not (five cells) plain is
1.05-1.5x faster in four and Compress++ is 1.14x faster in the fifth
(N = 10⁴, p = 10, 10%). The dimension moves the ratio against Compress++
(1.36x at p = 10 against 0.98x at p = 384 for N = 10⁴ and 1%; 1.92x against
1.31x at 5%). The speedup grows with N, and at N = 10⁵ it shrinks with n/N
from 6.9x at 1% to 1.4x at 10%; at N = 10⁴ it peaks at 5%. The price is
quality: where the rule fires, Compress++'s energy distance is 3-7% above
plain kernel thinning at N = 10⁴ and 8-16% above it at N = 10⁵, worst at
5-10% where `g` = 5-6 leaves the compressed set only about 2n rows for the
final thinning (`2^g √N ≈ 2.02n` there, against `3.2n` at N = 10⁴); both
stay at least 1.6x below random in every cell, and plain kernel thinning
up to 5.6x below it at N = 10⁵. `_compress_pays_off` therefore stays as
it is and `:auto` remains the default; pass `compress = :never` when the
last 8-16% of quality matter more than the run time at N ≥ 10⁵. Reproduce
with:

```sh
julia -t auto --project=benchmark benchmark/compress.jl
```

## [Gaussian update rule](@id gaussian-update)

`support_points(::GaussianKernel, …)` keeps the projected-gradient optimizer
with Armijo backtracking on full data, and runs the majorization–minimization
(MM) sweep (mean-shift data term, majorized repulsion; see
[Methods](@ref methods)) only in stochastic mode, i.e. when `kappa` is below
the number of target rows. Measured on the four benchmark datasets at
N = 1,000 and 10,000, n = 0.2N, `:median` bandwidth, three seeds: an MM
sweep costs 1.9-6.9x less than an Armijo iteration, but it takes smaller
steps — every MM run in the table uses its full iteration cap (200 at
N = 1,000, 100 at N = 10,000). The benchmark script's `mm` arm is a private
loop that always runs for a fixed number of iterations, so its iteration
column is the cap by construction, not evidence of non-convergence on its
own; a separate, earlier run of the same comparison through
`support_points` (made before the optimizer decision below) did reach the
same outcome by actually checking the rule — it never satisfied the
displacement rule within the cap either. Armijo stops early (by its
relative-decrease rule in that earlier run through `support_points`; the
table itself records only the iteration counts) on three of the four
datasets (mixture-2d, normal-10d, t3-3d). On `uniform-5d`, where Armijo also
runs to its cap, the extra MM iterations do not pay off: MM's selected-row
MMD is worse than Armijo's at both sizes (N = 1,000: 0.00267 vs 0.00116,
random 0.0036; N = 10,000: 0.000393 vs 0.000305, random 0.000401). On the
other three datasets MM's selected-row MMD stays within 8% of Armijo's
(mixture-2d at N = 1,000, both `normal-10d` cells, `t3-3d` at N = 10,000) or
up to 53% lower (mixture-2d at N = 10,000); the one exception is `t3-3d` at
N = 1,000, 32% higher. `kappa = 1,000` cuts the N = 10,000 MM time by
3.4-3.8x (0.84-1.56s against 2.93-5.88s) at MMD within 6% of the full-data
sweep on three datasets, and about 21x higher — but still about 2.2x below
a random subset — on `mixture-2d`. An over-relaxed sweep (adaptive
extrapolation along the MM direction, safeguarded by one objective
evaluation per iteration) was also tried during the design and rejected:
the objective barely improved while every iteration gained an objective
evaluation, which costs as much as the sweep itself; the design record
(`docs/superpowers/specs/2026-09-04-gaussian-mm-update-design.md`) has the
numbers. The damped uniform-weight fixed point of Belhadji, Sharp & Marzouk
(2025, eq. 29) diverges on every dataset because its denominator crosses
zero where the point set fits the data. Full table:
[`assets/benchmarks/gaussian_update.md`](assets/benchmarks/gaussian_update.md).
Reproduce with:

```sh
julia -t auto --project=benchmark benchmark/gaussian_update.jl
```
