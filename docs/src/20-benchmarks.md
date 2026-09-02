# [Benchmarks](@id benchmarks)

**Use `herding · energy` by default.** It has the lowest energy distance in
6 of the 8 (dataset, N) cases and the lowest Gaussian MMD in 3 of 4 at
``N = 10{,}000``, and it is the fastest optimized method at every size. At
``N = 1{,}000``, `support points · energy` wins `mixture-2d` on both
metrics and the `t3-3d` energy distance, and `herding · gaussian` has the
lowest MMD on the other three datasets. Reproduce with the command under
[Environment](@ref benchmarks-environment).

## Summary

| dataset | N | lowest energy distance | lowest MMD | fastest |
|---|---:|---|---|---|
| mixture-2d | 1000 | support points · energy | support points · energy | herding · energy |
| normal-10d | 1000 | herding · energy | herding · gaussian | herding · energy |
| uniform-5d | 1000 | herding · energy | herding · gaussian | herding · energy |
| t3-3d | 1000 | support points · energy | herding · gaussian | herding · energy |
| mixture-2d | 10000 | herding · energy | herding · energy | herding · energy |
| normal-10d | 10000 | herding · energy | herding · energy | herding · energy |
| uniform-5d | 10000 | herding · energy | herding · energy | herding · energy |
| t3-3d | 10000 | herding · energy | herding · gaussian | herding · energy |

Smaller discrepancy is better; times exclude JIT warm-up. Full per-cell
numbers are in the tables under [Results](@ref benchmarks-results).

## Setup

### Datasets

| dataset | distribution | dimensions |
|---|---|---:|
| mixture-2d | Gaussian mixture, 4 components | 2 |
| normal-10d | standard normal | 10 |
| uniform-5d | uniform on ``[0, 1]^5`` | 5 |
| t3-3d | Student-``t``, 3 degrees of freedom (heavy-tailed) | 3 |

All four datasets are seeded, and each is split with `ratio = 0.2` at
``N \in \{1{,}000, 10{,}000\}``.

### Methods

| method | splitter | N = 1,000 | N = 10,000 |
|---|---|---|---|
| support points · energy | `SupportPointSplitter(EnergyKernel())` | `kappa = nothing` (full data) | `kappa = 1_000` |
| support points · gaussian | `SupportPointSplitter(GaussianKernel())` | `max_iterations = 200` | `max_iterations = 100`; no stochastic mode |
| herding · energy | `HerdingSplitter(EnergyKernel())` | exact data term, no `kappa` | exact data term, no `kappa` |
| herding · gaussian | `HerdingSplitter(GaussianKernel())` | exact data term, no `kappa` | exact data term, no `kappa` |
| random | uniform random split | mean of 5 seeds | mean of 5 seeds |

### Protocol

- Measures energy distance and Gaussian-kernel MMD (median-heuristic
  bandwidth, resolved once per dataset) between the resulting train and
  test rows, plus wall-clock time.
- Every score is computed exactly via `splitquality(...; exact_threshold =
  typemax(Int))`, never the subsampled estimator, so there is no
  subsampling noise even at ``N = 10{,}000``.
- Each splitter's JIT warm-up runs on a throwaway copy seeded with
  `MersenneTwister(0)`, then the timed run uses a separate
  `MersenneTwister(1)`-seeded copy, so warm-up compilation never consumes
  the timed splitter's own random draws.
- The random split is the mean of 5 seeds.

### [Environment](@id benchmarks-environment)

Exact command:

```sh
julia -t auto --project=benchmark benchmark/run.jl
```

Recorded when the run below was produced:

- Julia: 1.10.12
- Threads: 16 (`-t auto`)
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor

## [Results](@id benchmarks-results)

Best value per row in **bold**. The raw per-run table is
`assets/benchmarks/results.md`.

### Energy distance

| dataset | N | support points · energy | support points · gaussian | herding · energy | herding · gaussian | random |
|---|---:|---:|---:|---:|---:|---:|
| mixture-2d | 1000 | **0.000323** | 0.00235 | 0.000439 | 0.00772 | 0.012 |
| normal-10d | 1000 | 0.0215 | 0.0247 | **0.00822** | 0.00927 | 0.0255 |
| uniform-5d | 1000 | 0.0047 | 0.015 | **0.00343** | 0.00535 | 0.0173 |
| t3-3d | 1000 | **0.00163** | 0.0052 | 0.00174 | 0.0035 | 0.0161 |
| mixture-2d | 10000 | 0.000173 | 0.0003 | **1.42e-5** | 0.00432 | 0.000885 |
| normal-10d | 10000 | 0.0025 | 0.00299 | **0.0006** | 0.000873 | 0.00208 |
| uniform-5d | 10000 | 0.000844 | 0.00187 | **0.000199** | 0.00046 | 0.00146 |
| t3-3d | 10000 | 0.000262 | 0.000567 | **8.06e-5** | 0.00082 | 0.00151 |

### Gaussian MMD

Bandwidth ``\sigma`` is the median-heuristic value, resolved once per
dataset.

| dataset | N | support points · energy | support points · gaussian | herding · energy | herding · gaussian | random |
|---|---:|---:|---:|---:|---:|---:|
| mixture-2d | 1000 | **1.57e-6** | 9.1e-6 | 1.08e-5 | 1.9e-5 | 0.00258 |
| normal-10d | 1000 | 0.00187 | 0.00224 | 0.000114 | **9.82e-5** | 0.00227 |
| uniform-5d | 1000 | 0.000255 | 0.00176 | 3.62e-5 | **3.54e-5** | 0.00217 |
| t3-3d | 1000 | 0.000116 | 0.000663 | 9.8e-5 | **7.58e-5** | 0.00375 |
| mixture-2d | 10000 | 3.2e-5 | 7.76e-7 | **1.87e-7** | 2.85e-7 | 0.000166 |
| normal-10d | 10000 | 0.000215 | 0.000289 | **1.91e-6** | 2.17e-6 | 0.000164 |
| uniform-5d | 10000 | 7.24e-5 | 0.000244 | **3.2e-7** | 4.64e-7 | 0.00016 |
| t3-3d | 10000 | 4.82e-5 | 7.64e-5 | 2.77e-6 | **2.25e-6** | 0.000305 |

### Wall time (seconds)

The random split does no optimization, so it is omitted here.

| dataset | N | support points · energy | support points · gaussian | herding · energy | herding · gaussian |
|---|---:|---:|---:|---:|---:|
| mixture-2d | 1000 | 0.26 | 1.0 | **0.0015** | 0.013 |
| normal-10d | 1000 | 0.71 | 1.4 | **0.0028** | 0.016 |
| uniform-5d | 1000 | 0.6 | 1.1 | **0.0022** | 0.016 |
| t3-3d | 1000 | 0.44 | 1.1 | **0.003** | 0.018 |
| mixture-2d | 10000 | 3.6 | 42.0 | **0.13** | 0.43 |
| normal-10d | 10000 | 9.0 | 0.49 | **0.21** | 0.52 |
| uniform-5d | 10000 | 6.2 | 0.48 | **0.16** | 0.5 |
| t3-3d | 10000 | 4.4 | 42.0 | **0.13** | 0.48 |

## Figures

![Split quality by method](assets/benchmarks/quality.png)

Grouped bars of energy distance and Gaussian MMD per method, one panel per
dataset and size, on a log scale; shorter bars are better splits.

![Split time by method](assets/benchmarks/time.png)

Wall time versus ``N`` for each method, log–log; the random split is
excluded since it does no optimization.

![Test-row selection on the 2-D mixture](assets/benchmarks/selection.png)

The 2-D Gaussian-mixture data with the test rows each method selects
overlaid, showing why the methods disagree on which rows to hold out.

## Key findings

- **At N = 1,000, herding is competitive or ahead.** `herding · energy` has
  the lowest energy distance on `normal-10d` and `uniform-5d`, and
  `herding · gaussian` has the lowest MMD on `normal-10d`, `uniform-5d`, and
  `t3-3d`, all at a small fraction of the optimizer's wall time.
- **At N = 1,000, `support points · energy` still wins on two datasets.** It
  takes both metrics on `mixture-2d` and the energy distance on `t3-3d`.
- **At N = 10,000, `herding · energy` has the lowest energy distance on
  every dataset**, 3.3x-12.2x below `support points · energy`. Part of that
  gap is the `kappa = 1_000` stochastic approximation on the support-point
  side; on `normal-10d` and `uniform-5d` the runner-up is
  `herding · gaussian`, not `support points · energy`.
- **At N = 10,000, the two herding methods take the lowest MMD on every
  dataset**, well below both support-point methods and random (e.g.
  `normal-10d`: 1.91e-6 for `herding · energy` versus 0.000215 for
  `support points · energy`).
- **Herding is also the fastest optimized method at N = 10,000.**
  `herding · energy` takes 0.13-0.21 s and `herding · gaussian` 0.43-0.52 s,
  versus 3.6-9.0 s for `support points · energy`. `support points ·
  gaussian` is bimodal, 0.48-0.49 s or 42 s; see Caveats.
- **Herding uses the exact data term at every size**, so its selections
  realize the greedy rule's guarantee at ``N = 10{,}000`` as well as at
  ``N = 1{,}000``.

## Caveats

`support points · gaussian` timings are bimodal. On `mixture-2d` and
`t3-3d` it runs the full 100-iteration cap without converging (42 s). On
`normal-10d` and `uniform-5d` it stops after one iteration (0.48-0.49 s,
`result.converged == true`). This is not near-stationarity: the
objective's ``1/n^2`` and ``1/(nN)`` scaling makes the initial gradient
row-norms of order ``10^{-6}``, so the first squared displacement
(``\sim 10^{-11}``) is already below `tolerance = 1e-10`; the absolute
displacement tolerance fires at the initial sample even though further
iterations would still decrease the objective. A scale-aware tolerance is
a planned follow-up. Those two fast cells are an early stop, not fast
convergence.

`support points · energy` runs with `kappa = 1_000` at ``N = 10{,}000``,
so its ``N = 10{,}000`` numbers include stochastic-MM approximation error;
at ``N = 1{,}000`` both it and herding are exact.
