# [Benchmarks](@id benchmarks)

## Setup

Four seeded datasets, each split with `ratio = 0.2`:

- `mixture-2d`: a 2-D Gaussian mixture with 4 components.
- `normal-10d`: 10-D standard normal.
- `uniform-5d`: 5-D uniform on ``[0, 1]^5``.
- `t3-3d`: 3-D Student-``t`` with 3 degrees of freedom (heavy-tailed).

Each dataset is generated at ``N \in \{1{,}000, 10{,}000\}``. Five methods
are compared per (dataset, ``N``): `SupportPointSplitter(EnergyKernel())`,
`SupportPointSplitter(GaussianKernel())`, `HerdingSplitter(EnergyKernel())`,
`HerdingSplitter(GaussianKernel())`, and a random split (mean of 5 seeds).
At ``N = 10{,}000`` the support-point methods use `kappa = 1_000`
(energy kernel) and a lower `max_iterations = 100` (Gaussian kernel, which
has no stochastic mode) to keep wall time reasonable; the herding methods
use `kappa = 2_000` at that size. On `normal-10d` and `uniform-5d` at that
size, `support points · gaussian` converges after a single iteration
(`result.converged == true`, ~1 s) rather than running the full
100-iteration cap that `mixture-2d` and `t3-3d` use there (not converged,
~40-45 s) — see "Reading the results" for why.

For every (dataset, method) pair the script records the energy distance and
the Gaussian-kernel MMD (median-heuristic bandwidth, resolved once per
dataset) between the resulting train and test rows, plus wall-clock time.
Smaller is better for both discrepancy metrics.

Exact command:

```sh
julia -t auto --project=benchmark benchmark/run.jl
```

Recorded when the run below was produced:

- Julia: 1.10.12
- Threads: 16 (`-t auto`)
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor

## Figures

![Split quality by method](assets/benchmarks/quality.png)

Grouped bars of energy distance and Gaussian MMD per method, one panel per
dataset and size, on a log scale — shorter bars are better splits.

![Split time by method](assets/benchmarks/time.png)

Wall time versus ``N`` for each method (log–log); the random split is
excluded since it does no optimization.

![Test-row selection on the 2-D mixture](assets/benchmarks/selection.png)

The 2-D Gaussian-mixture data with the test rows each method selects
overlaid, showing why the methods disagree on which rows to hold out.

## Results

| dataset | N | method | energy distance | MMD (Gaussian, median σ) | seconds |
|---|---:|---|---:|---:|---:|
| mixture-2d | 1000 | support points · energy | 0.000333 | 1.1e-6 | 0.46 |
| mixture-2d | 1000 | support points · gaussian | 0.00258 | 1.64e-5 | 1.1 |
| mixture-2d | 1000 | herding · energy | 0.000439 | 1.08e-5 | 0.0018 |
| mixture-2d | 1000 | herding · gaussian | 0.00772 | 1.9e-5 | 0.021 |
| mixture-2d | 1000 | random | 0.012 | 0.00258 | – |
| normal-10d | 1000 | support points · energy | 0.0229 | 0.00203 | 0.52 |
| normal-10d | 1000 | support points · gaussian | 0.0309 | 0.00302 | 1.2 |
| normal-10d | 1000 | herding · energy | 0.00822 | 0.000114 | 0.0029 |
| normal-10d | 1000 | herding · gaussian | 0.00927 | 9.82e-5 | 0.015 |
| normal-10d | 1000 | random | 0.0255 | 0.00227 | – |
| uniform-5d | 1000 | support points · energy | 0.00392 | 9.87e-5 | 0.54 |
| uniform-5d | 1000 | support points · gaussian | 0.0149 | 0.00176 | 1.2 |
| uniform-5d | 1000 | herding · energy | 0.00343 | 3.62e-5 | 0.002 |
| uniform-5d | 1000 | herding · gaussian | 0.00535 | 3.54e-5 | 0.012 |
| uniform-5d | 1000 | random | 0.0173 | 0.00217 | – |
| t3-3d | 1000 | support points · energy | 0.00156 | 8.0e-5 | 0.33 |
| t3-3d | 1000 | support points · gaussian | 0.00669 | 0.000993 | 1.1 |
| t3-3d | 1000 | herding · energy | 0.00174 | 9.8e-5 | 0.0017 |
| t3-3d | 1000 | herding · gaussian | 0.0035 | 7.58e-5 | 0.017 |
| t3-3d | 1000 | random | 0.0161 | 0.00375 | – |
| mixture-2d | 10000 | support points · energy | 0.00102 | 0.000119 | 3.4 |
| mixture-2d | 10000 | support points · gaussian | 0.00135 | 0.000184 | 41.0 |
| mixture-2d | 10000 | herding · energy | 0.00159 | 0.000393 | 0.14 |
| mixture-2d | 10000 | herding · gaussian | 0.00635 | 0.000297 | 0.38 |
| mixture-2d | 10000 | random | 0.00151 | 0.000286 | – |
| normal-10d | 10000 | support points · energy | 0.00361 | 0.000315 | 8.8 |
| normal-10d | 10000 | support points · gaussian | 0.00455 | 0.000457 | 0.49 |
| normal-10d | 10000 | herding · energy | 0.0045 | 0.000371 | 0.16 |
| normal-10d | 10000 | herding · gaussian | 0.0046 | 0.000431 | 0.38 |
| normal-10d | 10000 | random | 0.00355 | 0.000302 | – |
| uniform-5d | 10000 | support points · energy | 0.00265 | 0.000373 | 5.6 |
| uniform-5d | 10000 | support points · gaussian | 0.00303 | 0.00039 | 0.46 |
| uniform-5d | 10000 | herding · energy | 0.0026 | 0.000281 | 0.12 |
| uniform-5d | 10000 | herding · gaussian | 0.00338 | 0.00049 | 0.37 |
| uniform-5d | 10000 | random | 0.00252 | 0.000327 | – |
| t3-3d | 10000 | support points · energy | 0.00106 | 0.000246 | 4.2 |
| t3-3d | 10000 | support points · gaussian | 0.00165 | 0.000259 | 45.0 |
| t3-3d | 10000 | herding · energy | 0.00154 | 0.000426 | 0.091 |
| t3-3d | 10000 | herding · gaussian | 0.0024 | 0.000397 | 0.32 |
| t3-3d | 10000 | random | 0.00213 | 0.00044 | – |

## Reading the results

`support points · energy` has the lowest or near-lowest energy distance in
most rows, though at ``N = 1{,}000`` `herding · energy` is lower on
`normal-10d` (0.00822 vs. 0.0229) and `uniform-5d` (0.00343 vs. 0.00392).
`herding · gaussian` matches or beats `support points · energy` on MMD at
``N = 1{,}000`` on `normal-10d` and `uniform-5d`, at a small fraction of the
cost since herding has no iterative optimizer to run; at ``N = 10{,}000`` it
falls behind on both (0.000431 vs. 0.000315 on `normal-10d`; 0.00049 vs.
0.000373 on `uniform-5d`). `support points · gaussian` is the slowest
method on `mixture-2d` and `t3-3d`, where it runs the full 100-iteration
cap without converging (41-45 s at ``N = 10{,}000``); on `normal-10d` and
`uniform-5d` it instead converges after one iteration, because with the
median-heuristic bandwidth in 5-10 dimensions the initial random sample is
already near-stationary for the MMD objective, so the optimizer stops right
away — those two cells reflect an early stop (essentially a random sample),
not a capped optimization, and their 0.46-0.49 s timings are not evidence
of fast optimization. At ``N = 10{,}000``, energy distance favors the
optimized methods on `mixture-2d` and `t3-3d`, while on `normal-10d` and
`uniform-5d` every method is within noise of — or slightly behind — the
random baseline on both metrics (the subsampled `splitquality` estimate
above the 4,000-row exact threshold adds its own noise there). Recommendation:
use `support points · energy` as the default; reach for `herding · gaussian`
when Gaussian-kernel MMD is the target metric or a deterministic,
optimizer-free split is wanted.
