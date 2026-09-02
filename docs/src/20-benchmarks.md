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
(`result.converged == true`, 0.49-0.58 s) rather than running the full
100-iteration cap that `mixture-2d` and `t3-3d` use there (not converged,
43-44 s): the absolute displacement tolerance fires at the initial sample
because the 1/n-scaled gradient is below it; a scale-aware tolerance is a
planned follow-up — see "Reading the results" for the full explanation.

For every (dataset, method) pair the script records the energy distance and
the Gaussian-kernel MMD (median-heuristic bandwidth, resolved once per
dataset) between the resulting train and test rows, plus wall-clock time.
Every score is computed exactly (`splitquality(...; exact_threshold =
typemax(Int))`), never via the subsampled estimator, so the table below
carries no subsampling noise even at ``N = 10{,}000``. Each splitter's
JIT warm-up runs on a throwaway copy seeded with `MersenneTwister(0)`
before the timed run on a `MersenneTwister(1)`-seeded copy, so warm-up
compilation never consumes the timed splitter's own random draws.
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
| mixture-2d | 1000 | support points · energy | 0.000323 | 1.57e-6 | 0.33 |
| mixture-2d | 1000 | support points · gaussian | 0.00235 | 9.1e-6 | 1.1 |
| mixture-2d | 1000 | herding · energy | 0.000439 | 1.08e-5 | 0.0021 |
| mixture-2d | 1000 | herding · gaussian | 0.00772 | 1.9e-5 | 0.018 |
| mixture-2d | 1000 | random | 0.012 | 0.00258 | – |
| normal-10d | 1000 | support points · energy | 0.0215 | 0.00187 | 0.76 |
| normal-10d | 1000 | support points · gaussian | 0.0247 | 0.00224 | 1.3 |
| normal-10d | 1000 | herding · energy | 0.00822 | 0.000114 | 0.0027 |
| normal-10d | 1000 | herding · gaussian | 0.00927 | 9.82e-5 | 0.031 |
| normal-10d | 1000 | random | 0.0255 | 0.00227 | – |
| uniform-5d | 1000 | support points · energy | 0.0047 | 0.000255 | 0.56 |
| uniform-5d | 1000 | support points · gaussian | 0.015 | 0.00176 | 1.1 |
| uniform-5d | 1000 | herding · energy | 0.00343 | 3.62e-5 | 0.012 |
| uniform-5d | 1000 | herding · gaussian | 0.00535 | 3.54e-5 | 0.016 |
| uniform-5d | 1000 | random | 0.0173 | 0.00217 | – |
| t3-3d | 1000 | support points · energy | 0.00163 | 0.000116 | 0.43 |
| t3-3d | 1000 | support points · gaussian | 0.0052 | 0.000663 | 1.1 |
| t3-3d | 1000 | herding · energy | 0.00174 | 9.8e-5 | 0.0018 |
| t3-3d | 1000 | herding · gaussian | 0.0035 | 7.58e-5 | 0.014 |
| t3-3d | 1000 | random | 0.0161 | 0.00375 | – |
| mixture-2d | 10000 | support points · energy | 0.000173 | 3.2e-5 | 3.6 |
| mixture-2d | 10000 | support points · gaussian | 0.0003 | 7.76e-7 | 43.0 |
| mixture-2d | 10000 | herding · energy | 0.00121 | 0.000259 | 0.18 |
| mixture-2d | 10000 | herding · gaussian | 0.00466 | 0.000163 | 0.4 |
| mixture-2d | 10000 | random | 0.000885 | 0.000166 | – |
| normal-10d | 10000 | support points · energy | 0.0025 | 0.000215 | 9.2 |
| normal-10d | 10000 | support points · gaussian | 0.00299 | 0.000289 | 0.49 |
| normal-10d | 10000 | herding · energy | 0.00238 | 0.000216 | 0.15 |
| normal-10d | 10000 | herding · gaussian | 0.00313 | 0.000289 | 0.45 |
| normal-10d | 10000 | random | 0.00208 | 0.000164 | – |
| uniform-5d | 10000 | support points · energy | 0.000844 | 7.24e-5 | 6.0 |
| uniform-5d | 10000 | support points · gaussian | 0.00187 | 0.000244 | 0.58 |
| uniform-5d | 10000 | herding · energy | 0.00137 | 0.000147 | 0.14 |
| uniform-5d | 10000 | herding · gaussian | 0.00188 | 0.000246 | 0.36 |
| uniform-5d | 10000 | random | 0.00146 | 0.00016 | – |
| t3-3d | 10000 | support points · energy | 0.000262 | 4.82e-5 | 4.8 |
| t3-3d | 10000 | support points · gaussian | 0.000567 | 7.64e-5 | 44.0 |
| t3-3d | 10000 | herding · energy | 0.000886 | 0.000155 | 0.11 |
| t3-3d | 10000 | herding · gaussian | 0.00153 | 0.000217 | 0.35 |
| t3-3d | 10000 | random | 0.00151 | 0.000305 | – |

## Reading the results

At ``N = 1{,}000`` every method runs on the full data (no `kappa`), and
herding is competitive with or ahead of support points: `herding · energy`
has the lowest energy distance on `normal-10d` (0.00822 vs. 0.0215) and
`uniform-5d` (0.00343 vs. 0.0047), and `herding · gaussian` has the lowest
MMD on `normal-10d`, `uniform-5d`, and `t3-3d` (e.g. 9.82e-5 vs. 0.00187 on
`normal-10d`) at a small fraction of the optimizer's wall time.
`support points · energy` wins both metrics on `mixture-2d` and the energy
distance on `t3-3d`.

At ``N = 10{,}000``, where herding switches to `kappa = 2{,}000` (a 20%
row subsample drawn once for the whole run) and `support points · energy`
to `kappa = 1{,}000`, herding's advantage mostly disappears:
`support points · energy` has the lowest energy distance on `mixture-2d`
(0.000173), `uniform-5d` (0.000844), and `t3-3d` (0.000262), and the lowest
MMD on `uniform-5d` and `t3-3d`; `support points · gaussian` has the lowest
MMD on `mixture-2d` (7.76e-7). On `normal-10d` at ``N = 10{,}000`` every
optimized method is slightly behind the random baseline on both metrics
(random: 0.00208 energy distance, 0.000164 MMD) — with 10-D standard normal
data and this coarse a reference sample, none of the methods has enough
signal to reliably beat chance, and the gaps between methods there are
small enough not to be decisive.

`support points · gaussian` is the slowest method on `mixture-2d` and
`t3-3d`, where it runs the full 100-iteration cap without converging
(43-44 s at ``N = 10{,}000``); on `normal-10d` and `uniform-5d` it instead
converges after a single iteration (0.49-0.58 s). This is not the sample
being near-stationary: the objective's ``1/n^2`` and ``1/(nN)`` scaling
factors make the initial gradient row-norms of order ``10^{-6}``, so the
first squared displacement (``\sim 10^{-11}``) is already below
`tolerance = 1e-10` — the absolute displacement tolerance fires at the
initial sample because the 1/n-scaled gradient is below it, even though
further iterations do decrease the objective; a scale-aware tolerance is a
planned follow-up. Those two cells reflect this early stop, not a capped
optimization, and their fast timings are not evidence of fast convergence.

Recommendation: use `support points · energy` as the default at scale
(``N \gtrsim 10{,}000`` with `kappa`); at smaller ``N``, where herding runs
on the full data without `kappa`, `herding · gaussian` is a strong,
deterministic, and far cheaper alternative when Gaussian-kernel MMD is the
target metric.
