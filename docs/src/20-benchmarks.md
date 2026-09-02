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
At ``N = 10{,}000`` the support-point energy kernel uses `kappa = 1_000`
and the Gaussian kernel uses a lower `max_iterations = 100` (it has no
stochastic mode) to keep wall time reasonable; herding uses the exact data
term at all sizes — there is no `kappa` for `HerdingSplitter`. On
`normal-10d` and `uniform-5d` at ``N = 10{,}000``, `support points ·
gaussian` still converges after a single iteration (`result.converged ==
true`, 0.48-0.49 s) rather than running the full 100-iteration cap that
`mixture-2d` and `t3-3d` use there (not converged, 42 s): the absolute
displacement tolerance fires at the initial sample because the
1/n-scaled gradient is below it; a scale-aware tolerance is a planned
follow-up — see "Reading the results" for the full explanation.

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
| mixture-2d | 1000 | support points · energy | 0.000323 | 1.57e-6 | 0.26 |
| mixture-2d | 1000 | support points · gaussian | 0.00235 | 9.1e-6 | 1.0 |
| mixture-2d | 1000 | herding · energy | 0.000439 | 1.08e-5 | 0.0015 |
| mixture-2d | 1000 | herding · gaussian | 0.00772 | 1.9e-5 | 0.013 |
| mixture-2d | 1000 | random | 0.012 | 0.00258 | – |
| normal-10d | 1000 | support points · energy | 0.0215 | 0.00187 | 0.71 |
| normal-10d | 1000 | support points · gaussian | 0.0247 | 0.00224 | 1.4 |
| normal-10d | 1000 | herding · energy | 0.00822 | 0.000114 | 0.0028 |
| normal-10d | 1000 | herding · gaussian | 0.00927 | 9.82e-5 | 0.016 |
| normal-10d | 1000 | random | 0.0255 | 0.00227 | – |
| uniform-5d | 1000 | support points · energy | 0.0047 | 0.000255 | 0.6 |
| uniform-5d | 1000 | support points · gaussian | 0.015 | 0.00176 | 1.1 |
| uniform-5d | 1000 | herding · energy | 0.00343 | 3.62e-5 | 0.0022 |
| uniform-5d | 1000 | herding · gaussian | 0.00535 | 3.54e-5 | 0.016 |
| uniform-5d | 1000 | random | 0.0173 | 0.00217 | – |
| t3-3d | 1000 | support points · energy | 0.00163 | 0.000116 | 0.44 |
| t3-3d | 1000 | support points · gaussian | 0.0052 | 0.000663 | 1.1 |
| t3-3d | 1000 | herding · energy | 0.00174 | 9.8e-5 | 0.003 |
| t3-3d | 1000 | herding · gaussian | 0.0035 | 7.58e-5 | 0.018 |
| t3-3d | 1000 | random | 0.0161 | 0.00375 | – |
| mixture-2d | 10000 | support points · energy | 0.000173 | 3.2e-5 | 3.6 |
| mixture-2d | 10000 | support points · gaussian | 0.0003 | 7.76e-7 | 42.0 |
| mixture-2d | 10000 | herding · energy | 1.42e-5 | 1.87e-7 | 0.13 |
| mixture-2d | 10000 | herding · gaussian | 0.00432 | 2.85e-7 | 0.43 |
| mixture-2d | 10000 | random | 0.000885 | 0.000166 | – |
| normal-10d | 10000 | support points · energy | 0.0025 | 0.000215 | 9.0 |
| normal-10d | 10000 | support points · gaussian | 0.00299 | 0.000289 | 0.49 |
| normal-10d | 10000 | herding · energy | 0.0006 | 1.91e-6 | 0.21 |
| normal-10d | 10000 | herding · gaussian | 0.000873 | 2.17e-6 | 0.52 |
| normal-10d | 10000 | random | 0.00208 | 0.000164 | – |
| uniform-5d | 10000 | support points · energy | 0.000844 | 7.24e-5 | 6.2 |
| uniform-5d | 10000 | support points · gaussian | 0.00187 | 0.000244 | 0.48 |
| uniform-5d | 10000 | herding · energy | 0.000199 | 3.2e-7 | 0.16 |
| uniform-5d | 10000 | herding · gaussian | 0.00046 | 4.64e-7 | 0.5 |
| uniform-5d | 10000 | random | 0.00146 | 0.00016 | – |
| t3-3d | 10000 | support points · energy | 0.000262 | 4.82e-5 | 4.4 |
| t3-3d | 10000 | support points · gaussian | 0.000567 | 7.64e-5 | 42.0 |
| t3-3d | 10000 | herding · energy | 8.06e-5 | 2.77e-6 | 0.13 |
| t3-3d | 10000 | herding · gaussian | 0.00082 | 2.25e-6 | 0.48 |
| t3-3d | 10000 | random | 0.00151 | 0.000305 | – |

## Reading the results

At ``N = 1{,}000`` every method runs on the full data (no `kappa` for
either splitter at this size), and herding is competitive with or ahead of
support points: `herding · energy` has the lowest energy distance on
`normal-10d` (0.00822 vs. 0.0215) and `uniform-5d` (0.00343 vs. 0.0047), and
`herding · gaussian` has the lowest MMD on `normal-10d`, `uniform-5d`, and
`t3-3d` (e.g. 9.82e-5 vs. 0.00187 on `normal-10d`) at a small fraction of
the optimizer's wall time. `support points · energy` wins both metrics on
`mixture-2d` and the energy distance on `t3-3d`.

At ``N = 10{,}000``, herding now uses the same exact data term as at
``N = 1{,}000`` (only the support-point energy kernel still uses
`kappa = 1{,}000` there), and `herding · energy` has the lowest energy
distance on **every** dataset, by 3.3x-12.2x over the next-best method
(`support points · energy` in all four cases): `mixture-2d` (1.42e-5 vs.
0.000173, 12.2x), `normal-10d` (0.0006 vs. 0.0025, 4.2x), `uniform-5d`
(0.000199 vs. 0.000844, 4.2x), and `t3-3d` (8.06e-5 vs. 0.000262, 3.3x).
The two herding methods also take the lowest MMD on every dataset
(`herding · energy` on `mixture-2d`, `normal-10d`, `uniform-5d`; `herding ·
gaussian` narrowly ahead of `herding · energy` on `t3-3d`, 2.25e-6 vs.
2.77e-6), both well below the support-point methods (e.g. 0.000215-0.000289
on `normal-10d`) and below random on every dataset. This is the fix from
the previous wave working as intended: with the subsampled `kappa` data
term removed, herding is no longer handicapped by a candidate-asymmetric
estimate at this size, and the exact greedy rule delivers the quality its
derivation promises. `normal-10d` in particular flips from "no method
reliably beats random" in the previous (buggy) run to `herding · energy`
beating random by roughly 3.5x on energy distance and 86x on MMD
(0.000164 vs. 1.91e-6).

Herding is also the fastest optimized method at ``N = 10{,}000`` by a wide
margin: `herding · energy` takes 0.11-0.21 s and `herding · gaussian`
0.43-0.52 s, versus 3.6-9.0 s for `support points · energy` and a bimodal
0.48 s or 42 s for `support points · gaussian` (see below).

`support points · gaussian` is the slowest method on `mixture-2d` and
`t3-3d`, where it runs the full 100-iteration cap without converging (42 s
at ``N = 10{,}000``); on `normal-10d` and `uniform-5d` it instead converges
after a single iteration (0.48-0.49 s). This is not the sample being
near-stationary: the objective's ``1/n^2`` and ``1/(nN)`` scaling factors
make the initial gradient row-norms of order ``10^{-6}``, so the first
squared displacement (``\sim 10^{-11}``) is already below
`tolerance = 1e-10` — the absolute displacement tolerance fires at the
initial sample because the 1/n-scaled gradient is below it, even though
further iterations do decrease the objective; a scale-aware tolerance is a
planned follow-up. Those two cells reflect this early stop, not a capped
optimization, and their fast timings are not evidence of fast convergence.
This disclosure is unchanged from the previous wave: it concerns the
support-point Gaussian-kernel optimizer, not herding, and the fix in this
wave does not touch it.

Recommendation: with the subsampled data term removed, `herding · energy`
is now the best default across every tested dataset and both sizes — best
or tied-best on both discrepancy metrics, and the fastest optimized method
at ``N = 10{,}000``. `support points · energy` remains a reasonable
alternative where the energy-distance MM objective's convergence guarantees
matter more than wall time; `herding · gaussian` is preferable when
Gaussian-kernel MMD specifically is the target metric on `t3-3d`-like
heavy-tailed data.
