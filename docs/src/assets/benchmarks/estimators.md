# Estimator selection experiment

Absolute error against the exact value and wall time of every candidate
[`DiscrepancyEstimator`](@ref) for `energydistance`/`mmd`, measured on the
splits from `support points · energy`, `herding · energy` (scored by
energy distance) and `herding · gaussian` (scored by MMD, median
bandwidth resolved once per dataset), on the four Phase 2b datasets at
N = 10,000, over 5 rng seeds per (dataset, split, estimator) cell.

## Per-row results

| dataset | split | kernel | estimator | mean abs error | max abs error | mean time (s) |
|---|---|---|---|---:|---:|---:|
| mixture-2d | support points · energy | EnergyKernel | Subsample(2000, 8) | 0.000616 | 0.000759 | 0.408 |
| mixture-2d | support points · energy | EnergyKernel | RandomSlices(64) | 9.18e-6 | 1.47e-5 | 0.0244 |
| mixture-2d | support points · energy | EnergyKernel | RandomSlices(256) | 6.85e-6 | 1.66e-5 | 0.0861 |
| mixture-2d | support points · energy | EnergyKernel | RandomSlices(1024) | 2.12e-6 | 5.09e-6 | 0.401 |
| mixture-2d | herding · energy | EnergyKernel | Subsample(2000, 8) | 0.000663 | 0.000743 | 0.305 |
| mixture-2d | herding · energy | EnergyKernel | RandomSlices(64) | 1.1e-7 | 1.89e-7 | 0.0199 |
| mixture-2d | herding · energy | EnergyKernel | RandomSlices(256) | 7.56e-8 | 1.99e-7 | 0.109 |
| mixture-2d | herding · energy | EnergyKernel | RandomSlices(1024) | 2.89e-8 | 4.27e-8 | 0.355 |
| mixture-2d | herding · gaussian | GaussianKernel | Subsample(2000, 8) | 0.000117 | 0.000161 | 0.337 |
| mixture-2d | herding · gaussian | GaussianKernel | RandomFeatures(512) | 1.78e-7 | 5.36e-7 | 0.0769 |
| mixture-2d | herding · gaussian | GaussianKernel | RandomFeatures(2048) | 8.25e-8 | 1.13e-7 | 0.324 |
| normal-10d | support points · energy | EnergyKernel | Subsample(2000, 8) | 0.00177 | 0.0019 | 0.14 |
| normal-10d | support points · energy | EnergyKernel | RandomSlices(64) | 8.01e-5 | 0.00014 | 0.016 |
| normal-10d | support points · energy | EnergyKernel | RandomSlices(256) | 3.42e-5 | 7.65e-5 | 0.0805 |
| normal-10d | support points · energy | EnergyKernel | RandomSlices(1024) | 2.23e-5 | 4.14e-5 | 0.337 |
| normal-10d | herding · energy | EnergyKernel | Subsample(2000, 8) | 0.00171 | 0.00197 | 0.161 |
| normal-10d | herding · energy | EnergyKernel | RandomSlices(64) | 2.34e-5 | 3.93e-5 | 0.0169 |
| normal-10d | herding · energy | EnergyKernel | RandomSlices(256) | 1.83e-5 | 3.18e-5 | 0.0784 |
| normal-10d | herding · energy | EnergyKernel | RandomSlices(1024) | 7.19e-6 | 1.0e-5 | 0.309 |
| normal-10d | herding · gaussian | GaussianKernel | Subsample(2000, 8) | 0.000148 | 0.000172 | 0.289 |
| normal-10d | herding · gaussian | GaussianKernel | RandomFeatures(512) | 2.71e-7 | 4.19e-7 | 0.0533 |
| normal-10d | herding · gaussian | GaussianKernel | RandomFeatures(2048) | 1.66e-7 | 2.11e-7 | 0.32 |
| uniform-5d | support points · energy | EnergyKernel | Subsample(2000, 8) | 0.00119 | 0.00134 | 0.176 |
| uniform-5d | support points · energy | EnergyKernel | RandomSlices(64) | 9.32e-5 | 0.000111 | 0.0187 |
| uniform-5d | support points · energy | EnergyKernel | RandomSlices(256) | 2.75e-5 | 5.39e-5 | 0.0769 |
| uniform-5d | support points · energy | EnergyKernel | RandomSlices(1024) | 1.74e-5 | 3.86e-5 | 0.308 |
| uniform-5d | herding · energy | EnergyKernel | Subsample(2000, 8) | 0.00114 | 0.00126 | 0.151 |
| uniform-5d | herding · energy | EnergyKernel | RandomSlices(64) | 3.67e-6 | 7.9e-6 | 0.0192 |
| uniform-5d | herding · energy | EnergyKernel | RandomSlices(256) | 2.71e-6 | 4.92e-6 | 0.0826 |
| uniform-5d | herding · energy | EnergyKernel | RandomSlices(1024) | 6.3e-7 | 1.32e-6 | 0.31 |
| uniform-5d | herding · gaussian | GaussianKernel | Subsample(2000, 8) | 0.000138 | 0.000168 | 0.278 |
| uniform-5d | herding · gaussian | GaussianKernel | RandomFeatures(512) | 8.02e-8 | 1.05e-7 | 0.0689 |
| uniform-5d | herding · gaussian | GaussianKernel | RandomFeatures(2048) | 2.86e-8 | 4.11e-8 | 0.318 |
| t3-3d | support points · energy | EnergyKernel | Subsample(2000, 8) | 0.000814 | 0.000957 | 0.157 |
| t3-3d | support points · energy | EnergyKernel | RandomSlices(64) | 4.08e-6 | 1.53e-5 | 0.0184 |
| t3-3d | support points · energy | EnergyKernel | RandomSlices(256) | 6.08e-6 | 1.52e-5 | 0.0736 |
| t3-3d | support points · energy | EnergyKernel | RandomSlices(1024) | 1.99e-6 | 4.5e-6 | 0.312 |
| t3-3d | herding · energy | EnergyKernel | Subsample(2000, 8) | 0.000821 | 0.00109 | 0.162 |
| t3-3d | herding · energy | EnergyKernel | RandomSlices(64) | 1.87e-6 | 2.56e-6 | 0.0174 |
| t3-3d | herding · energy | EnergyKernel | RandomSlices(256) | 4.97e-7 | 1.0e-6 | 0.081 |
| t3-3d | herding · energy | EnergyKernel | RandomSlices(1024) | 2.59e-7 | 5.2e-7 | 0.312 |
| t3-3d | herding · gaussian | GaussianKernel | Subsample(2000, 8) | 0.000153 | 0.000175 | 0.29 |
| t3-3d | herding · gaussian | GaussianKernel | RandomFeatures(512) | 1.74e-7 | 3.13e-7 | 0.064 |
| t3-3d | herding · gaussian | GaussianKernel | RandomFeatures(2048) | 2.09e-7 | 3.85e-7 | 0.338 |

## Decision

Rule: an estimator becomes the automatic `splitquality` fallback if, at
no more than `Subsample(2000, 8)`'s mean wall time, its worst-case max
error over every (dataset, split) row is at most one third of
`Subsample(2000, 8)`'s; otherwise `Subsample(2000, 8)` stays the fallback.
Aggregates below take the worst (maximum) max-abs-error and the mean
mean-time over every row for that kernel.

### EnergyKernel

| estimator | max abs error (worst over rows) | mean time (s) (over rows) |
|---|---:|---:|
| Subsample(2000, 8) | 0.00197 | 0.208 |
| RandomSlices(64) | 0.00014 | 0.0189 |
| RandomSlices(256) | 7.65e-5 | 0.0835 |
| RandomSlices(1024) | 4.14e-5 | 0.331 |

**Decision: `RandomSlices(64)`**

### GaussianKernel

| estimator | max abs error (worst over rows) | mean time (s) (over rows) |
|---|---:|---:|
| Subsample(2000, 8) | 0.000175 | 0.298 |
| RandomFeatures(512) | 5.36e-7 | 0.0658 |
| RandomFeatures(2048) | 3.85e-7 | 0.325 |

**Decision: `RandomFeatures(512)`**
