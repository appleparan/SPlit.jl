# Approximate herding data terms (negative result)

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
`O(N²)` data term for `N` around 10⁵. `RandomSlices`/`RandomFeatures`
remain available for `energydistance`/`mmd` quality diagnostics;
`HerdingSplitter`'s data term is exact only.

N = 1500, p = 3, n = 300, 3 rng seeds per row.

| kernel | estimator | selected-subset discrepancy (3 seeds) | exact herding | random | ratio to exact |
|---|---|---|---:|---:|---:|
| EnergyKernel | RandomSlices(64) | 0.0255, 0.0561, 0.0692 (mean 0.0503) | 0.000643 | 0.00713 | 78.2× |
| EnergyKernel | RandomSlices(256) | 0.0181, 0.0222, 0.0262 (mean 0.0222) | 0.000643 | 0.00713 | 34.5× |
| EnergyKernel | RandomSlices(2048) | 0.00467, 0.00486, 0.0108 (mean 0.00679) | 0.000643 | 0.00713 | 10.6× |
| EnergyKernel | RandomSlices(8192) | 0.00138, 0.00252, 0.00228 (mean 0.00206) | 0.000643 | 0.00713 | 3.2× |
| GaussianKernel | RandomFeatures(512) | 0.00448, 0.00385, 0.00407 (mean 0.00413) | 5.88e-5 | 0.00268 | 70.3× |
| GaussianKernel | RandomFeatures(2048) | 0.0019, 0.00168, 0.00171 (mean 0.00177) | 5.88e-5 | 0.00268 | 30.0× |
| GaussianKernel | RandomFeatures(8192) | 0.000474, 0.000503, 0.000343 (mean 0.00044) | 5.88e-5 | 0.00268 | 7.48× |
| GaussianKernel | RandomFeatures(32768) | 0.000203, 0.000225, 0.000187 (mean 0.000205) | 5.88e-5 | 0.00268 | 3.48× |
