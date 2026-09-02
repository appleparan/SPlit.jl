# [Methods](@id methods)

See [How SPlit works](@ref intuition) for the same quantities explained with
analogies instead of formulas.

This page states the quantities SPlit.jl computes and names the function that
implements each one. Rows of the data matrix are observations
``x_1, \dots, x_N`` after preprocessing (`preprocess`: Helmert encoding of
categorical columns, constant-column removal, standardization). Support
points are ``\xi_1, \dots, \xi_n`` with ``n`` the size of the smaller subset.

## Energy distance and support points

The energy distance between two samples (Székely & Rizzo) is

```math
\mathrm{ED}(X, Y) = \frac{2}{|X||Y|} \sum_{x \in X} \sum_{y \in Y} \|x - y\|
  - \frac{1}{|X|^2} \sum_{x, x' \in X} \|x - x'\|
  - \frac{1}{|Y|^2} \sum_{y, y' \in Y} \|y - y'\|.
```

`energydistance` evaluates this V-statistic exactly by block-wise
accumulation. With `subsample = m` it averages the statistic over random
size-``m`` subsets; because the within-sample terms keep the zero diagonal,
that estimate carries a positive bias of order ``1/m`` and is meant for
comparing splits, not as an absolute value.

Support points (Mak & Joseph, 2018) minimize ``\mathrm{ED}(\{\xi\}, \{x\})``
over the point locations. The objective is minimized by
majorization-minimization: at iteration ``t`` every point moves to

```math
\xi_m^{(t+1)} = \frac{\displaystyle \frac{N}{n} \sum_{o \ne m}
  \frac{\xi_m^{(t)} - \xi_o^{(t)}}{\|\xi_m^{(t)} - \xi_o^{(t)}\|}
  + \sum_{l=1}^{N} \frac{x_l}{\|x_l - \xi_m^{(t)}\|}}
  {\displaystyle \sum_{l=1}^{N} \frac{1}{\|x_l - \xi_m^{(t)}\|}},
```

clamped to the data's bounding box. Each sweep decreases the energy distance
monotonically (`support_points(::EnergyKernel, …)`; the sweep is
`_mm_sweep!`). With `kappa` ``= \kappa < N`` the stochastic variant of
Joseph & Vakayil (2021) uses a fresh random subset of ``\kappa`` rows per
iteration and blends the update with a running average whose weight
``n_0 / (t + n_0)``, ``n_0 = 0.2\,n``, decays with ``t``. Iteration stops when
the largest squared displacement of any point falls below `tolerance`.

## Maximum mean discrepancy and the Gaussian kernel

For a kernel ``k`` the squared maximum mean discrepancy (Gretton et al.,
2012) between two samples is

```math
\mathrm{MMD}^2(X, Y) = \frac{1}{|X|^2} \sum_{x, x'} k(x, x')
  + \frac{1}{|Y|^2} \sum_{y, y'} k(y, y')
  - \frac{2}{|X||Y|} \sum_{x, y} k(x, y).
```

`mmd` evaluates it (block-wise, with the same optional subsampling and the
same bias caveat as `energydistance`). With ``k(u, v) = -\|u - v\|`` this is
the energy distance, which is why `mmd` covers both `EnergyKernel` and
`GaussianKernel`.

`GaussianKernel(σ)` uses ``k(u, v) = \exp\!\left(-\|u - v\|^2 / 2\sigma^2\right)``.
Support points under it minimize, up to a constant,

```math
f(\xi) = \frac{1}{n^2} \sum_{i,j} k(\xi_i, \xi_j)
  - \frac{2}{nN} \sum_{i,l} k(\xi_i, x_l),
\qquad
\nabla_{\xi_m} f = \frac{2}{n^2} \sum_{j \ne m} \nabla_u k(\xi_m, \xi_j)
  - \frac{2}{nN} \sum_{l} \nabla_u k(\xi_m, x_l),
```

with ``\nabla_u k(u, v) = -k(u, v)\,(u - v)/\sigma^2``. The optimizer
(`support_points(::GaussianKernel, …)`) takes projected gradient steps
``\xi_{\text{new}} = \operatorname{clamp}(\xi - t \nabla f)`` where ``t`` is
chosen by Armijo backtracking on the projected step,
``f(\xi_{\text{new}}) \le f(\xi) - 10^{-4}\, \langle \nabla f, \xi - \xi_{\text{new}} \rangle``;
the objective therefore never increases across accepted steps. The gradient
carries ``1/n^2`` and ``1/(nN)`` factors whose magnitude varies enormously
with ``n`` and ``N``, so the first trial step (`_first_step`) is scale-aware
rather than a fixed constant: ``t_0 = 0.1\,\bar w / \max_m \|\nabla_m f\|``,
with ``\bar w`` the median per-dimension data range, making the first move a
tenth of the data scale regardless of ``n``, ``N``. Later iterations warm-start
from twice the previous accepted step. Convergence never fires before the
second iteration, and then when *either* the largest squared displacement is
below `tolerance` *or* the relative objective decrease
``|f_{t-1} - f_t| / \max(|f_t|, 10^{-12})`` is below `rtol` (default
``10^{-8}``); `f` here is the shifted objective above, which omits the
constant data self-term and is bounded in ``[-1, 1]`` for a Gaussian kernel,
so `rtol` acts as an absolute-in-effect tolerance
rather than a tolerance on the (orders-of-magnitude smaller) true MMD².
When `bandwidth = :median`, ``\sigma`` is the median pairwise distance over
(a sample of) the standardized rows (Gretton et al., 2012), resolved once
per `datasplit` and stored in `result.method.kernel`. The stochastic
`kappa` mode is not available for this kernel.

## Kernel herding

`HerdingSplitter` builds the smaller subset row by row. With data
``x_1, \dots, x_N`` and rows ``s_1, \dots, s_T`` already selected, the next
row is (Chen, Welling & Smola, 2010, Eq. 8, applied to the empirical
distribution and restricted to unselected rows)

```math
s_{T+1} = \arg\max_{x \notin \{s_1,\dots,s_T\}}
  \; \frac{1}{N} \sum_{l=1}^{N} k(x, x_l) \;-\; \frac{1}{T+1} \sum_{t=1}^{T} k(x, s_t).
```

Appending ``x`` changes the MMD² between the selected rows and the data by

```math
\Delta(x) = \frac{k(x,x)}{(T+1)^2} + \frac{2}{(T+1)^2} \sum_{t=1}^{T} k(x, s_t)
  - \frac{2}{(T+1)N} \sum_{l=1}^{N} k(x, x_l) + \text{const},
```

so for kernels with constant ``k(x, x)``, namely the Gaussian kernel (``1``)
and the energy kernel ``k(u,v) = -\|u - v\|`` (``0``), the herding choice is exactly
the greedy MMD² (energy-distance) minimizer. For the Gaussian kernel (a
bounded feature map), the error ``\mathcal{E}_T`` of Eq. (9) decreases as
``O(1/T)`` (Proposition 1); for the energy kernel only the greedy-step
equivalence with MMD²/energy-distance minimization above is claimed, not the
``O(1/T)`` rate. `herd` computes the exact data term once (``O(N^2)``) and
maintains the running sum over selected rows in ``O(N)`` per selection, for a
total cost of ``O(N^2 + nN)``; the procedure is deterministic for a numeric
kernel.

## Estimators

`energydistance`, `mmd`, and `splitquality` accept an `estimator` keyword
that selects how the discrepancy above is computed. Which estimator/kernel
combinations exist is expressed by method dispatch (a method per
combination, never an `if`); an undefined combination raises an
`ArgumentError`.

| estimator | `energydistance` (`EnergyKernel`) | `mmd` (`GaussianKernel`) |
|---|---|---|
| `Exact` | yes, threaded | yes, threaded |
| `Subsample(m, repeats)` | yes | yes |
| `RandomSlices(k)` | yes | no |
| `RandomFeatures(D)` | no | yes |

### RandomSlices: the projection identity

For ``\theta`` uniform on the unit sphere ``S^{p-1}`` and any
``u \in \mathbb{R}^p``,

```math
\mathbb{E}_\theta\,|\langle \theta, u \rangle| = \kappa_p \|u\|, \qquad
\kappa_p = \frac{\Gamma(p/2)}{\sqrt{\pi}\,\Gamma\!\left((p+1)/2\right)},
```

computed by `sphere_constant` via the recursion ``\kappa_1 = 1``,
``\kappa_2 = 2/\pi``, ``\kappa_{p+2} = \kappa_p\,p/(p+1)``. The energy
distance is linear in the pairwise norms, so with ``u^\theta = X\theta``,
``v^\theta = Y\theta``,

```math
\mathrm{ED}(X, Y) = \kappa_p^{-1}\, \mathbb{E}_\theta\, \mathrm{ED}_1(u^\theta, v^\theta),
```

and `RandomSlices(k)` averages this over `k` directions drawn with `rng`
(`_sliced_energydistance`). The one-dimensional energy distance
``\mathrm{ED}_1`` is computed exactly from sorted projections
(`_ed1d`): for a sorted sample ``a_{(1)} \le \dots \le a_{(n)}``,

```math
\sum_{i<j} (a_{(j)} - a_{(i)}) = \sum_i (2i - n - 1)\, a_{(i)}
```

gives the within-sample mean (`_within_mean_abs`), and the cross term
``\sum_{i,j} |a_i - b_j|`` follows from prefix sums of one sorted sample and
the ranks of the other (`_cross_mean_abs`), for a total cost of
``O(k (n+m) \log(n+m))``.

### RandomFeatures: random Fourier features

For the Gaussian kernel ``k(x,y) = \exp(-\|x-y\|^2/2\sigma^2)``, with
``\omega_j \sim \mathcal{N}(0, \sigma^{-2} I_p)`` and
``b_j \sim U[0, 2\pi]`` (Rahimi & Recht, 2007),

```math
z(x) = \sqrt{2/D}\,\big[\cos(\omega_j^\top x + b_j)\big]_{j=1}^{D}, \qquad
\mathbb{E}\big[z(x)^\top z(y)\big] = k(x, y),
```

drawn once per call from `rng` as `FourierFeatureMap`. `RandomFeatures(D)`
estimates squared MMD as ``\|\bar z_X - \bar z_Y\|^2`` with
``\bar z_X = \frac{1}{n}\sum_i z(x_i)`` (`_rff_mmd`, `_feature_mean`), an
unbiased estimator of the V-statistic, cost ``O((n+m)Dp)``.

### Exact: threaded

`_mean_pairwise` and `_mean_kernel` split their block loop over row-block
pairs across `n_threads` spawned tasks, each writing disjoint entries of a
preallocated accumulator that is then summed in a fixed pair order, so the
result is identical for every thread count, not just numerically close.

### Automatic rule for `splitquality`

`estimator = nothing` (the default) selects `Exact()` when the total row
count is at most `exact_threshold` (20,000), and otherwise the fallback
chosen by the [Estimator selection](@ref estimator-selection) experiment
on the Design experiments page (`_fallback_estimator`).

### Why herding stays exact

`RandomSlices`/`RandomFeatures` give an unbiased data term for kernel
herding too, but measurement rejected them: every candidate row's estimate
shares the same random directions or features, so the estimator noise is
correlated across rows, and the greedy `argmax` follows that correlated
noise into a direction-dependent region of the data rather than averaging it
out. At feasible budgets the selected subset was worse than a random subset.
See [Approximate herding data terms (rejected)](@ref herding-estimators-rejected)
on the Design experiments page for the table. `HerdingSplitter`'s data
term (`_data_term`) therefore stays exact only.

## Nearest-neighbor assignment

Each support point, in order, claims its nearest not-yet-claimed data row
(Joseph & Vakayil, 2021). `select_nearest` serves the queries from a k-d
tree, doubling the neighbor count and retrying when every returned neighbor
is already claimed. The claimed rows form the smaller subset.

This rounding step has a limitation: when the optimizer's displacement is
below the spacing between data rows, as is typical in high dimension on
standardized data, every point's nearest row is still its own starting row,
so the claimed subset is exactly the initial random sample and the
optimization has no effect on which rows are selected. Measured on the
Benchmarks page. `HerdingSplitter` selects rows directly and has no rounding
step, so it is unaffected.

## Optimal split ratio

For a linear model with ``p`` parameters (intercept included), Joseph (2022,
Eq. 11) gives the test fraction that minimizes the variance of the fitted
model:

```math
\gamma^* = \frac{1}{\sqrt{p} + 1}.
```

`optimal_split_ratio(x, y)` takes ``p`` as the number of encoded predictor
columns plus one.

## References

- Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel Herding. *UAI*, 109-116.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. *JMLR*, 13, 723-773.
- Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining*, 15(4), 537-546.
- Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502.
- Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.
- Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NIPS*, 20.
- Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics based on distances. *Journal of Statistical Planning and Inference*, 143(8), 1249-1272.
