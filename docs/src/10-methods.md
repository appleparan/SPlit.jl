# [Methods](@id methods)

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
majorization–minimization: at iteration ``t`` every point moves to

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
``f(\xi_{\text{new}}) \le f(\xi) - 10^{-4}\, \langle \nabla f, \xi - \xi_{\text{new}} \rangle``,
starting from twice the previous accepted step; the objective therefore never
increases across accepted steps. When `bandwidth = :median`, ``\sigma`` is
the median pairwise distance over (a sample of) the standardized rows
(Gretton et al., 2012), resolved once per `datasplit` and stored in
`result.method.kernel`. The stochastic `kappa` mode is not available for
this kernel.

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

so for kernels with constant ``k(x, x)`` — the Gaussian kernel (``1``) and the
energy kernel ``k(u,v) = -\|u - v\|`` (``0``) — the herding choice is exactly
the greedy MMD² (energy-distance) minimizer. For the Gaussian kernel (a
bounded feature map), the error ``\mathcal{E}_T`` of Eq. (9) decreases as
``O(1/T)`` (Proposition 1); for the energy kernel only the greedy-step
equivalence with MMD²/energy-distance minimization above is claimed, not the
``O(1/T)`` rate. `herd` computes the exact data term once (``O(N^2)``) and
maintains the running sum over selected rows in ``O(N)`` per selection, for a
total cost of ``O(N^2 + nN)``; the procedure is deterministic for a numeric
kernel.

## Nearest-neighbor assignment

Each support point, in order, claims its nearest not-yet-claimed data row
(Joseph & Vakayil, 2021). `select_nearest` serves the queries from a k-d
tree, doubling the neighbor count and retrying when every returned neighbor
is already claimed. The claimed rows form the smaller subset.

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

- Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel Herding. *UAI*, 109–116.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. *JMLR*, 13, 723–773.
- Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical Analysis and Data Mining*, 15(4), 537–546.
- Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492–502.
- Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562–2592.
- Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics based on distances. *Journal of Statistical Planning and Inference*, 143(8), 1249–1272.
