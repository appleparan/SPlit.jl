"""
Support-point optimization.

Full-data mode applies the closed-form MM update of Mak & Joseph (2018),
which decreases the energy-distance objective monotonically. When `kappa` is
given, the stochastic variant of Joseph & Vakayil (2021) resamples `kappa`
rows per iteration and stabilizes the update with running averages.
For `GaussianKernel`, support points minimize the squared MMD by projected
gradient descent with Armijo backtracking (Gretton et al. 2012 for the
objective).
"""

using LinearAlgebra
using Random
using StatsBase: sample, Weights
using Statistics: median

function _data_bounds(data::Matrix{Float64})
  p = size(data, 2)
  bounds = Matrix{Float64}(undef, p, 2)
  for j = 1:p
    lo, hi = extrema(view(data, :, j))
    if lo == hi
      lo -= 1e-3
      hi += 1e-3
    end
    bounds[j, 1] = lo
    bounds[j, 2] = hi
  end
  return bounds
end

function _jitter!(rng::AbstractRNG, M::Matrix{Float64}, bounds::Matrix{Float64})
  for j in axes(M, 2)
    amount = (bounds[j, 2] - bounds[j, 1]) * 1e-3
    @views M[:, j] .+= amount .* (2 .* rand(rng, size(M, 1)) .- 1)
    @views M[:, j] .= clamp.(M[:, j], bounds[j, 1], bounds[j, 2])
  end
  return M
end

function _initial_points(
  rng::AbstractRNG,
  data::Matrix{Float64},
  n::Int,
  bounds::Matrix{Float64},
)
  idx = sample(rng, 1:size(data, 1), n; replace = false)
  return _jitter!(rng, data[idx, :], bounds)
end

# One MM sweep over all support points. Reads `points`, writes `new_points`
# and `current_const`; each m is independent, so chunks run in parallel.
# `subsample_weights` are ŵ (mean one) for the rows of `subsample_data`:
# with normalized weights w̄ the update is
#   ξ_m ← [ (1/n) Σ_{o≠m} (ξ_m − ξ_o)/‖ξ_m − ξ_o‖ + Σ_i w̄_i x_i/‖x_i − ξ_m‖ ]
#         / Σ_i w̄_i/‖x_i − ξ_m‖
# (Mak & Joseph 2018, Theorem 3, with the empirical measure replaced by
# Σ w̄_i δ(x_i); the majorizer is the same bound term by term). Multiplying
# numerator and denominator by n_sub gives the form below with ŵ = n_sub w̄
# and the (n_sub/n) factor on the repulsion term. Uniform weights make
# ŵ ≡ 1.0 exactly, so the arithmetic is the unweighted one bit for bit.
function _mm_sweep!(
  new_points::Matrix{Float64},
  current_const::Vector{Float64},
  points::Matrix{Float64},
  subsample_data::AbstractMatrix{Float64},
  subsample_weights::AbstractVector{Float64},
  running_const::Vector{Float64},
  alpha::Float64,
  bounds::Matrix{Float64},
  n_threads::Int,
)
  n, p = size(points)
  n_sub = size(subsample_data, 1)
  chunks = collect(Iterators.partition(1:n, cld(n, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn for m in chunk
      xprime = zeros(p)
      for o = 1:n
        o == m && continue
        s = 0.0
        for j = 1:p
          s += (points[m, j] - points[o, j])^2
        end
        d = sqrt(s) + eps(Float64)
        for j = 1:p
          xprime[j] += (points[m, j] - points[o, j]) / d
        end
      end
      xprime .*= n_sub / n
      c = 0.0
      for i = 1:n_sub
        s = 0.0
        for j = 1:p
          s += (subsample_data[i, j] - points[m, j])^2
        end
        d = sqrt(s) + eps(Float64)
        wi = subsample_weights[i]
        c += wi / d
        for j = 1:p
          xprime[j] += wi * subsample_data[i, j] / d
        end
      end
      current_const[m] = c
      denom = (1 - alpha) * running_const[m] + alpha * c
      if denom > 0
        for j = 1:p
          xprime[j] =
            ((1 - alpha) * running_const[m] * points[m, j] + alpha * xprime[j]) / denom
        end
      else
        for j = 1:p
          xprime[j] = points[m, j]
        end
      end
      for j = 1:p
        new_points[m, j] = clamp(xprime[j], bounds[j, 1], bounds[j, 2])
      end
    end
  end
  return nothing
end

# Stochastic-mode subsample: row indices and their ŵ (mean one within the
# subsample). `:uniform` draws rows uniformly and rescales their weights;
# `:proportional` draws rows with probability ∝ w and treats them as uniform.
function _draw_subsample(
  rng::AbstractRNG,
  N::Int,
  kappa::Int,
  w_hat::Vector{Float64},
  ::Val{:uniform},
)
  idx = sample(rng, 1:N, kappa; replace = false)
  sub = w_hat[idx]
  sum(sub) > 0 || throw(
    ArgumentError("the kappa subsample drew only zero-weight rows; use a larger kappa"),
  )
  return idx, _mean_one_weights(sub)
end
function _draw_subsample(
  rng::AbstractRNG,
  N::Int,
  kappa::Int,
  w_hat::Vector{Float64},
  ::Val{:proportional},
)
  idx = sample(rng, 1:N, Weights(w_hat), kappa; replace = false)
  return idx, ones(kappa)
end

"""
    support_points(kernel, data, n; kwargs...) -> (points, converged, iterations)

Compute `n` support points for `data` (rows are observations) under `kernel`.
Returns the points, whether the point-movement tolerance was reached, and the
number of iterations actually used.

Convergence compares the largest *squared* displacement of any support point
in one iteration to `tolerance`. In stochastic mode (`kappa !== nothing`),
the running-average weight for iteration `i` is `n0 / (i + n0)` with
`n0 = 0.2n`, which decays toward zero as iterations proceed, so convergence
there partly reflects this step-size decay rather than the objective
flattening out. `n0 = 0.2n` is an implementation constant, not from the
papers, chosen by a small convergence experiment (see `_n0_factor`, an
internal tuning knob not exposed on `SupportPointSplitter`).

`weights` (one non-negative entry per row, `nothing` for uniform) makes the
points approximate the weighted empirical distribution `Σ w̄ᵢ δ(xᵢ)`: the
data sums in the MM update carry `ŵᵢ = N w̄ᵢ`, which is exactly `1.0` for
uniform weights. In stochastic mode `_subsampling` (internal) selects how
the `kappa` rows are drawn: `:uniform` draws them uniformly and rescales
their weights to mean one within the subsample; `:proportional` draws them
with probability proportional to the weights and treats the subsample as
uniform (this needs at least `kappa` rows with positive weight). The
default was chosen by the weighted-`kappa` experiment on the Design
experiments page. A constant weight vector is treated as `nothing`, so
uniform weights take the unweighted path and reproduce it exactly.
"""
function support_points(
  ::EnergyKernel,
  data::Matrix{Float64},
  n::Int;
  kappa::Union{Nothing,Int} = nothing,
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
  verbose::Bool = false,
  weights::Union{Nothing,AbstractVector} = nothing,
  _n0_factor::Float64 = 0.2,
  _subsampling::Symbol = :uniform,
)
  N = size(data, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  kappa === nothing ||
    kappa > 0 ||
    throw(ArgumentError("kappa must be positive, got $kappa"))
  max_iterations > 0 ||
    throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
  _subsampling in (:uniform, :proportional) || throw(
    ArgumentError("_subsampling must be :uniform or :proportional, got :$_subsampling"),
  )
  weights === nothing || _check_weights(weights, N)
  weights = _uniform_as_nothing(weights)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(weights)

  bounds = _data_bounds(data)
  working = copy(data)
  if length(unique(eachrow(working))) < N
    _jitter!(rng, working, bounds)
  end

  points = _initial_points(rng, working, n, bounds)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  stochastic = kappa !== nothing && kappa < N
  rule = Val(_subsampling)
  # Implementation constant (not from the papers): running-average weight
  # n0 = 0.2n, chosen by a small convergence experiment; see docstring.
  n0 = _n0_factor * n

  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    verbose && print("\rIteration $iteration/$max_iterations")

    if stochastic
      idx, sub_w = _draw_subsample(rng, N, kappa, w_hat, rule)
      sub = working[idx, :]
    else
      sub, sub_w = working, w_hat
    end
    alpha = stochastic ? n0 / (iteration + n0) : 1.0

    _mm_sweep!(
      new_points,
      current_const,
      points,
      sub,
      sub_w,
      running_const,
      alpha,
      bounds,
      n_threads,
    )

    max_move = 0.0
    @views for m = 1:n
      max_move = max(max_move, sum(abs2, new_points[m, :] .- points[m, :]))
    end
    points, new_points = new_points, points
    if stochastic
      running_const .= (1 - alpha) .* running_const .+ alpha .* current_const
    end
    converged = max_move < tolerance
  end
  verbose && println()

  return points, converged, iteration
end

# Test helper: energy objective E(points, data) after each full-data MM sweep,
# weighted when `weights` is given.
function _objective_trajectory(
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
  weights::Union{Nothing,AbstractVector} = nothing,
)
  N = size(data, 1)
  weights === nothing || _check_weights(weights, N)
  weights = _uniform_as_nothing(weights)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(weights)
  w_bar = weights === nothing ? _uniform_weights(N) : _normalize_weights(weights, N)
  u = _uniform_weights(n)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  traj = Float64[_exact_energydistance(points, data, u, w_bar)]
  for _ = 1:max_iterations
    _mm_sweep!(
      new_points,
      current_const,
      points,
      data,
      w_hat,
      running_const,
      1.0,
      bounds,
      1,
    )
    points, new_points = new_points, points
    push!(traj, _exact_energydistance(points, data, u, w_bar))
  end
  return traj
end

# MMD² objective up to the constant mean k(x, x): mean k(ξ, ξ) − 2 mean k(ξ, x).
function _mmd_objective(
  k::GaussianKernel{Float64},
  points::AbstractMatrix{Float64},
  data::AbstractMatrix{Float64},
)
  return _mean_kernel(k, points, points) - 2 * _mean_kernel(k, points, data)
end

_mmd_objective(
  k::GaussianKernel{Float64},
  points::AbstractMatrix{Float64},
  data::AbstractMatrix{Float64},
  ::Nothing,
) = _mmd_objective(k, points, data)

# Weighted MMD² objective up to the constant data self-term:
# mean k(ξ, ξ) − 2 Σ_l w̄_l mean_m k(ξ_m, x_l), with w̄ scaled to sum one.
function _mmd_objective(
  k::GaussianKernel{Float64},
  points::AbstractMatrix{Float64},
  data::AbstractMatrix{Float64},
  w_bar::AbstractVector{Float64},
)
  n = size(points, 1)
  return _mean_kernel(k, points, points) -
         2 * _mean_kernel(k, points, data, _uniform_weights(n), w_bar)
end

# Full gradient of _mmd_objective with respect to every support point.
# Row m of G is (2/n²) Σ_{j≠m} ∇k(ξ_m, ξ_j) − (2/n) Σ_l w̄_l ∇k(ξ_m, x_l);
# with ŵ = N w̄ (mean one, exactly 1.0 for uniform weights) the data term is
# (2/(nN)) Σ_l ŵ_l ∇k(ξ_m, x_l). Chunks write disjoint rows of G; `points`
# and `data` are read-only.
function _mmd_gradient!(
  G::Matrix{Float64},
  k::GaussianKernel{Float64},
  points::Matrix{Float64},
  data::Matrix{Float64},
  w_hat::AbstractVector{Float64},
  n_threads::Int,
)
  n, p = size(points)
  N = size(data, 1)
  chunks = collect(Iterators.partition(1:n, cld(n, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn begin
      g = zeros(p)
      for m in chunk
        @views ξ = points[m, :]
        for j = 1:p
          G[m, j] = 0.0
        end
        for o = 1:n
          o == m && continue
          @views kernelgrad!(g, k, ξ, points[o, :])
          for j = 1:p
            G[m, j] += (2 / n^2) * g[j]
          end
        end
        for l = 1:N
          @views kernelgrad!(g, k, ξ, data[l, :])
          wl = w_hat[l]
          for j = 1:p
            G[m, j] -= (2 / (n * N)) * wl * g[j]
          end
        end
      end
    end
  end
  return G
end

# One projected-gradient step with Armijo backtracking on the projected step:
# ξ_new is ξ − tG clamped to the bounding box, and the accepted step size t
# is the largest tried (starting from t0, halving) satisfying
# f(ξ_new) ≤ f(ξ) − 1e-4 · ⟨G, ξ − ξ_new⟩ — the sufficient-decrease test
# against the actual projected move rather than the unprojected ‖G‖², so a
# point held at the bounding box (where ξ − ξ_new can be much smaller than
# tG) can still be accepted as converged. Returns (accepted step size,
# objective at the accepted points), or (0.0, f0) if none of the 30 tried
# steps decreased the objective.
function _armijo_step!(
  new_points::Matrix{Float64},
  points::Matrix{Float64},
  G::Matrix{Float64},
  f0::Float64,
  t0::Float64,
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  bounds::Matrix{Float64},
  w_bar::Union{Nothing,AbstractVector{Float64}},
)
  t = t0
  for _ = 1:30
    @inbounds for m in axes(points, 1), j in axes(points, 2)
      new_points[m, j] = clamp(points[m, j] - t * G[m, j], bounds[j, 1], bounds[j, 2])
    end
    decrease = 0.0
    @inbounds for m in axes(points, 1), j in axes(points, 2)
      decrease += G[m, j] * (points[m, j] - new_points[m, j])
    end
    f_new = _mmd_objective(k, new_points, data, w_bar)
    if f_new <= f0 - 1e-4 * decrease
      return t, f_new
    end
    t /= 2
  end
  return 0.0, f0
end

# Scale-aware initial trial step for the first Armijo iteration: a tenth of
# the median per-dimension data range, divided by the largest gradient row
# norm (guarded against a near-zero gradient by `eps()`). The gradient
# carries 1/n², 1/(nN) factors whose magnitude varies enormously with n and
# N, so a fixed t0 = 1.0 can be far too small; this keeps the first move a
# fixed fraction of the data scale regardless. Shared by
# `support_points(::GaussianKernel, …)` and `_mmd_trajectory`.
function _first_step(G::Matrix{Float64}, bounds::Matrix{Float64})
  n = size(G, 1)
  scale = median(view(bounds, :, 2) .- view(bounds, :, 1))
  return 0.1 * scale / max(maximum(norm(view(G, m, :)) for m = 1:n), eps())
end

"""
    support_points(kernel::GaussianKernel, data, n; kwargs...)

Support points under a Gaussian kernel: minimize the squared MMD between the
point set and the data by projected gradient descent with Armijo
backtracking on the projected step (the objective never increases across
accepted steps). The kernel must be resolved (numeric bandwidth);
`datasplit` resolves it. The stochastic `kappa` mode is not available for
this kernel yet.

The first trial step is scale-aware: `t0 = 0.1 * scale / max(‖∇f‖, eps())`,
with `scale` the median per-dimension data range and `‖∇f‖` the largest
gradient row norm, so the initial move is a tenth of the data scale
regardless of the point count or gradient magnitude (Armijo backtracking and
the `2t` warm start on later iterations are unchanged). Convergence never
fires before the second iteration, and then when either the largest squared
displacement is below `tolerance` or the relative objective decrease
`|f_{t-1} - f_t| / max(|f_t|, 1e-12)` is below `rtol`. `f` is
`_mmd_objective`, which omits the constant data self-term and is bounded in
`[-1, 1]` for a Gaussian kernel, so `rtol` acts as an
absolute per-iteration tolerance on that bounded objective, not a relative
tolerance on the (orders-of-magnitude smaller) true MMD².

`weights` (one non-negative entry per row, `nothing` for uniform) makes the
points minimize the MMD² to the weighted empirical distribution
`Σ w̄ᵢ δ(xᵢ)`: the data term of the objective and of the gradient carries
`w̄`, with `ŵ = N w̄` (exactly `1.0` for uniform weights) inside the
gradient loop, so unweighted results are unchanged.
"""
function support_points(
  k::GaussianKernel,
  data::Matrix{Float64},
  n::Int;
  kappa::Union{Nothing,Int} = nothing,
  max_iterations::Int = 500,
  tolerance::Float64 = 1e-10,
  rtol::Float64 = 1e-8,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
  verbose::Bool = false,
  weights::Union{Nothing,AbstractVector} = nothing,
)
  isresolved(k) ||
    throw(ArgumentError("GaussianKernel bandwidth must be resolved; call resolve first"))
  kappa === nothing ||
    throw(ArgumentError("stochastic mode (kappa) is not available for GaussianKernel yet"))
  N = size(data, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  max_iterations > 0 ||
    throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
  rtol > 0 || throw(ArgumentError("rtol must be positive, got $rtol"))
  weights === nothing || _check_weights(weights, N)
  weights = _uniform_as_nothing(weights)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(weights)
  w_bar = weights === nothing ? nothing : _normalize_weights(weights, N)

  bounds = _data_bounds(data)
  working = copy(data)
  if length(unique(eachrow(working))) < N
    _jitter!(rng, working, bounds)
  end
  points = _initial_points(rng, working, n, bounds)
  new_points = similar(points)
  G = similar(points)
  f = _mmd_objective(k, points, working, w_bar)
  t = 1.0

  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    verbose && print("\rIteration $iteration/$max_iterations  objective(mmd2 − const)=$f")
    _mmd_gradient!(G, k, points, working, w_hat, n_threads)
    t0 = iteration == 1 ? _first_step(G, bounds) : 2t
    f_prev = f
    t, f = _armijo_step!(new_points, points, G, f, t0, k, working, bounds, w_bar)
    t == 0.0 && break   # no decreasing step found: stop, report not converged
    max_move = 0.0
    @views for m = 1:n
      max_move = max(max_move, sum(abs2, new_points[m, :] .- points[m, :]))
    end
    points, new_points = new_points, points
    rel = abs(f_prev - f) / max(abs(f), 1e-12)
    converged = iteration >= 2 && (max_move < tolerance || rel < rtol)
  end
  verbose && println()
  return points, converged, iteration
end

# Test helper: objective after each accepted step (full-data Gaussian path),
# weighted when `weights` is given. Mirrors the scale-aware first step of
# `support_points(::GaussianKernel, …)`.
function _mmd_trajectory(
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
  weights::Union{Nothing,AbstractVector} = nothing,
)
  N = size(data, 1)
  weights === nothing || _check_weights(weights, N)
  weights = _uniform_as_nothing(weights)
  w_hat = weights === nothing ? ones(N) : _mean_one_weights(weights)
  w_bar = weights === nothing ? nothing : _normalize_weights(weights, N)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  G = similar(points)
  f = _mmd_objective(k, points, data, w_bar)
  traj = Float64[f]
  t = 1.0
  for iteration = 1:max_iterations
    _mmd_gradient!(G, k, points, data, w_hat, 1)
    t0 = iteration == 1 ? _first_step(G, bounds) : 2t
    t, f = _armijo_step!(new_points, points, G, f, t0, k, data, bounds, w_bar)
    t == 0.0 && break
    points, new_points = new_points, points
    push!(traj, f)
  end
  return traj
end
