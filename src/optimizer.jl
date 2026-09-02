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
using StatsBase: sample
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
function _mm_sweep!(
  new_points::Matrix{Float64},
  current_const::Vector{Float64},
  points::Matrix{Float64},
  subsample_data::AbstractMatrix{Float64},
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
        c += 1.0 / d
        for j = 1:p
          xprime[j] += subsample_data[i, j] / d
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
  _n0_factor::Float64 = 0.2,
)
  N = size(data, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  kappa === nothing ||
    kappa > 0 ||
    throw(ArgumentError("kappa must be positive, got $kappa"))
  max_iterations > 0 ||
    throw(ArgumentError("max_iterations must be positive, got $max_iterations"))

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
  # Implementation constant (not from the papers): running-average weight
  # n0 = 0.2n, chosen by a small convergence experiment; see docstring.
  n0 = _n0_factor * n

  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    verbose && print("\rIteration $iteration/$max_iterations")

    sub = stochastic ? working[sample(rng, 1:N, kappa; replace = false), :] : working
    alpha = stochastic ? n0 / (iteration + n0) : 1.0

    _mm_sweep!(
      new_points,
      current_const,
      points,
      sub,
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

# Test helper: energy objective E(points, data) after each full-data MM sweep.
function _objective_trajectory(
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  traj = Float64[_exact_energydistance(points, data)]
  for _ = 1:max_iterations
    _mm_sweep!(new_points, current_const, points, data, running_const, 1.0, bounds, 1)
    points, new_points = new_points, points
    push!(traj, _exact_energydistance(points, data))
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

# Full gradient of _mmd_objective with respect to every support point.
# Row m of G is (2/n²) Σ_{j≠m} ∇k(ξ_m, ξ_j) − (2/(nN)) Σ_l ∇k(ξ_m, x_l).
# Chunks write disjoint rows of G; `points` and `data` are read-only.
function _mmd_gradient!(
  G::Matrix{Float64},
  k::GaussianKernel{Float64},
  points::Matrix{Float64},
  data::Matrix{Float64},
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
          for j = 1:p
            G[m, j] -= (2 / (n * N)) * g[j]
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
    f_new = _mmd_objective(k, new_points, data)
    if f_new <= f0 - 1e-4 * decrease
      return t, f_new
    end
    t /= 2
  end
  return 0.0, f0
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
`|f_{t-1} - f_t| / max(|f_t|, 1e-12)` is below `rtol`.
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
)
  isresolved(k) ||
    throw(ArgumentError("GaussianKernel bandwidth must be resolved; call resolve first"))
  kappa === nothing ||
    throw(ArgumentError("stochastic mode (kappa) is not available for GaussianKernel yet"))
  N = size(data, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  max_iterations > 0 ||
    throw(ArgumentError("max_iterations must be positive, got $max_iterations"))

  bounds = _data_bounds(data)
  working = copy(data)
  if length(unique(eachrow(working))) < N
    _jitter!(rng, working, bounds)
  end
  points = _initial_points(rng, working, n, bounds)
  new_points = similar(points)
  G = similar(points)
  f = _mmd_objective(k, points, working)
  t = 1.0
  scale = median(view(bounds, :, 2) .- view(bounds, :, 1))

  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    verbose && print("\rIteration $iteration/$max_iterations  objective(mmd2 − const)=$f")
    _mmd_gradient!(G, k, points, working, n_threads)
    t0 =
      iteration == 1 ? 0.1 * scale / max(maximum(norm(view(G, m, :)) for m = 1:n), eps()) :
      2t
    f_prev = f
    t, f = _armijo_step!(new_points, points, G, f, t0, k, working, bounds)
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

# Test helper: objective after each accepted step (full-data Gaussian path).
# Mirrors the scale-aware first step of `support_points(::GaussianKernel, …)`.
function _mmd_trajectory(
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  G = similar(points)
  f = _mmd_objective(k, points, data)
  traj = Float64[f]
  t = 1.0
  scale = median(view(bounds, :, 2) .- view(bounds, :, 1))
  for iteration = 1:max_iterations
    _mmd_gradient!(G, k, points, data, 1)
    t0 =
      iteration == 1 ? 0.1 * scale / max(maximum(norm(view(G, m, :)) for m = 1:n), eps()) :
      2t
    t, f = _armijo_step!(new_points, points, G, f, t0, k, data, bounds)
    t == 0.0 && break
    points, new_points = new_points, points
    push!(traj, f)
  end
  return traj
end
