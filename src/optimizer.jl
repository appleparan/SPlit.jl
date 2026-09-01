"""
Support-point optimization.

Full-data mode applies the closed-form MM update of Mak & Joseph (2018),
which decreases the energy-distance objective monotonically. When `kappa` is
given, the stochastic variant of Joseph & Vakayil (2021) resamples `kappa`
rows per iteration and stabilizes the update with running averages.
"""

using LinearAlgebra
using Random
using StatsBase: sample

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
  n0 = 0.2 * n

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
