# Negative-result benchmark: approximating the kernel-herding data term with
# RandomSlices/RandomFeatures makes greedy selection unreliable, because all
# candidate rows share the same random directions/features and the greedy
# argmax follows the resulting correlated noise. The table it writes is
# embedded on the Benchmarks page.
#
# For each estimator budget: draw the (noisy) data term with the package's
# internal estimator building blocks, run a local copy of the exact greedy
# loop against it, and compare the selected subset's discrepancy to exact
# herding and to a random subset. Run: `julia -t auto --project=benchmark
# benchmark/herding_estimators.jl`.

using SPlit
using Random
using Statistics

const N = 1_500
const P = 3
const N_SELECT = 300
const SEEDS = (133, 200, 201)

# Local copy of SPlit's exact greedy loop (Chen, Welling & Smola 2010, Eq. 8),
# taking the data term `d` as an argument instead of always computing it
# exactly — the package itself no longer offers this.
function greedy_select(kernel, X::Matrix{Float64}, n::Int, d::Vector{Float64})
  N = size(X, 1)
  c = zeros(N)
  used = falses(N)
  selected = Vector{Int}(undef, n)
  for T = 0:(n-1)
    best, bestscore = 0, -Inf
    for i = 1:N
      used[i] && continue
      score = d[i] - c[i] / (T + 1)
      score > bestscore && ((best, bestscore) = (i, score))
    end
    selected[T+1] = best
    used[best] = true
    @views for i = 1:N
      c[i] += SPlit.kernelvalue(kernel, X[i, :], X[best, :])
    end
  end
  return selected
end

function random_slices_data_term(X::Matrix{Float64}, k::Int, rng::AbstractRNG)
  n_rows, p = size(X)
  κ = SPlit.sphere_constant(p)
  Θ = SPlit._project_directions(rng, p, k)
  d = zeros(n_rows)
  rank = Vector{Int}(undef, n_rows)
  for j = 1:k
    u = X * @view(Θ[:, j])
    order = sortperm(u)
    for (r, i) in enumerate(order)
      rank[i] = r
    end
    prefix = cumsum(u[order])
    for i = 1:n_rows
      r = rank[i]
      d[i] -= (u[i] * (2r - n_rows) - 2 * prefix[r] + prefix[n_rows]) / (k * κ * n_rows)
    end
  end
  return d
end

function random_features_data_term(
  kernel::GaussianKernel{Float64},
  X::Matrix{Float64},
  D::Int,
  rng::AbstractRNG,
)
  n_rows, p = size(X)
  φ = SPlit.FourierFeatureMap(kernel, p, D, rng)
  z̄ = SPlit._feature_mean(φ, X)
  return [sum(φ(@view X[i, :]) .* z̄) for i = 1:n_rows]
end

random_baseline(kernel, X, n; repeats = 10) = mean(
  mmd(X[randperm(MersenneTwister(300 + i), size(X, 1))[1:n], :], X, kernel) for
  i = 1:repeats
)

X = randn(MersenneTwister(130), N, P)
rows = NamedTuple[]

energy = EnergyKernel()
exact_energy = mmd(X[SPlit.herd(energy, X, N_SELECT), :], X, energy)
rand_energy = random_baseline(energy, X, N_SELECT)
for k in (64, 256, 2048, 8192)
  qs = [
    mmd(
      X[
        greedy_select(
          energy,
          X,
          N_SELECT,
          random_slices_data_term(X, k, MersenneTwister(s)),
        ),
        :,
      ],
      X,
      energy,
    ) for s in SEEDS
  ]
  push!(
    rows,
    (
      kernel = "EnergyKernel",
      estimator = "RandomSlices($k)",
      qs = qs,
      exact = exact_energy,
      random = rand_energy,
    ),
  )
end

gauss = GaussianKernel(1.0)
exact_gauss = mmd(X[SPlit.herd(gauss, X, N_SELECT), :], X, gauss)
rand_gauss = random_baseline(gauss, X, N_SELECT)
for D in (512, 2048, 8192, 32768)
  qs = [
    mmd(
      X[
        greedy_select(
          gauss,
          X,
          N_SELECT,
          random_features_data_term(gauss, X, D, MersenneTwister(s)),
        ),
        :,
      ],
      X,
      gauss,
    ) for s in SEEDS
  ]
  push!(
    rows,
    (
      kernel = "GaussianKernel",
      estimator = "RandomFeatures($D)",
      qs = qs,
      exact = exact_gauss,
      random = rand_gauss,
    ),
  )
end

fmt(x) = string(round(x; sigdigits = 3))
lines = [
  "| kernel | estimator | selected-subset discrepancy (3 seeds) | exact herding | random | ratio to exact |",
  "|---|---|---|---:|---:|---:|",
]
for r in rows
  qs_str = join(fmt.(r.qs), ", ")
  ratio = mean(r.qs) / r.exact
  push!(
    lines,
    "| $(r.kernel) | $(r.estimator) | $qs_str (mean $(fmt(mean(r.qs)))) | $(fmt(r.exact)) | $(fmt(r.random)) | $(fmt(ratio))× |",
  )
end
table = join(lines, "\n")
println(table)

out_path =
  joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks", "herding_estimators.md")
open(out_path, "w") do io
  println(
    io,
    """
    # Approximate herding data terms — negative result

    Kernel herding's data term (`mean_l k(x_i, x_l)`, Chen, Welling & Smola
    2010, Eq. 8) was tried with `RandomSlices`/`RandomFeatures` approximations
    and rejected: all candidate rows share the same random directions or
    features, so the estimator's noise is correlated across rows, and greedy
    `argmax` selection tracks that noise rather than averaging it out. In the
    table below the small budgets select subsets *worse than a random subset*,
    the mid budgets roughly match random, and only the largest budgets come
    within about 3× of exact herding — at which point the estimator's own cost
    (`O(kN log N)` for slices, `O(NDp)` for Fourier features) matches the exact
    `O(N²)` data term for `N` around 10⁵. `RandomSlices`/`RandomFeatures`
    remain available for `energydistance`/`mmd` quality diagnostics;
    `HerdingSplitter`'s data term is exact only.

    N = $N, p = $P, n = $N_SELECT, 3 rng seeds per row.
    """,
  )
  println(io, table)
end
println("\nWrote ", out_path)
