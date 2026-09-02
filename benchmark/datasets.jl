using Distributions, Random

# Benchmark dataset generators, shared by `run.jl` and `estimators.jl` so
# both scripts draw identical data for a given N and rng.
datasets(N, rng) = [
  ("mixture-2d", let c = rand(rng, 1:4, N)
    centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
    centers[c, :] .+ randn(rng, N, 2)
  end),
  ("normal-10d", randn(rng, N, 10)),
  ("uniform-5d", rand(rng, N, 5)),
  ("t3-3d", rand(rng, TDist(3), N, 3)),
]
