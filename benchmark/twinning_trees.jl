# Nearest-neighbor structure for twinning: wall time of `_twin_groups` with
# a k-d tree and with brute force on standard-normal data at N = 10,000,
# ratio 0.2, for dimensions 2 to 768. Decides
# `SPlit.TWINNING_BRUTE_FORCE_DIMENSION` (Design experiments page). Writes
# `docs/src/assets/benchmarks/twinning_trees.md`. Run:
# `julia --project=benchmark benchmark/twinning_trees.jl` (serial by design).

using SPlit, Random, Statistics

const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const N = 10_000
const DIMS = [2, 10, 50, 200, 768]
const REPEATS = 3

function timed(X, brute)
  SPlit._twin_groups(X[1:500, :], 100, 1, MersenneTwister(0); brute_force = brute)  # warm-up
  return minimum(
    (@elapsed SPlit._twin_groups(X, N ÷ 5, 1, MersenneTwister(0); brute_force = brute)) for
    _ = 1:REPEATS
  )
end

io = IOBuffer()
println(io, "| p | k-d tree (s) | brute force (s) | brute / k-d |")
println(io, "|---:|---:|---:|---:|")
for p in DIMS
  X = SPlit.preprocess(randn(MersenneTwister(p), N, p))
  t_kd = timed(X, false)
  t_bf = timed(X, true)
  line = "| $p | $(round(t_kd; sigdigits = 3)) | $(round(t_bf; sigdigits = 3)) | $(round(t_bf / t_kd; sigdigits = 3)) |"
  println(line)
  println(io, line)
end
write(joinpath(OUT, "twinning_trees.md"), String(take!(io)))
println("wrote $(joinpath(OUT, "twinning_trees.md"))")
