# Data behind docs/diagrams/split-overview.html, the figure on the README
# and the docs landing page.
#
# Draws 240 rows from a three-component 2D mixture, splits them with the
# default splitter at ratio 0.2, and picks the random 20% draw whose energy
# distance is the median over 200 seeds, so the "random split" panel shows a
# typical draw rather than a lucky or unlucky one. Writes the coordinates,
# both test-index sets, and the energy distances to
# docs/diagrams/.cache/split-overview.json; split-overview.py turns that into
# the HTML source, and export.py renders the SVG pair.
#
#     julia --project=. docs/diagrams/split-overview.jl
#     python docs/diagrams/split-overview.py

using SPlit, Random, Statistics

rng = Xoshiro(7)
N = 240
n1, n2, n3 = 144, 60, 36
A = randn(rng, n1, 2) .* [1.0 0.9]
B = randn(rng, n2, 2) .* [1.6 0.35] .+ [3.2 2.6]
C = randn(rng, n3, 2) .* [0.35 0.35] .+ [-2.6 -2.2]
X = vcat(A, B, C)

result = datasplit(SupportPointSplitter(ratio = 0.2, rng = Xoshiro(2)), X)
q_split = splitquality(X, result)

ntest = length(result.test_indices)
qualities = map(1:200) do seed
  perm = randperm(Xoshiro(seed), N)
  energydistance(X[perm[(ntest+1):end], :], X[perm[1:ntest], :])
end
seed = argmin(abs.(qualities .- median(qualities)))
rand_test = sort(randperm(Xoshiro(seed), N)[1:ntest])
q_rand = qualities[seed]

arr(v) = "[" * join(string.(v), ",") * "]"
cache = joinpath(@__DIR__, ".cache")
mkpath(cache)
open(joinpath(cache, "split-overview.json"), "w") do io
  print(
    io,
    "{\"x\":",
    arr(round.(X[:, 1]; digits = 4)),
    ",\"y\":",
    arr(round.(X[:, 2]; digits = 4)),
    ",\"split_test\":",
    arr(result.test_indices),
    ",\"rand_test\":",
    arr(rand_test),
    ",\"q_split\":",
    q_split,
    ",\"q_rand\":",
    q_rand,
    ",\"q_rand_median\":",
    median(qualities),
    "}",
  )
end
println("split = ", q_split, "  random (seed ", seed, ") = ", q_rand)
