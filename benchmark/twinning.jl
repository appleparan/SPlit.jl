# Twinning at scale: wall time and energy distance of `TwinningSplitter`
# against the current splitters on `normal-10d` at N = 10^4, 10^5, 10^6
# (ratio 0.2). Support points stop at N = 10^5 (the MM repulsion term is
# quadratic in n); herding runs up to 10^6 (one O(N²) data-term pass).
# Scores use `splitquality`'s automatic estimator with a fixed rng, the
# same for every method at a given N. Writes
# `docs/src/assets/benchmarks/twinning.md` and `twinning.png`. Run:
# `julia -t auto --project=benchmark benchmark/twinning.jl [--quick]`.

using SPlit, DataFrames, Random, Statistics, CairoMakie

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

sizes() = QUICK ? [2_000, 5_000] : [10_000, 100_000, 1_000_000]
const SUPPORT_POINTS_MAX_N = 100_000
const HERDING_MAX_N = 1_000_000

function methods(N; rng_seed::Int)
  ms = Tuple{String,AbstractSplitter}[("twinning", TwinningSplitter())]
  N <= HERDING_MAX_N && push!(
    ms,
    (
      "herding · energy",
      HerdingSplitter(kernel = EnergyKernel(), rng = MersenneTwister(rng_seed)),
    ),
  )
  N <= SUPPORT_POINTS_MAX_N && push!(
    ms,
    (
      "support points · energy",
      SupportPointSplitter(
        kernel = EnergyKernel(),
        kappa = 1_000,
        max_iterations = 100,
        rng = MersenneTwister(rng_seed),
      ),
    ),
  )
  return ms
end

normal10(N) = randn(MersenneTwister(2026), N, 10)

random_split(N, n_test, rng) =
  let p = randperm(rng, N)
    SPlit.SplitResult(p[(n_test+1):end], p[1:n_test], true, 0, TwinningSplitter())
  end

score(data, r) = splitquality(data, r; rng = MersenneTwister(11))

rows =
  DataFrame(N = Int[], method = String[], energy_distance = Float64[], seconds = Float64[])

for N in sizes()
  data = normal10(N)
  n_test = round(Int, 0.2N)
  for ((label, s_warm), (_, s)) in zip(methods(N; rng_seed = 0), methods(N; rng_seed = 1))
    datasplit(s_warm, data[1:min(N, 200), :])                  # JIT warm-up on a throwaway copy
    t = @elapsed r = datasplit(s, data)
    q = score(data, r)
    println("N = $N  $label  $(round(t; sigdigits = 3)) s  ED = $(round(q; sigdigits = 3))")
    flush(stdout)
    push!(rows, (N, label, q, t))
  end
  qs = [score(data, random_split(N, n_test, MersenneTwister(100 + i))) for i = 1:5]
  push!(rows, (N, "random", mean(qs), 0.0))
end

io = IOBuffer()
println(io, "| N | method | energy distance | seconds |")
println(io, "|---:|---|---:|---:|")
for r in eachrow(rows)
  println(
    io,
    "| $(r.N) | $(r.method) | $(round(r.energy_distance; sigdigits = 3)) | $(r.method == "random" ? "–" : round(r.seconds; sigdigits = 2)) |",
  )
end
table = String(take!(io))
print(table)
write(joinpath(OUT, "twinning.md"), table)

order = ["twinning", "herding · energy", "support points · energy"]
colors = Makie.wong_colors()[1:3]
fig = Figure(size = (1100, 430))
ax1 = Axis(
  fig[1, 1];
  xscale = log10,
  yscale = log10,
  xlabel = "N",
  ylabel = "seconds (wall)",
  title = "Split time on normal-10d",
)
ax2 = Axis(
  fig[1, 2];
  xscale = log10,
  yscale = log10,
  xlabel = "N",
  ylabel = "relative to random split (lower is better)",
  title = "Energy distance",
)
hlines!(ax2, [1.0]; color = :gray50, linestyle = :dash)
for (m, col) in zip(order, colors)
  sub = filter(r -> r.method == m, rows)
  isempty(sub) && continue
  scatterlines!(ax1, sub.N, max.(sub.seconds, 1e-4); color = col, label = m)
  rel = [
    r.energy_distance /
    only(filter(x -> x.N == r.N && x.method == "random", rows).energy_distance) for
    r in eachrow(sub)
  ]
  scatterlines!(ax2, sub.N, rel; color = col, label = m)
end
axislegend(ax1; position = :lt)
axislegend(ax2; position = :rt)
save(joinpath(OUT, "twinning.png"), fig; px_per_unit = 2)
println("wrote figures to $OUT")
