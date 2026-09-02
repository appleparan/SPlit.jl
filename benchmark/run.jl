using SPlit, DataFrames, Distributions, Random, Statistics, CairoMakie

const QUICK = "--quick" in ARGS
const OUT = let i = findfirst(==("--out"), ARGS)
  i === nothing ? joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks") : ARGS[i+1]
end
mkpath(OUT)

sizes() = QUICK ? [200] : [1_000, 10_000]
datasets(N, rng) = [
  ("mixture-2d", let c = rand(rng, 1:4, N)
    centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
    centers[c, :] .+ randn(rng, N, 2)
  end),
  ("normal-10d", randn(rng, N, 10)),
  ("uniform-5d", rand(rng, N, 5)),
  ("t3-3d", rand(rng, TDist(3), N, 3)),
]

function methods(N)
  big = N >= 10_000
  return [
    (
      "support points · energy",
      SupportPointSplitter(
        kernel = EnergyKernel(),
        kappa = big ? 1_000 : nothing,
        rng = MersenneTwister(1),
      ),
    ),
    (
      "support points · gaussian",
      SupportPointSplitter(
        kernel = GaussianKernel(),
        max_iterations = big ? 100 : 200,
        rng = MersenneTwister(1),
      ),
    ),
    (
      "herding · energy",
      HerdingSplitter(
        kernel = EnergyKernel(),
        kappa = big ? 2_000 : nothing,
        rng = MersenneTwister(1),
      ),
    ),
    (
      "herding · gaussian",
      HerdingSplitter(
        kernel = GaussianKernel(),
        kappa = big ? 2_000 : nothing,
        rng = MersenneTwister(1),
      ),
    ),
  ]
end

random_split(N, n_test, rng) =
  let p = randperm(rng, N)
    SPlit.SplitResult(p[(n_test+1):end], p[1:n_test], true, 0, HerdingSplitter())
  end

rows = DataFrame(
  dataset = String[],
  N = Int[],
  method = String[],
  energy_distance = Float64[],
  mmd = Float64[],
  seconds = Float64[],
)
selections = Dict{String,Vector{Int}}()   # test rows on mixture-2d, N = first size, per method

for N in sizes(), (name, data) in datasets(N, MersenneTwister(2026))
  X = SPlit.preprocess(data)
  gk = SPlit.resolve(GaussianKernel(), X, MersenneTwister(7))   # one bandwidth per dataset
  n_test = round(Int, 0.2N)
  for (label, s) in methods(N)
    datasplit(s, data[1:min(N, 200), :])                          # warm-up (compilation)
    t = @elapsed r = datasplit(s, data)
    push!(
      rows,
      (name, N, label, splitquality(data, r), splitquality(data, r; kernel = gk), t),
    )
    name == "mixture-2d" && N == first(sizes()) && (selections[label] = test_indices(r))
  end
  qs = [(r = random_split(N, n_test, MersenneTwister(100 + i));
  (splitquality(data, r), splitquality(data, r; kernel = gk))) for i = 1:5]
  push!(rows, (name, N, "random", mean(first.(qs)), mean(last.(qs)), 0.0))
  name == "mixture-2d" &&
    N == first(sizes()) &&
    (selections["random"] = test_indices(random_split(N, n_test, MersenneTwister(100))))
end

# ---- table
function markdown_table(df)
  io = IOBuffer()
  println(
    io,
    "| dataset | N | method | energy distance | MMD (Gaussian, median σ) | seconds |",
  )
  println(io, "|---|---:|---|---:|---:|---:|")
  for r in eachrow(df)
    println(
      io,
      "| $(r.dataset) | $(r.N) | $(r.method) | $(round(r.energy_distance; sigdigits = 3)) | $(round(r.mmd; sigdigits = 3)) | $(r.method == "random" ? "–" : round(r.seconds; sigdigits = 2)) |",
    )
  end
  return String(take!(io))
end
table = markdown_table(rows)
print(table)
write(joinpath(OUT, "results.md"), table)

# ---- figures
methods_order = [
  "support points · energy",
  "support points · gaussian",
  "herding · energy",
  "herding · gaussian",
  "random",
]
colors = Makie.wong_colors()[1:5]

fig = Figure(size = (1100, 700))
for (j, N) in enumerate(sizes()), (i, dname) in enumerate(unique(rows.dataset))
  ax = Axis(
    fig[i, j],
    title = "$dname, N = $N",
    yscale = log10,
    xticks = (1:5, ["SP·E", "SP·G", "H·E", "H·G", "rand"]),
    ylabel = i == 1 ? "discrepancy" : "",
  )
  sub = rows[(rows.dataset.==dname).&(rows.N.==N), :]
  idx = [findfirst(==(m), sub.method) for m in methods_order]
  barplot!(
    ax,
    (1:5) .- 0.2,
    max.(sub.energy_distance[idx], 1e-6);
    width = 0.4,
    color = (:gray30, 0.9),
    label = "energy distance",
  )
  barplot!(
    ax,
    (1:5) .+ 0.2,
    max.(sub.mmd[idx], 1e-6);
    width = 0.4,
    color = (:steelblue, 0.9),
    label = "MMD",
  )
  i == 1 && j == 1 && axislegend(ax; position = :rt)
end
save(joinpath(OUT, "quality.png"), fig; px_per_unit = 2)

fig2 = Figure(size = (700, 450))
ax2 = Axis(
  fig2[1, 1],
  xscale = log10,
  yscale = log10,
  xlabel = "N",
  ylabel = "seconds (wall)",
  title = "Split time by method (all datasets)",
)
for (m, col) in zip(methods_order[1:4], colors)
  sub = rows[rows.method.==m, :]
  scatterlines!(ax2, sub.N, max.(sub.seconds, 1e-4); label = m, color = col)
end
axislegend(ax2; position = :lt)
save(joinpath(OUT, "time.png"), fig2; px_per_unit = 2)

data2d = datasets(first(sizes()), MersenneTwister(2026))[1][2]
fig3 = Figure(size = (1300, 300))
for (i, m) in enumerate(methods_order)
  ax = Axis(fig3[1, i], title = m, aspect = DataAspect())
  scatter!(ax, data2d[:, 1], data2d[:, 2]; color = (:gray70, 0.6), markersize = 4)
  sel = selections[m]
  scatter!(ax, data2d[sel, 1], data2d[sel, 2]; color = colors[i], markersize = 7)
  hidedecorations!(ax)
end
save(joinpath(OUT, "selection.png"), fig3; px_per_unit = 2)
println("\nwrote figures to $OUT")
