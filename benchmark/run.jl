using SPlit, DataFrames, Distributions, Random, Statistics, CairoMakie

include(joinpath(@__DIR__, "datasets.jl"))

const QUICK = "--quick" in ARGS
const OUT = let i = findfirst(==("--out"), ARGS)
  i === nothing ? joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks") : ARGS[i+1]
end
mkpath(OUT)

sizes() = QUICK ? [200] : [1_000, 10_000]

function methods(N; rng_seed::Int)
  big = N >= 10_000
  return [
    (
      "support points · energy",
      SupportPointSplitter(
        kernel = EnergyKernel(),
        kappa = big ? 1_000 : nothing,
        rng = MersenneTwister(rng_seed),
      ),
    ),
    (
      "support points · gaussian",
      SupportPointSplitter(
        kernel = GaussianKernel(),
        max_iterations = big ? 100 : 200,
        rng = MersenneTwister(rng_seed),
      ),
    ),
    (
      "herding · energy",
      HerdingSplitter(kernel = EnergyKernel(), rng = MersenneTwister(rng_seed)),
    ),
    (
      "herding · gaussian",
      HerdingSplitter(kernel = GaussianKernel(), rng = MersenneTwister(rng_seed)),
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
  # separate rng seeds so the warm-up run (compilation only, on a throwaway
  # splitter copy) never consumes the timed splitter's own rng stream
  warmup_methods = methods(N; rng_seed = 0)
  timed_methods = methods(N; rng_seed = 1)
  for ((label, s_warmup), (_, s)) in zip(warmup_methods, timed_methods)
    datasplit(s_warmup, data[1:min(N, 200), :])                   # warm-up (compilation)
    t = @elapsed r = datasplit(s, data)
    push!(
      rows,
      (
        name,
        N,
        label,
        splitquality(data, r; exact_threshold = typemax(Int)),
        splitquality(data, r; kernel = gk, exact_threshold = typemax(Int)),
        t,
      ),
    )
    name == "mixture-2d" && N == first(sizes()) && (selections[label] = test_indices(r))
  end
  qs = [
    (
      r = random_split(N, n_test, MersenneTwister(100 + i));
      (
        splitquality(data, r; exact_threshold = typemax(Int)),
        splitquality(data, r; kernel = gk, exact_threshold = typemax(Int)),
      )
    ) for i = 1:5
  ]
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
markers = [:circle, :rect, :utriangle, :diamond]

# 8 (dataset, N) cells in the order rows were generated: N = 1,000 first, then N = 10,000
dataset_names = unique(rows.dataset)
cells = [(N, dname) for N in sizes() for dname in dataset_names]
cell_labels = ["$dname\n$(N ÷ 1000)k" for (N, dname) in cells]
metric_panels = [(:energy_distance, "Energy distance"), (:mmd, "Gaussian MMD")]

# for each optimized method, its discrepancy relative to the random split's, per cell
function relative_quality(metric)
  return map(methods_order[1:4]) do m
    map(cells) do (N, dname)
      sub = filter(r -> r.dataset == dname && r.N == N, rows)
      rand_val = only(filter(r -> r.method == "random", sub)[!, metric])
      m_val = only(filter(r -> r.method == m, sub)[!, metric])
      m_val / rand_val
    end
  end
end

fig = Figure(size = (1200, 460))
for (j, (metric, title)) in enumerate(metric_panels)
  ax = Axis(
    fig[1, j],
    title = title,
    yscale = log10,
    xticks = (1:8, cell_labels),
    xticklabelsize = 12,
    ylabel = j == 1 ? "relative to random split (lower is better)" : "",
  )
  hlines!(ax, [1.0]; color = :gray50, linestyle = :dash)
  vlines!(ax, [4.5]; color = :gray80)
  for (m, col, mk, r) in zip(methods_order[1:4], colors, markers, relative_quality(metric))
    scatter!(ax, 1:8, r; markersize = 14, marker = mk, color = col, label = m)
  end
  j == 2 && axislegend(ax; position = :rt)
end
save(joinpath(OUT, "quality.png"), fig; px_per_unit = 2)

fig2 = Figure(size = (700, 450))
ax2 = Axis(
  fig2[1, 1],
  xscale = log10,
  yscale = log10,
  xlabel = "N",
  ylabel = "seconds (wall)",
  title = "Split time by method and dataset",
)
for (m, col) in zip(methods_order[1:4], colors)
  for (di, dname) in enumerate(unique(rows.dataset))
    sub = filter(r -> r.method == m && r.dataset == dname, rows)
    isempty(sub) && continue
    label_kwargs = di == 1 ? (; label = m) : NamedTuple()
    scatterlines!(ax2, sub.N, max.(sub.seconds, 1e-4); color = col, label_kwargs...)
  end
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
