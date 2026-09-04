# Flattening time-series windows into rows for SPlit.jl.
#
# A window of `L` timesteps over `p` variables is flattened to one row of
# length `L*p` (variable-major), so any splitter can select a
# distribution-preserving subset of windows. Demonstrates the fixture from
# the design doc, checks that point-level moments match across two regimes
# with the same mean/variance but different temporal dependence while
# window-level (flattened) distributions differ, compares every splitter
# against random and against each other, recovers the original time-series
# slices for a few selected windows, and runs two contrasts: representing
# less than the dependence length (contrast 1) and representing far more
# dimensions than a splitter can use well (contrast 2, the `L*p` ladder).
# Prints a markdown table per section and writes them all to one file.
#
# Setup (from the repository root; path="." is the checkout):
#   julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia -t auto --project=examples examples/time_series_windows.jl
#
# Run (contrast 2 can take minutes at L = 4096): julia -t auto --project=examples examples/time_series_windows.jl
# Options: --quick (small sizes, short ladder, for a fast smoke run), --out PATH

using SPlit, DataFrames, LinearAlgebra, Printf, Random, Statistics

include(joinpath(@__DIR__, "time_series_windows_helpers.jl"))

# Value of `flag`, or `default` when the flag is absent; a trailing flag
# without a value is an error rather than a `BoundsError`.
function argvalue(flag, default)
  i = findfirst(==(flag), ARGS)
  i === nothing && return default
  i < length(ARGS) || error("$flag needs a value")
  return ARGS[i+1]
end

const QUICK = "--quick" in ARGS
const OUT = argvalue(
  "--out",
  joinpath(@__DIR__, "..", "docs", "src", "assets", "examples", "time_series_windows.md"),
)

const M_MAIN = QUICK ? 200 : 1000              # windows in the main demo and contrast 1
const N_MAIN = QUICK ? 20 : 100                # selected windows (10%)
const L_MAIN = 32                              # dependence length of the "persistent" regime
const P_MAIN = 3
const SHARE_A = 0.7
const SEEDS_STOCHASTIC = QUICK ? 1 : 3
const RANDOM_DRAWS_MAIN = QUICK ? 5 : 20
const RANDOM_DRAWS_LADDER = QUICK ? 5 : 10
const LADDER1 = [1, 2, 4, 8, 16, 32]           # contrast 1: L_short
const DATA_SEEDS_CONTRAST1 = QUICK ? 2 : 5     # contrast 1: independent datasets to average over
# Ladder stops at L = 1024 (L·p = 3,072): TwinningSplitter's brute-force path
# and SupportPointSplitter's `select_nearest` build static-vector
# nearest-neighbor structures (NearestNeighbors' BruteTree/KDTree) sized to
# the row width; that compilation fails outright at L·p = 12,288 (L = 4096,
# "invalid syntax (memory-error out of gc handles)") and did not finish
# within 7 minutes at L·p = 6,144 (L = 2048). Measured 2026-09-05.
const LADDER2 = QUICK ? [8, 64] : [8, 64, 512, 1024]   # contrast 2: L
const M_LADDER2 = 2000
const N_LADDER2 = 200
const KAPPA_LADDER2 = 500

random_rows(n, N, rng) = randperm(rng, N)[1:n]

# A single markdown string, built section by section, printed as it grows
# (useful for a detached long run) and written to `OUT` at the end.
const REPORT = IOBuffer()
function emit(str)
  println(str)
  println(REPORT, str)
end

# ---- Fixture
let
  X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0; 5.0 50.0]
  L, stride = 2, 2
  Z, starts = windows(X, L; stride = stride)
  io = IOBuffer()
  println(io, "## Fixture\n")
  println(io, "`X` (N = 5 rows, p = 2 variables), `L = $L`, `stride = $stride`:\n")
  println(io, "```text")
  println(io, "X = ", X)
  println(io, "windows(X, $L; stride = $stride) = ", Z)
  println(io, "starts = ", starts)
  println(io, "```\n")
  println(io, "Row `i` is `vec(X[starts[i]:starts[i]+L-1, :])`: variable-major, all `L`")
  println(io, "offsets of variable 1 then variable 2. The trailing row 5 does not fill a")
  println(io, "full window and is dropped, not padded.")
  emit(String(take!(io)))
end

# ---- Synthetic demo: point-level moments and energy distance, point vs window level
rng_data = MersenneTwister(1)
X, labels =
  two_regime_series(rng_data; M = M_MAIN, L = L_MAIN, p = P_MAIN, share_a = SHARE_A)
Z, starts = windows(X, L_MAIN; stride = L_MAIN)
Zs = standardize_by_variable(Z, L_MAIN, P_MAIN)
idx_a = findall(==(:A), labels)
idx_b = findall(==(:B), labels)

let
  points_a = reduce(vcat, [recover_window(X, starts[m], L_MAIN) for m in idx_a])
  points_b = reduce(vcat, [recover_window(X, starts[m], L_MAIN) for m in idx_b])
  io = IOBuffer()
  println(io, "## Point vs window level\n")
  println(
    io,
    "M = $M_MAIN windows of length $L_MAIN (regime A share ≈ $SHARE_A). Point-level",
  )
  println(io, "per-variable mean and variance should match across regimes:\n")
  println(io, "| variable | mean A | mean B | var A | var B |")
  println(io, "|---|---:|---:|---:|---:|")
  for v = 1:P_MAIN
    @printf(
      io,
      "| %d | %.3g | %.3g | %.3g | %.3g |\n",
      v,
      mean(view(points_a, :, v)),
      mean(view(points_b, :, v)),
      var(view(points_a, :, v)),
      var(view(points_b, :, v)),
    )
  end
  ed_point = energydistance(points_a, points_b)
  ed_window = energydistance(view(Zs, idx_a, :), view(Zs, idx_b, :))
  println(io)
  @printf(io, "Energy distance, A vs B, point level: %.3g\n\n", ed_point)
  @printf(
    io,
    "Energy distance, A vs B, window level (flattened, standardized): %.3g\n",
    ed_window,
  )
  println(
    io,
    "\nThe temporal dependence that distinguishes the regimes is invisible at the",
  )
  println(io, "point level and only shows up once each window is kept as one sample.")
  emit(String(take!(io)))
end

# ---- Selectors vs random
function main_metrics(sel)
  ed = energydistance(view(Zs, sel, :), Zs)
  prop_err = abs(mean(view(labels, sel) .== :A) - SHARE_A)
  ac = mean(lag1_autocorrelation(view(Z, i, :), L_MAIN, P_MAIN) for i in sel)
  return ed, prop_err, ac
end

# Mean ± sd; a spread below rounding noise (identical draws) prints as 0.
function fmt_stat(v::AbstractVector)
  length(v) > 1 || return @sprintf("%.3g", v[1])
  s = std(v)
  return @sprintf("%.3g ± %.3g", mean(v), s < 1e-9 ? 0.0 : s)
end

main_rows = DataFrame(
  method = String[],
  energy_distance = String[],
  regime_prop_error = String[],
  mean_lag1_autocorr = String[],
)

random_main_draws =
  [random_rows(N_MAIN, M_MAIN, MersenneTwister(1000 + d)) for d = 1:RANDOM_DRAWS_MAIN]
random_main_metrics = main_metrics.(random_main_draws)
push!(
  main_rows,
  (
    "random",
    fmt_stat(first.(random_main_metrics)),
    fmt_stat(getindex.(random_main_metrics, 2)),
    fmt_stat(last.(random_main_metrics)),
  ),
)

for (label, sel) in (
  ("twinning", selectrows(TwinningSplitter(), Zs, N_MAIN; standardize = false)),
  (
    "herding · energy",
    selectrows(HerdingSplitter(kernel = EnergyKernel()), Zs, N_MAIN; standardize = false),
  ),
)
  ed, prop_err, ac = main_metrics(sel)
  push!(
    main_rows,
    (label, @sprintf("%.3g", ed), @sprintf("%.3g", prop_err), @sprintf("%.3g", ac)),
  )
end

for (label, splitter_of) in (
  ("kernel thinning · energy", seed -> KernelThinningSplitter(rng = MersenneTwister(seed))),
  (
    "support points · energy",
    seed ->
      SupportPointSplitter(kappa = 300, max_iterations = 100, rng = MersenneTwister(seed)),
  ),
)
  ms = [
    main_metrics(selectrows(splitter_of(seed), Zs, N_MAIN; standardize = false)) for
    seed = 1:SEEDS_STOCHASTIC
  ]
  push!(
    main_rows,
    (label, fmt_stat(first.(ms)), fmt_stat(getindex.(ms, 2)), fmt_stat(last.(ms))),
  )
end

let
  io = IOBuffer()
  println(io, "## Selectors vs random\n")
  println(
    io,
    "n = $N_MAIN of M = $M_MAIN windows (L = $L_MAIN, p = $P_MAIN), evaluated on the",
  )
  println(
    io,
    "standardized flattened windows. Random: $RANDOM_DRAWS_MAIN draws; stochastic",
  )
  println(io, "selectors: $SEEDS_STOCHASTIC seeds.\n")
  println(
    io,
    "| method | energy distance | regime-proportion error | mean lag-1 autocorrelation |",
  )
  println(io, "|---|---:|---:|---:|")
  for r in eachrow(main_rows)
    @printf(
      io,
      "| %s | %s | %s | %s |\n",
      r.method,
      r.energy_distance,
      r.regime_prop_error,
      r.mean_lag1_autocorr,
    )
  end
  emit(String(take!(io)))
end

# ---- datasplit: interpolation, not forecasting; recovering original windows
let
  # Twinning's groups are formed from the whole time range, so train and test
  # windows interleave in time: this measures how well the smaller set
  # interpolates the whole distribution, not whether it can forecast beyond
  # the observed range (a chronological holdout would be needed for that).
  result = datasplit(TwinningSplitter(), Zs; standardize = false)
  sel = result.selected === :train ? train_indices(result) : test_indices(result)
  io = IOBuffer()
  println(io, "## datasplit\n")
  println(io, "`datasplit(TwinningSplitter(), Zs; standardize = false)`: $(length(sel)) of")
  println(io, "$M_MAIN windows on the `$(result.selected)` side.\n")
  println(io, "Recovering the original time-series slice for the first 3 selected windows")
  println(
    io,
    "(each stays a separate L × p sample; selected windows are never concatenated):\n",
  )
  println(io, "```text")
  for i in first(sel, 3)
    w = recover_window(X, starts[i], L_MAIN)
    println(io, "window $i (rows $(starts[i]):$(starts[i]+L_MAIN-1)):")
    println(io, w[1:min(3, L_MAIN), :], " …")
  end
  println(io, "```")
  emit(String(take!(io)))
end

# ---- Contrast 1: L below the dependence length
let
  # Twinning is deterministic, so run-to-run noise in the ratio to random
  # comes from the dataset, not the selector: average over
  # DATA_SEEDS_CONTRAST1 independently generated datasets rather than
  # judging L_short on the main demo's single series.
  ratios = [Float64[] for _ in LADDER1]
  prop_errs = [Float64[] for _ in LADDER1]
  for d = 1:DATA_SEEDS_CONTRAST1
    Xd, labels_d = two_regime_series(
      MersenneTwister(500 + d);
      M = M_MAIN,
      L = L_MAIN,
      p = P_MAIN,
      share_a = SHARE_A,
    )
    Zd, _ = windows(Xd, L_MAIN; stride = L_MAIN)
    Zsd = standardize_by_variable(Zd, L_MAIN, P_MAIN)

    random_draws_d = [
      random_rows(N_MAIN, M_MAIN, MersenneTwister(1000 + d * 100 + k)) for
      k = 1:RANDOM_DRAWS_MAIN
    ]
    ed_random_d = mean(energydistance(view(Zsd, sel, :), Zsd) for sel in random_draws_d)

    for (i, L_short) in enumerate(LADDER1)
      Zshort, _ = windows(Xd, L_short; stride = L_MAIN)   # same segment starts as the L = 32 windowing
      Zshorts = standardize_by_variable(Zshort, L_short, P_MAIN)
      sel = selectrows(TwinningSplitter(), Zshorts, N_MAIN; standardize = false)
      ed = energydistance(view(Zsd, sel, :), Zsd)         # evaluated in this dataset's full-L space
      push!(ratios[i], ed / ed_random_d)
      push!(prop_errs[i], abs(mean(view(labels_d, sel) .== :A) - SHARE_A))
    end
  end

  rows = DataFrame(
    L_short = LADDER1,
    ratio_to_random = fmt_stat.(ratios),
    regime_prop_error = fmt_stat.(prop_errs),
  )

  io = IOBuffer()
  println(io, "## Contrast 1: representation below the dependence length\n")
  println(
    io,
    "TwinningSplitter, n = $N_MAIN of M = $M_MAIN windows, representation built from",
  )
  println(
    io,
    "only the first `L_short` rows of each length-$L_MAIN segment, evaluated in the",
  )
  println(
    io,
    "full L = $L_MAIN space (dependence length ≈ 1/(1-stay_a) ≈ 16), averaged over",
  )
  println(
    io,
    "$DATA_SEEDS_CONTRAST1 independently generated datasets (mean ± sd over data seeds).\n",
  )
  println(io, "| L_short | ratio to random | regime-proportion error |")
  println(io, "|---:|---:|---:|")
  for r in eachrow(rows)
    @printf(io, "| %d | %s | %s |\n", r.L_short, r.ratio_to_random, r.regime_prop_error)
  end
  emit(String(take!(io)))
end

# ---- Contrast 2: the L*p dimension ladder
let
  rows = DataFrame(
    L = Int[],
    Lp = Int[],
    method = String[],
    compile_seconds = Float64[],
    seconds = Float64[],
    energy_distance = Float64[],
    ratio_to_random = Float64[],
  )
  for L in LADDER2
    # Warm up per L, not once: the static-vector nearest-neighbor structures
    # (NearestNeighbors' BruteTree for twinning's brute-force path, KDTree for
    # select_nearest) are compiled fresh for each row width, so a warm-up at a
    # single fixed width would still leave every other ladder width paying
    # its own width-specific compilation on the timed call below. Run on a
    # small throwaway matrix of the same width as this L (separate rng seeds
    # from the timed runs) and record the elapsed time as `compile_seconds`.
    warmup = randn(MersenneTwister(0), 60, L * P_MAIN)
    t_compile_twin = @elapsed selectrows(TwinningSplitter(), warmup, 6; standardize = false)
    t_compile_sp = @elapsed selectrows(
      SupportPointSplitter(kappa = 30, max_iterations = 3, rng = MersenneTwister(0)),
      warmup,
      6;
      standardize = false,
    )

    X2, _ = two_regime_series(MersenneTwister(2000 + L); M = M_LADDER2, L = L, p = P_MAIN)
    Z2, _ = windows(X2, L; stride = L)
    Zs2 = standardize_by_variable(Z2, L, P_MAIN)

    # Cached pairwise-distance matrix (Gram-matrix trick, LinearAlgebra only)
    # so the O(M² L p) full-set term is paid once per L instead of once per
    # selector call — energydistance() alone recomputes it every call.
    G = Zs2 * Zs2'
    sq = diag(G)
    D = sqrt.(max.(sq .+ sq' .- 2 .* G, 0.0))
    mean_yy = mean(D)
    ed_cached(sel) = 2 * mean(view(D, sel, :)) - mean(view(D, sel, sel)) - mean_yy

    if L == first(LADDER2)
      sel_check = collect(1:min(50, M_LADDER2))
      ed_direct = energydistance(view(Zs2, sel_check, :), Zs2)
      isapprox(ed_cached(sel_check), ed_direct; rtol = 1e-6) ||
        error("cached energy distance disagrees with energydistance at L = $L")
    end

    random_draws = [
      random_rows(N_LADDER2, M_LADDER2, MersenneTwister(3000 + L * 100 + d)) for
      d = 1:RANDOM_DRAWS_LADDER
    ]
    ed_random = mean(ed_cached.(random_draws))
    push!(rows, (L, L * P_MAIN, "random", NaN, NaN, ed_random, 1.0))

    t_twin = @elapsed sel_twin =
      selectrows(TwinningSplitter(), Zs2, N_LADDER2; standardize = false)
    ed_twin = ed_cached(sel_twin)
    push!(
      rows,
      (L, L * P_MAIN, "twinning", t_compile_twin, t_twin, ed_twin, ed_twin / ed_random),
    )

    times_sp = Float64[]
    eds_sp = Float64[]
    for seed = 1:SEEDS_STOCHASTIC
      s = SupportPointSplitter(
        kappa = KAPPA_LADDER2,
        max_iterations = 100,
        rng = MersenneTwister(4000 + L * 100 + seed),
      )
      t = @elapsed sel_sp = selectrows(s, Zs2, N_LADDER2; standardize = false)
      push!(times_sp, t)
      push!(eds_sp, ed_cached(sel_sp))
    end
    push!(
      rows,
      (
        L,
        L * P_MAIN,
        "support points · energy",
        t_compile_sp,
        minimum(times_sp),
        mean(eds_sp),
        mean(eds_sp) / ed_random,
      ),
    )
  end

  io = IOBuffer()
  println(io, "## Contrast 2: the L·p dimension ladder\n")
  println(
    io,
    "M = $M_LADDER2, n = $N_LADDER2, L ∈ $(LADDER2). Support points: $SEEDS_STOCHASTIC",
  )
  println(io, "seeds (min time, mean energy distance); random: $RANDOM_DRAWS_LADDER draws.")
  println(
    io,
    "The cached energy distance at L = $(first(LADDER2)) was checked to agree with",
  )
  println(io, "`energydistance` directly. \"compile seconds\" is the first call at that")
  println(
    io,
    "width, on a throwaway 60-row matrix of the same width, paying the width-specific",
  )
  println(
    io,
    "compilation of the static-vector nearest-neighbor structures; \"seconds\" is the",
  )
  println(io, "timed run that follows it.\n")
  println(
    io,
    "| L | L·p | method | compile seconds | seconds | energy distance | ratio to random |",
  )
  println(io, "|---:|---:|---|---:|---:|---:|---:|")
  for r in eachrow(rows)
    @printf(
      io,
      "| %d | %d | %s | %s | %s | %.3g | %.3g |\n",
      r.L,
      r.Lp,
      r.method,
      r.method == "random" ? "–" : @sprintf("%.2g", r.compile_seconds),
      r.method == "random" ? "–" : @sprintf("%.2g", r.seconds),
      r.energy_distance,
      r.ratio_to_random,
    )
  end
  emit(String(take!(io)))
end

# ---- write
mkpath(dirname(OUT))
write(OUT, String(take!(REPORT)))
println("\nwrote $OUT")
