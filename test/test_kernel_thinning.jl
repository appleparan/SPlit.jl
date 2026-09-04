using Test
using SPlit
using Random
using Statistics
using DataFrames
using LinearAlgebra

# Kernel halving exactly as KT 2024, Alg. 2, with the paper's α form
# (Σ over all previous points minus twice the Σ over S₁), consuming one rng
# draw per step with a > 0, as the implementation does.
function naive_kernel_halving(kernel, X, seq, δ, rng)
  S1 = Int[]
  S2 = Int[]
  σ = 0.0
  for i = 1:(length(seq)÷2)
    x, x′ = seq[2i-1], seq[2i]
    k(u, v) = SPlit.kernelvalue(kernel, X[u, :], X[v, :])
    b = sqrt(max(k(x, x) + k(x′, x′) - 2k(x, x′), 0.0))
    if b > 0
      a = max(b * σ * sqrt(2 * log(2 / δ)), b^2)
      σ = sqrt(σ^2 + b^2 * max(0.0, 1 + (b^2 - 2a) * σ^2 / a^2))
      prev = seq[1:(2i-2)]
      α =
        sum(k(j, x) - k(j, x′) for j in prev; init = 0.0) -
        2 * sum(k(z, x) - k(z, x′) for z in S1; init = 0.0)
      if rand(rng) < min(1.0, 0.5 * max(0.0, 1 - α / a))
        x, x′ = x′, x
      end
    end
    push!(S1, x)
    push!(S2, x′)
  end
  return S1, S2
end

@testset "kernel halving" begin
  @testset "swap parameters follow the paper's update" begin
    a, σ = SPlit._swap_params(0.0, 2.0, 0.1)
    @test a == 4.0 && σ == 2.0                      # σ = 0: a = b², σ² = b²
    a2, σ2 = SPlit._swap_params(σ, 1.0, 0.1)
    @test a2 == max(1.0 * σ * sqrt(2 * log(20.0)), 1.0)
    @test σ2^2 ≈ σ^2 + 1.0 * max(0.0, 1 + (1.0 - 2a2) * σ^2 / a2^2)
    @test SPlit._swap_params(1.5, 0.0, 0.1) == (0.0, 1.5)   # identical rows: no threshold, σ kept
  end

  @testset "difference sums match a plain loop and ignore n_threads" begin
    X = SPlit.preprocess(randn(MersenneTwister(1), 3_000, 3))
    Xt = permutedims(X)
    idx = collect(1:2_900)
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      @views ref = sum(
        SPlit.kernelvalue(kernel, X[j, :], X[2_999, :]) -
        SPlit.kernelvalue(kernel, X[j, :], X[3_000, :]) for j in idx
      )
      @views s1 = SPlit._kernel_diff_sum(kernel, Xt, idx, Xt[:, 2_999], Xt[:, 3_000], 1)
      @views s4 = SPlit._kernel_diff_sum(kernel, Xt, idx, Xt[:, 2_999], Xt[:, 3_000], 4)
      @test s1 ≈ ref
      @test s1 == s4
    end
  end

  @testset "halving equals the paper's algorithm step for step" begin
    for (kernel, seed) in ((EnergyKernel(), 2), (GaussianKernel(0.8), 3))
      X = SPlit.preprocess(randn(MersenneTwister(seed), 201, 2))
      seq = randperm(MersenneTwister(seed + 10), 201)
      S1, S2 = SPlit._kernel_halving(
        kernel,
        permutedims(X),
        seq,
        0.5 / 200,
        MersenneTwister(7);
        n_threads = 2,
      )
      T1, T2 = naive_kernel_halving(kernel, X, seq, 0.5 / 200, MersenneTwister(7))
      @test S1 == T1 && S2 == T2
      @test length(S1) == 100 && sort(vcat(S1, S2)) == sort(seq[1:200])   # odd trailing row dropped
    end
  end

  @testset "reproducible under the same rng, independent of n_threads" begin
    X = SPlit.preprocess(randn(MersenneTwister(4), 400, 3))
    Xt = permutedims(X)
    seq = collect(1:400)
    a = SPlit._kernel_halving(
      EnergyKernel(),
      Xt,
      seq,
      1e-3,
      MersenneTwister(5);
      n_threads = 1,
    )
    b = SPlit._kernel_halving(
      EnergyKernel(),
      Xt,
      seq,
      1e-3,
      MersenneTwister(5);
      n_threads = 4,
    )
    @test a == b
  end

  @testset "halves are more balanced than random halves" begin
    rng = MersenneTwister(6)
    N = 400
    c = rand(rng, 1:4, N)
    centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
    X = SPlit.preprocess(centers[c, :] .+ randn(rng, N, 2))
    Xt = permutedims(X)
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      S1, _ = SPlit._kernel_halving(
        kernel,
        Xt,
        collect(1:N),
        0.5 / N,
        MersenneTwister(8);
        n_threads = 2,
      )
      q = mmd(X[S1, :], X, kernel)
      random_q = mean(
        mmd(X[randperm(MersenneTwister(100 + i), N)[1:200], :], X, kernel) for i = 1:20
      )
      @test q < random_q
    end
  end
end

# KT-SWAP objective up to its constant: (1/n²) Σ_{a,b∈S} k(a,b) − (2/n) Σ_{a∈S} d(a).
function swap_objective(kernel, X, S, d)
  n = length(S)
  self = sum(SPlit.kernelvalue(kernel, X[a, :], X[b, :]) for a in S, b in S)
  return self / n^2 - 2 * sum(d[S]) / n
end

@testset "KT-SPLIT and KT-SWAP" begin
  X = SPlit.preprocess(randn(MersenneTwister(20), 480, 3))
  Xt = permutedims(X)

  @testset "KT-SPLIT: 2^m candidates of size n partitioning the sequence" begin
    seq = randperm(MersenneTwister(21), 480)
    cands =
      SPlit._kt_split(EnergyKernel(), Xt, seq, 3, 0.5, MersenneTwister(22); n_threads = 2)
    @test length(cands) == 8
    @test all(c -> length(c) == 60, cands)
    @test sort(reduce(vcat, cands)) == sort(seq)
    # level order: the first two candidates come from the first-level S₁
    S1, _ = SPlit._kernel_halving(
      EnergyKernel(),
      Xt,
      seq,
      0.5 / (3 * 480),
      MersenneTwister(22);
      n_threads = 2,
    )
    @test sort(vcat(cands[1], cands[2], cands[3], cands[4])) == sort(S1)
  end

  @testset "KT-SWAP: never worse than the baseline, monotone, distinct rows" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      d = SPlit._data_term(kernel, X, 2)
      seq = randperm(MersenneTwister(23), 480)
      cands = SPlit._kt_split(kernel, Xt, seq, 2, 0.5, MersenneTwister(24); n_threads = 2)
      baseline = randperm(MersenneTwister(25), 480)[1:120]
      rows, swaps = SPlit._kt_swap(kernel, Xt, cands, baseline, d, 2)
      @test length(rows) == 120 && allunique(rows) && all(in(1:480), rows)
      obj = swap_objective(kernel, X, rows, d)
      @test obj <=
            minimum(swap_objective(kernel, X, c, d) for c in vcat(cands, [baseline])) +
            1e-12
      @test mmd(X[rows, :], X, kernel) <= mmd(X[baseline, :], X, kernel) + 1e-12
      @test swaps >= 0
      # the objective differs from the exact MMD² by a constant: check on two candidates
      c1, c2 = cands[1], cands[2]
      @test (swap_objective(kernel, X, c1, d) - swap_objective(kernel, X, c2, d)) ≈
            (mmd(X[c1, :], X, kernel) - mmd(X[c2, :], X, kernel)) atol = 1e-9
    end
  end

  @testset "KT-SWAP result is independent of n_threads" begin
    d = SPlit._data_term(EnergyKernel(), X, 1)
    cands = SPlit._kt_split(
      EnergyKernel(),
      Xt,
      collect(1:480),
      2,
      0.5,
      MersenneTwister(26);
      n_threads = 1,
    )
    baseline = collect(1:120)
    @test SPlit._kt_swap(EnergyKernel(), Xt, cands, baseline, d, 1) ==
          SPlit._kt_swap(EnergyKernel(), Xt, cands, baseline, d, 4)
  end

  @testset "_kt_swap rejects a candidate size mismatch" begin
    d = SPlit._data_term(EnergyKernel(), X, 1)
    @test_throws ArgumentError SPlit._kt_swap(
      EnergyKernel(),
      Xt,
      [collect(1:10)],
      collect(1:12),
      d,
      1,
    )
  end

  @testset "kernel_thinning: sizes, validation, reproducibility" begin
    rows, swaps = SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(30))
    @test length(rows) == 96 && allunique(rows)
    @test (rows, swaps) ==
          SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(30))
    @test rows != SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(31))[1]
    rows300, _ = SPlit.kernel_thinning(EnergyKernel(), X, 300; rng = MersenneTwister(34))
    rows180, _ = SPlit.kernel_thinning(EnergyKernel(), X, 180; rng = MersenneTwister(34))
    @test length(rows300) == 300 && allunique(rows300)   # n > N ÷ 2: the complement of N - n
    @test sort(rows300) == sort(setdiff(1:480, rows180))
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 480)   # n = N
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 0)
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 96; delta = 0.0)
    @test_throws ArgumentError SPlit.kernel_thinning(GaussianKernel(), X, 96)   # unresolved kernel
    # ratio 0.25 uses every row in KT-SPLIT (L = N); 0.2 uses L = 0.8 N
    @test length(
      SPlit.kernel_thinning(EnergyKernel(), X, 120; rng = MersenneTwister(32))[1],
    ) == 120
  end

  @testset "kernel_thinning end-to-end result is independent of n_threads" begin
    @test SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      96;
      n_threads = 1,
      rng = MersenneTwister(72),
    ) == SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      96;
      n_threads = 4,
      rng = MersenneTwister(72),
    )
  end

  @testset "kernel_thinning beats random subsets under its own discrepancy" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      rows, _ = SPlit.kernel_thinning(kernel, X, 96; rng = MersenneTwister(33))
      q = mmd(X[rows, :], X, kernel)
      random_q = mean(
        mmd(X[randperm(MersenneTwister(200 + i), 480)[1:96], :], X, kernel) for i = 1:20
      )
      @test q < random_q
    end
  end

  @testset "weights and target enter through the swap objective" begin
    w = ones(480)
    @test SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      96;
      weights = w,
      rng = MersenneTwister(40),
    ) == SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(40))
    heavy = X[:, 1] .> 0
    w2 = ifelse.(heavy, 20.0, 1.0)
    plain, _ = SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(41))
    weighted, _ =
      SPlit.kernel_thinning(EnergyKernel(), X, 96; weights = w2, rng = MersenneTwister(41))
    @test count(heavy[weighted]) > count(heavy[plain])
    R = X[heavy, :]
    targeted, _ =
      SPlit.kernel_thinning(EnergyKernel(), X, 96; target = R, rng = MersenneTwister(41))
    @test count(heavy[targeted]) > count(heavy[plain])
    @test_throws ArgumentError SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      96;
      weights = w2,
      target = R,
    )
  end
end

@testset "KernelThinningSplitter" begin
  @testset "construction, validation, show" begin
    s = KernelThinningSplitter()
    @test s isa AbstractSplitter
    @test s.kernel isa EnergyKernel && s.ratio == 0.2 && s.delta == 0.5
    @test KernelThinningSplitter(
      kernel = GaussianKernel(),
      ratio = 1 // 4,
      delta = 0.1,
    ).delta == 0.1
    @test_throws ArgumentError KernelThinningSplitter(ratio = 0.0)
    @test_throws ArgumentError KernelThinningSplitter(delta = 1.0)
    @test_throws ArgumentError KernelThinningSplitter(n_threads = 0)
    @test occursin(
      "KernelThinningSplitter(kernel=EnergyKernel(), ratio=0.2, delta=0.5, compress=:auto)",
      sprint(show, s),
    )
  end

  @testset "datasplit: partition, sizes, report, both kernels" begin
    data = randn(MersenneTwister(50), 300, 3)
    for kernel in (EnergyKernel(), GaussianKernel())
      r =
        datasplit(KernelThinningSplitter(kernel = kernel, rng = MersenneTwister(51)), data)
      @test length(test_indices(r)) == 60 && length(train_indices(r)) == 240
      @test sort(vcat(train_indices(r), test_indices(r))) == 1:300
      @test r.converged && r.iterations >= 0 && r.selected === :test
      @test SPlit.isresolved(r.method.kernel)
    end
    r25 = datasplit(KernelThinningSplitter(ratio = 0.25, rng = MersenneTwister(52)), data)
    @test length(test_indices(r25)) == 75
  end

  @testset "ratio > 0.5 puts the selected rows in train; selectrows agrees" begin
    data = randn(MersenneTwister(53), 200, 2)
    r = datasplit(KernelThinningSplitter(ratio = 0.7, rng = MersenneTwister(54)), data)
    @test length(test_indices(r)) == 140 && r.selected === :train
    @test train_indices(r) == selectrows(
      KernelThinningSplitter(ratio = 0.7, rng = MersenneTwister(54)),
      data,
      60,
    )
    selected = selectrows(KernelThinningSplitter(rng = MersenneTwister(60)), data, 150)
    @test length(selected) == 150 && allunique(selected)   # n > N ÷ 2: the complement rule
  end

  @testset "ratio = 0.5 regression: N ≡ 3 (mod 4)" begin
    data203 = randn(MersenneTwister(62), 203, 2)
    r = datasplit(KernelThinningSplitter(ratio = 0.5, rng = MersenneTwister(61)), data203)
    @test sort(vcat(train_indices(r), test_indices(r))) == 1:203
    @test length(test_indices(r)) == 102
  end

  @testset "reproducible with rng; DataFrame and vector inputs; compare" begin
    data = randn(MersenneTwister(55), 240, 2)
    a = datasplit(KernelThinningSplitter(rng = MersenneTwister(1)), data)
    b = datasplit(KernelThinningSplitter(rng = MersenneTwister(1)), data)
    @test test_indices(a) == test_indices(b)
    df = DataFrame(x = randn(MersenneTwister(56), 90), g = repeat(["a", "b", "c"], 30))
    @test length(
      test_indices(datasplit(KernelThinningSplitter(rng = MersenneTwister(2)), df)),
    ) == 18
    @test length(
      test_indices(
        datasplit(
          KernelThinningSplitter(rng = MersenneTwister(3)),
          randn(MersenneTwister(57), 50),
        ),
      ),
    ) == 10
    c = compare(
      [
        KernelThinningSplitter(rng = MersenneTwister(4)),
        HerdingSplitter(kernel = EnergyKernel()),
      ],
      data,
    )
    @test DataFrame(c).method == ["KernelThinningSplitter", "HerdingSplitter"]
    @test all(isfinite, c.qualities)
  end

  @testset "weights and reference through datasplit and selectrows" begin
    data = randn(MersenneTwister(58), 300, 2)
    s = KernelThinningSplitter(rng = MersenneTwister(5))
    @test test_indices(datasplit(s, data; weights = ones(300))) ==
          test_indices(datasplit(KernelThinningSplitter(rng = MersenneTwister(5)), data))
    heavy = data[:, 1] .> 0
    plain = selectrows(KernelThinningSplitter(rng = MersenneTwister(6)), data, 60)
    weighted = selectrows(
      KernelThinningSplitter(rng = MersenneTwister(6)),
      data,
      60;
      weights = ifelse.(heavy, 20.0, 1.0),
    )
    targeted = selectrows(
      KernelThinningSplitter(rng = MersenneTwister(6)),
      data,
      60;
      reference = data[heavy, :],
    )
    @test count(heavy[weighted]) > count(heavy[plain])
    @test count(heavy[targeted]) > count(heavy[plain])
    folds = multiplet(KernelThinningSplitter(rng = MersenneTwister(7)), data, 4)
    @test sort(reduce(vcat, folds)) == 1:300
  end

  @testset "multiplet :halving partitions with balanced fold sizes" begin
    data301 = randn(MersenneTwister(71), 301, 2)
    folds = multiplet(
      KernelThinningSplitter(rng = MersenneTwister(70)),
      data301,
      4;
      strategy = :halving,
    )
    @test sort(reduce(vcat, folds)) == 1:301
    @test maximum(length.(folds)) - minimum(length.(folds)) <= 1
  end
end

@testset "Compress and Compress++" begin
  @testset "g and the cost rule" begin
    @test SPlit._compress_g(10_000, 500) == 4
    @test SPlit._compress_g(10_000, 2_000) == 6
    @test SPlit._compress_g(1_000_000, 10_000) == 5
    @test SPlit._compress_pays_off(10_000, 500)
    @test !SPlit._compress_pays_off(10_000, 2_000)
    @test !SPlit._compress_pays_off(1_000, 50)
    @test !SPlit._compress_pays_off(10_000, 2_000) &&
          !SPlit._compress_pays_off(100_000, 20_000)  # split ratios never
  end

  @testset "four parts and the halving count" begin
    @test SPlit._four_parts(collect(1:10)) == [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10]]
    @test SPlit._compress_halvings(256, 4) == 0
    @test SPlit._compress_halvings(1024, 4) == 1
    @test SPlit._compress_halvings(4096, 4) == 5
  end

  @testset "symmetrized halving returns half of the block in its order" begin
    X = SPlit.preprocess(randn(MersenneTwister(80), 400, 2))
    S = randperm(MersenneTwister(81), 400)[1:201]
    outs = [
      SPlit._symmetrized_halve(
        EnergyKernel(),
        X,
        S,
        0.1,
        MersenneTwister(s);
        n_threads = 2,
      ) for s = 1:8
    ]
    @test all(o -> length(o) == 100 && allunique(o) && all(in(S), o), outs)
    @test all(o -> issorted(indexin(o, S)), outs)                    # block order preserved
    @test length(unique(outs)) > 1                                    # both halves occur across seeds
  end

  @testset "Compress returns about 2^g √N rows of the input, deterministically" begin
    X = SPlit.preprocess(randn(MersenneTwister(82), 4096, 3))
    seq = randperm(MersenneTwister(83), 4096)
    S = SPlit._compress(EnergyKernel(), X, seq, 4, 1e-3, MersenneTwister(84); n_threads = 2)
    @test allunique(S) && all(in(seq), S)
    @test 512 <= length(S) <= 2048                                    # 2^4 √4096 = 1024
    @test S == SPlit._compress(
      EnergyKernel(),
      X,
      seq,
      4,
      1e-3,
      MersenneTwister(84);
      n_threads = 1,
    )
    @test SPlit._compress(EnergyKernel(), X, seq[1:200], 4, 1e-3, MersenneTwister(0)) ==
          seq[1:200]   # base case
  end

  @testset "Compress++ selects n distinct rows and beats random" begin
    mixture = let rng = MersenneTwister(85), N = 8_000
      c = rand(rng, 1:4, N)
      centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
      SPlit.preprocess(centers[c, :] .+ randn(rng, N, 2))
    end
    rows, swaps = SPlit._compress_plus_plus(
      EnergyKernel(),
      mixture,
      200;
      delta = 0.5,
      rng = MersenneTwister(86),
      n_threads = 2,
    )
    @test length(rows) == 200 && allunique(rows) && swaps >= 0
    q = energydistance(mixture[rows, :], mixture)
    random_q = mean(
      energydistance(mixture[randperm(MersenneTwister(300 + i), 8_000)[1:200], :], mixture) for i = 1:10
    )
    @test q < random_q
    @test rows == SPlit._compress_plus_plus(
      EnergyKernel(),
      mixture,
      200;
      delta = 0.5,
      rng = MersenneTwister(86),
      n_threads = 1,
    )[1]
  end
end

@testset "compress keyword" begin
  mixture = let rng = MersenneTwister(90), N = 8_000
    c = rand(rng, 1:4, N)
    centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
    centers[c, :] .+ randn(rng, N, 2)
  end
  X = SPlit.preprocess(mixture)

  @testset "kernel_thinning: :auto follows the cost rule, :always and :never are explicit" begin
    a = SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :auto,
      rng = MersenneTwister(1),
    )
    @test a == SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :always,
      rng = MersenneTwister(1),
    )
    @test a != SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :never,
      rng = MersenneTwister(1),
    )
    b = SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      1_600;
      compress = :auto,
      rng = MersenneTwister(2),
    )
    @test b == SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      1_600;
      compress = :never,
      rng = MersenneTwister(2),
    )
    @test b == SPlit.kernel_thinning(EnergyKernel(), X, 1_600; rng = MersenneTwister(2))   # default :never
    @test_throws ArgumentError SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :sometimes,
    )
    w = rand(MersenneTwister(3), 8_000)
    @test_throws ArgumentError SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :always,
      weights = w,
    )
    @test_throws ArgumentError SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :always,
      target = X[1:100, :],
    )
    @test SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :auto,
      weights = w,
      rng = MersenneTwister(4),
    ) == SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :never,
      weights = w,
      rng = MersenneTwister(4),
    )
    # the complement rule composes with compress
    hi, _ = SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      7_800;
      compress = :always,
      rng = MersenneTwister(5),
    )
    lo, _ = SPlit.kernel_thinning(
      EnergyKernel(),
      X,
      200;
      compress = :always,
      rng = MersenneTwister(5),
    )
    @test hi == sort(setdiff(1:8_000, lo))
  end

  @testset "KernelThinningSplitter: field, show, selectrows, datasplit unchanged" begin
    s = KernelThinningSplitter()
    @test s.compress === :auto
    @test KernelThinningSplitter(compress = :never).compress === :never
    @test_throws ArgumentError KernelThinningSplitter(compress = :maybe)
    @test occursin("compress=:auto", sprint(show, s))
    sel = selectrows(KernelThinningSplitter(rng = MersenneTwister(6)), mixture, 200)
    @test sel == selectrows(
      KernelThinningSplitter(compress = :always, rng = MersenneTwister(6)),
      mixture,
      200,
    )
    @test sel != selectrows(
      KernelThinningSplitter(compress = :never, rng = MersenneTwister(6)),
      mixture,
      200,
    )
    small = mixture[1:600, :]
    @test test_indices(
      datasplit(KernelThinningSplitter(rng = MersenneTwister(7)), small),
    ) == test_indices(
      datasplit(KernelThinningSplitter(compress = :never, rng = MersenneTwister(7)), small),
    )
    @test_throws ArgumentError selectrows(
      KernelThinningSplitter(compress = :always),
      mixture,
      200;
      weights = rand(8_000),
    )
    folds = multiplet(KernelThinningSplitter(rng = MersenneTwister(8)), small, 3)
    @test sort(reduce(vcat, folds)) == 1:600
  end
end
