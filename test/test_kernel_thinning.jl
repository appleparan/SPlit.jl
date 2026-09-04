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

  @testset "kernel_thinning: sizes, validation, reproducibility" begin
    rows, swaps = SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(30))
    @test length(rows) == 96 && allunique(rows)
    @test (rows, swaps) ==
          SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(30))
    @test rows != SPlit.kernel_thinning(EnergyKernel(), X, 96; rng = MersenneTwister(31))[1]
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 241)   # > N ÷ 2
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 0)
    @test_throws ArgumentError SPlit.kernel_thinning(EnergyKernel(), X, 96; delta = 0.0)
    @test_throws ArgumentError SPlit.kernel_thinning(GaussianKernel(), X, 96)   # unresolved kernel
    # ratio 0.25 uses every row in KT-SPLIT (L = N); 0.2 uses L = 0.8 N
    @test length(
      SPlit.kernel_thinning(EnergyKernel(), X, 120; rng = MersenneTwister(32))[1],
    ) == 120
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
