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
