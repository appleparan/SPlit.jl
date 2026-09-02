using Test
using SPlit
using Random
using Statistics
using DataFrames
using LinearAlgebra

# Brute-force greedy step from the spec: Δ(x) for appending x to the T selected rows.
function delta(kernel, X, selected, i)
  N = size(X, 1)
  T = length(selected)
  kxx = SPlit.kernelvalue(kernel, X[i, :], X[i, :])
  ksel = sum((SPlit.kernelvalue(kernel, X[i, :], X[s, :]) for s in selected); init = 0.0)
  kdat = sum(SPlit.kernelvalue(kernel, X[i, :], X[l, :]) for l = 1:N)
  return kxx / (T + 1)^2 + 2 * ksel / (T + 1)^2 - 2 * kdat / ((T + 1) * N)
end

function brute_force_herd(kernel, X, n)
  N = size(X, 1)
  selected = Int[]
  for _ = 1:n
    best = 0
    bestval = Inf
    for i = 1:N
      i in selected && continue
      v = delta(kernel, X, selected, i)
      if v < bestval
        bestval = v
        best = i
      end
    end
    push!(selected, best)
  end
  return selected
end

@testset "herd" begin
  X = randn(MersenneTwister(100), 80, 2)

  @testset "each selection is the greedy MMD² step (both kernels)" begin
    for k in (GaussianKernel(1.0), EnergyKernel())
      @test SPlit.herd(k, X, 12) == brute_force_herd(k, X, 12)
    end
  end

  @testset "discrepancy to the data decreases along the sequence" begin
    for k in (GaussianKernel(1.0), EnergyKernel())
      sel = SPlit.herd(k, X, 40)
      q(T) = mmd(X[sel[1:T], :], X, k)
      @test q(40) < q(20) < q(10)
      rng = MersenneTwister(101)
      random_q = mean(mmd(X[randperm(rng, 80)[1:40], :], X, k) for _ = 1:20)
      @test q(40) < random_q
    end
  end

  @testset "deterministic for a numeric kernel; kappa path reproducible" begin
    k = GaussianKernel(1.0)
    a = SPlit.herd(k, X, 15; rng = MersenneTwister(1), n_threads = 1)
    b = SPlit.herd(k, X, 15; rng = MersenneTwister(2), n_threads = 4)
    @test a == b
    c1 = SPlit.herd(k, X, 15; kappa = 30, rng = MersenneTwister(5))
    c2 = SPlit.herd(k, X, 15; kappa = 30, rng = MersenneTwister(5))
    @test c1 == c2
    @test length(unique(c1)) == 15
    # kappa ≥ N is the exact path
    @test SPlit.herd(k, X, 15; kappa = 80, rng = MersenneTwister(1)) == a
  end

  @testset "validation" begin
    @test_throws ArgumentError SPlit.herd(GaussianKernel(1.0), X, 0)
    @test_throws ArgumentError SPlit.herd(GaussianKernel(1.0), X, 81)
    @test_throws ArgumentError SPlit.herd(GaussianKernel(), X, 5)      # unresolved
    @test_throws ArgumentError SPlit.herd(GaussianKernel(1.0), X, 5; kappa = 0)
  end
end

@testset "HerdingSplitter" begin
  @testset "construction and validation" begin
    s = HerdingSplitter()
    @test s isa AbstractSplitter
    @test s.kernel == GaussianKernel() && s.ratio == 0.2 && s.kappa === nothing
    @test_throws ArgumentError HerdingSplitter(ratio = 0)
    @test_throws ArgumentError HerdingSplitter(ratio = 1)
    @test_throws ArgumentError HerdingSplitter(kappa = 0)
    @test_throws ArgumentError HerdingSplitter(n_threads = 0)
    @test HerdingSplitter(ratio = 1 // 4).ratio == 0.25
  end

  @testset "datasplit: partition, honest report, resolved kernel stored" begin
    data = randn(MersenneTwister(102), 200, 3)
    s = HerdingSplitter(rng = MersenneTwister(103))
    r = datasplit(s, data)
    @test r isa SplitResult{<:HerdingSplitter}
    @test length(test_indices(r)) == 40
    @test sort(vcat(train_indices(r), test_indices(r))) == collect(1:200)
    @test r.converged && r.iterations == 40
    @test r.method.kernel isa GaussianKernel{Float64}
    @test s.kernel.bandwidth === :median
    train, test = r
    @test train == train_indices(r)
    @test size(data[r, :test], 1) == 40
  end

  @testset "ratio > 0.5 puts the selected rows in train" begin
    data = randn(MersenneTwister(104), 100, 2)
    r = datasplit(HerdingSplitter(kernel = EnergyKernel(), ratio = 0.7), data)
    @test length(train_indices(r)) == 30
    @test length(test_indices(r)) == 70
  end

  @testset "DataFrame and vector inputs; EnergyKernel" begin
    df = DataFrame(x = randn(MersenneTwister(105), 90), g = repeat(["a", "b", "c"], 30))
    r = datasplit(HerdingSplitter(kernel = EnergyKernel()), df)
    @test length(test_indices(r)) == 18
    v = randn(MersenneTwister(106), 50)
    r2 = datasplit(HerdingSplitter(kernel = GaussianKernel(0.5)), v)
    @test length(test_indices(r2)) == 10
  end
end
