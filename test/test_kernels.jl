using Test
using SPlit
using Random
using Statistics
using LinearAlgebra

@testset "GaussianKernel" begin
  @testset "construction and validation" begin
    @test GaussianKernel() isa SplitKernel
    @test GaussianKernel().bandwidth === :median
    @test GaussianKernel(2).bandwidth === 2.0
    @test_throws ArgumentError GaussianKernel(0.0)
    @test_throws ArgumentError GaussianKernel(-1)
    @test_throws ArgumentError GaussianKernel(Inf)
    @test_throws ArgumentError GaussianKernel(NaN)
    @test_throws ArgumentError GaussianKernel(:mean)
  end

  @testset "kernelvalue and kernelgrad! match the formula" begin
    k = GaussianKernel(1.5)
    u = [0.3, -1.0, 2.0]
    v = [1.0, 0.5, 1.5]
    d2 = sum(abs2, u .- v)
    @test SPlit.kernelvalue(k, u, v) ≈ exp(-d2 / (2 * 1.5^2))
    @test SPlit.kernelvalue(k, u, u) ≈ 1.0
    g = zeros(3)
    SPlit.kernelgrad!(g, k, u, v)
    @test g ≈ -SPlit.kernelvalue(k, u, v) .* (u .- v) ./ 1.5^2
    # finite-difference check of the gradient
    h = 1e-6
    for j = 1:3
      e = zeros(3)
      e[j] = h
      fd = (SPlit.kernelvalue(k, u .+ e, v) - SPlit.kernelvalue(k, u .- e, v)) / (2h)
      @test isapprox(g[j], fd; rtol = 1e-5)
    end
  end

  @testset "resolve: median heuristic is reproducible and sane" begin
    data = randn(MersenneTwister(3), 2_500, 2)
    a = SPlit.resolve(GaussianKernel(), data, MersenneTwister(11))
    b = SPlit.resolve(GaussianKernel(), data, MersenneTwister(11))
    @test a isa GaussianKernel{Float64}
    @test a.bandwidth == b.bandwidth
    # median pairwise distance of 2-D standard normal data is ≈ 1.7
    @test 1.2 < a.bandwidth < 2.3
    # small data uses every row: exact median of all pairwise distances
    small = randn(MersenneTwister(4), 30, 2)
    r = SPlit.resolve(GaussianKernel(), small, MersenneTwister(1))
    dists = [norm(small[i, :] .- small[j, :]) for i = 1:30 for j = (i+1):30]
    @test r.bandwidth ≈ median(dists)
    # numeric bandwidth passes through; EnergyKernel resolves to itself
    @test SPlit.resolve(GaussianKernel(0.7), data, MersenneTwister(1)).bandwidth == 0.7
    @test SPlit.resolve(EnergyKernel(), data, MersenneTwister(1)) === EnergyKernel()
    @test SPlit.isresolved(EnergyKernel())
    @test SPlit.isresolved(GaussianKernel(1.0))
    @test !SPlit.isresolved(GaussianKernel())
  end

  @testset "resolve rejects degenerate data" begin
    same = ones(50, 2)
    @test_throws ArgumentError SPlit.resolve(GaussianKernel(), same, MersenneTwister(1))
  end
end

@testset "EnergyKernel kernelvalue" begin
  u = [0.0, 3.0]
  v = [4.0, 0.0]
  @test SPlit.kernelvalue(EnergyKernel(), u, v) == -5.0
  @test SPlit.kernelvalue(EnergyKernel(), u, u) == 0.0
end

@testset "weighted median bandwidth" begin
  X = randn(MersenneTwister(40), 300, 2)
  @test SPlit.resolve(GaussianKernel(), X, MersenneTwister(1), nothing) ==
        SPlit.resolve(GaussianKernel(), X, MersenneTwister(1))
  @test SPlit.resolve(EnergyKernel(), X, MersenneTwister(1), ones(300)) == EnergyKernel()
  @test SPlit.resolve(GaussianKernel(2.0), X, MersenneTwister(1), ones(300)) ==
        GaussianKernel(2.0)

  # Two clusters far apart, more rows than the 1_000 the heuristic draws:
  # weight concentrated on one cluster makes most drawn pairs intra-cluster,
  # so the median distance drops.
  Y = vcat(randn(MersenneTwister(41), 750, 2), randn(MersenneTwister(42), 750, 2) .+ 20.0)
  w = vcat(fill(100.0, 750), fill(1e-3, 750))
  σ_uniform = SPlit.resolve(GaussianKernel(), Y, MersenneTwister(3)).bandwidth
  σ_weighted = SPlit.resolve(GaussianKernel(), Y, MersenneTwister(3), w).bandwidth
  @test σ_weighted < σ_uniform
  @test_throws ArgumentError SPlit.resolve(
    GaussianKernel(),
    Y,
    MersenneTwister(3),
    ones(10),
  )
end
