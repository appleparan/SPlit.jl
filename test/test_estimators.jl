using Test
using SPlit
using Random
using Statistics
using LinearAlgebra

@testset "estimator types" begin
  @test Exact() isa DiscrepancyEstimator
  @test Subsample(100).repeats == 8
  @test_throws ArgumentError Subsample(1)
  @test_throws ArgumentError Subsample(10, 0)
  @test_throws ArgumentError RandomSlices(0)
  @test_throws ArgumentError RandomFeatures(0)
  @test RandomSlices(16).k == 16 && RandomFeatures(32).D == 32
end

@testset "sphere constant κ_p" begin
  @test SPlit.sphere_constant(1) ≈ 1.0
  @test SPlit.sphere_constant(2) ≈ 2 / π
  @test SPlit.sphere_constant(3) ≈ 0.5
  @test SPlit.sphere_constant(4) ≈ 4 / (3π)
  # Monte-Carlo check of E|⟨θ,u⟩| = κ_p ‖u‖ for p ∈ {2, 3, 10}
  for p in (2, 3, 10)
    rng = MersenneTwister(p)
    u = randn(rng, p)
    Θ = SPlit._project_directions(rng, p, 200_000)
    est = mean(abs.(Θ' * u))
    @test isapprox(est, SPlit.sphere_constant(p) * norm(u); rtol = 0.02)
  end
end

@testset "1-D energy distance from sorted formulas" begin
  rng = MersenneTwister(5)
  a = randn(rng, 40)
  b = randn(rng, 25) .+ 0.3
  naive =
    2 * mean(abs(x - y) for x in a, y in b) - mean(abs(x - y) for x in a, y in a) -
    mean(abs(x - y) for x in b, y in b)
  @test isapprox(SPlit._ed1d(a, b), naive; atol = 1e-12)
  @test isapprox(SPlit._ed1d(a, b), energydistance(a, b); atol = 1e-12)
  @test isapprox(SPlit._ed1d(a, copy(a)), 0.0; atol = 1e-12)
end

@testset "sliced energy distance converges to exact" begin
  rng = MersenneTwister(6)
  X = randn(rng, 300, 3)
  Y = randn(rng, 200, 3) .+ 0.5
  exact = energydistance(X, Y)
  ks = (16, 64, 256)
  mean_err = map(ks) do k
    mean(
      abs(
        energydistance(
          X,
          Y;
          estimator = RandomSlices(k),
          rng = MersenneTwister(1_000 + t),
        ) - exact,
      ) for t = 1:20
    )
  end
  @test mean_err[3] < 0.5 * mean_err[1]
  @test mean_err[3] < 0.1 * exact
  @test energydistance(X, Y; estimator = RandomSlices(64), rng = MersenneTwister(1)) ==
        energydistance(X, Y; estimator = RandomSlices(64), rng = MersenneTwister(1))
  # p == 1 is exact for any k
  x1 = randn(MersenneTwister(8), 50, 1)
  y1 = randn(MersenneTwister(9), 40, 1) .+ 1
  @test isapprox(
    energydistance(x1, y1; estimator = RandomSlices(3)),
    energydistance(x1, y1);
    atol = 1e-10,
  )
end

@testset "estimator keyword and compatibility keywords" begin
  rng = MersenneTwister(10)
  X = randn(rng, 120, 2)
  Y = randn(rng, 80, 2)
  @test energydistance(X, Y; estimator = Exact()) == energydistance(X, Y)
  @test energydistance(X, Y; subsample = 50, repeats = 4, rng = MersenneTwister(2)) ==
        energydistance(X, Y; estimator = Subsample(50, 4), rng = MersenneTwister(2))
  @test_throws ArgumentError energydistance(X, Y; estimator = RandomFeatures(16))
end

@testset "threaded exact means are bit-identical across n_threads" begin
  rng = MersenneTwister(11)
  X = randn(rng, 2_500, 3)
  Y = randn(rng, 1_700, 3)
  @test energydistance(X, Y; n_threads = 1) == energydistance(X, Y; n_threads = 4)
  @test SPlit._mean_pairwise(X, Y; block = 300, n_threads = 1) ==
        SPlit._mean_pairwise(X, Y; block = 300, n_threads = 3)
  k = GaussianKernel(1.0)
  @test SPlit._mean_kernel(k, X, Y; block = 300, n_threads = 1) ==
        SPlit._mean_kernel(k, X, Y; block = 300, n_threads = 3)
end

@testset "random Fourier features" begin
  k = GaussianKernel(1.3)
  rng = MersenneTwister(20)
  x = randn(rng, 4)
  y = randn(rng, 4)
  errs = Float64[]
  for D in (64, 1024, 16_384)
    φ = SPlit.FourierFeatureMap(k, 4, D, MersenneTwister(21))
    push!(errs, abs(dot(φ(x), φ(y)) - SPlit.kernelvalue(k, x, y)))
  end
  @test errs[3] < errs[1]
  @test errs[3] < 0.05
  X = randn(rng, 300, 4)
  Y = randn(rng, 200, 4) .+ 0.4
  exact = mmd(X, Y, k)
  est = mmd(X, Y, k; estimator = RandomFeatures(4096), rng = MersenneTwister(22))
  @test isapprox(est, exact; rtol = 0.25)
  @test mmd(X, Y, k; estimator = RandomFeatures(256), rng = MersenneTwister(1)) ==
        mmd(X, Y, k; estimator = RandomFeatures(256), rng = MersenneTwister(1))
  @test_throws ArgumentError mmd(X, Y, k; estimator = RandomSlices(8))
  @test mmd(X, Y, EnergyKernel(); estimator = RandomSlices(32), rng = MersenneTwister(3)) ==
        energydistance(X, Y; estimator = RandomSlices(32), rng = MersenneTwister(3))
  @test_throws ArgumentError mmd(X, Y, EnergyKernel(); estimator = RandomFeatures(8))
end
