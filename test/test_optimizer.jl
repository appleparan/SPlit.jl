using Test
using SPlit
using Random
using LinearAlgebra

@testset "kernels" begin
  @test EnergyKernel() isa SplitKernel
end

@testset "support_points" begin
  @testset "shape, bounds, and convergence report" begin
    rng = MersenneTwister(3)
    data = randn(rng, 200, 2)
    pts, converged, iters = SPlit.support_points(
      EnergyKernel(),
      data,
      20;
      max_iterations = 300,
      rng = MersenneTwister(1),
    )
    @test size(pts) == (20, 2)
    @test 1 <= iters <= 300
    @test converged isa Bool
    for j = 1:2
      lo, hi = extrema(view(data, :, j))
      @test all(lo .<= pts[:, j] .<= hi)
    end
    # a tight tolerance run that cannot converge must say so honestly
    _, conv2, iters2 = SPlit.support_points(
      EnergyKernel(),
      data,
      20;
      max_iterations = 3,
      tolerance = 1e-30,
      rng = MersenneTwister(1),
    )
    @test conv2 == false
    @test iters2 == 3
  end

  @testset "full-data MM monotonically decreases the energy objective" begin
    rng = MersenneTwister(5)
    data = randn(rng, 150, 2)
    traj =
      SPlit._objective_trajectory(data, 15; max_iterations = 40, rng = MersenneTwister(2))
    @test length(traj) >= 2
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-8
    end
  end

  @testset "reproducibility: same rng ⇒ same points, regardless of threads" begin
    data = randn(MersenneTwister(9), 120, 3)
    a, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      12;
      max_iterations = 50,
      rng = MersenneTwister(7),
      n_threads = 1,
    )
    b, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      12;
      max_iterations = 50,
      rng = MersenneTwister(7),
      n_threads = 4,
    )
    @test a == b
  end

  @testset "stochastic mode runs and respects rng" begin
    data = randn(MersenneTwister(13), 500, 2)
    a, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      25;
      kappa = 100,
      max_iterations = 60,
      rng = MersenneTwister(21),
    )
    b, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      25;
      kappa = 100,
      max_iterations = 60,
      rng = MersenneTwister(21),
    )
    @test a == b
    @test size(a) == (25, 2)
  end

  @testset "argument validation" begin
    data = randn(MersenneTwister(1), 30, 2)
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 0)
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 31)
    @test_throws ArgumentError SPlit.support_points(EnergyKernel(), data, 5; kappa = 0)
  end
end

@testset "support_points with GaussianKernel" begin
  k = GaussianKernel(1.0)

  @testset "gradient matches finite differences" begin
    data = randn(MersenneTwister(60), 40, 2)
    points = randn(MersenneTwister(61), 6, 2)
    G = zeros(6, 2)
    SPlit._mmd_gradient!(G, k, points, data, 1)
    h = 1e-6
    for m = 1:6, j = 1:2
      plus = copy(points)
      plus[m, j] += h
      minus = copy(points)
      minus[m, j] -= h
      fd =
        (SPlit._mmd_objective(k, plus, data) - SPlit._mmd_objective(k, minus, data)) / (2h)
      @test isapprox(G[m, j], fd; rtol = 1e-5, atol = 1e-9)
    end
    # threaded gradient equals serial gradient
    G4 = zeros(6, 2)
    SPlit._mmd_gradient!(G4, k, points, data, 4)
    @test G4 == G
  end

  @testset "objective is non-increasing along accepted steps" begin
    data = randn(MersenneTwister(62), 150, 2)
    traj =
      SPlit._mmd_trajectory(k, data, 15; max_iterations = 40, rng = MersenneTwister(63))
    @test length(traj) >= 2
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-10
    end
  end

  @testset "shape, bounds, honest convergence, reproducibility" begin
    data = randn(MersenneTwister(64), 200, 3)
    pts, conv, iters =
      SPlit.support_points(k, data, 20; max_iterations = 200, rng = MersenneTwister(65))
    @test size(pts) == (20, 3)
    @test conv isa Bool && 1 <= iters <= 200
    for j = 1:3
      lo, hi = extrema(view(data, :, j))
      @test all(lo .<= pts[:, j] .<= hi)
    end
    _, conv2, iters2 = SPlit.support_points(
      k,
      data,
      20;
      max_iterations = 2,
      tolerance = 1e-30,
      rng = MersenneTwister(65),
    )
    @test conv2 == false && iters2 == 2
    a, _, _ = SPlit.support_points(
      k,
      data,
      12;
      max_iterations = 30,
      rng = MersenneTwister(7),
      n_threads = 1,
    )
    b, _, _ = SPlit.support_points(
      k,
      data,
      12;
      max_iterations = 30,
      rng = MersenneTwister(7),
      n_threads = 4,
    )
    @test a == b
  end

  @testset "argument validation" begin
    data = randn(MersenneTwister(66), 30, 2)
    @test_throws ArgumentError SPlit.support_points(k, data, 5; kappa = 10)
    @test_throws ArgumentError SPlit.support_points(GaussianKernel(), data, 5)
    @test_throws ArgumentError SPlit.support_points(k, data, 0)
  end
end
