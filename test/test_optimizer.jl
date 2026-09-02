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

  @testset "does not stop at the initial sample on high-dimensional data" begin
    data = randn(MersenneTwister(140), 10_000, 10)
    k = SPlit.resolve(GaussianKernel(), data, MersenneTwister(141))
    pts, conv, iters =
      SPlit.support_points(k, data, 2_000; max_iterations = 2, rng = MersenneTwister(142))
    @test iters == 2
    @test conv == false
    @test all(pts .>= minimum(data; dims = 1)) && all(pts .<= maximum(data; dims = 1))
    traj =
      SPlit._mmd_trajectory(k, data, 2_000; max_iterations = 3, rng = MersenneTwister(142))
    @test traj[end] < traj[1]
  end

  @testset "relative-decrease rule stops a flat objective honestly" begin
    data = randn(MersenneTwister(143), 200, 2)
    _, conv, iters = SPlit.support_points(
      GaussianKernel(1.0),
      data,
      20;
      max_iterations = 300,
      rtol = 1e-3,
      rng = MersenneTwister(144),
    )
    @test conv && 2 <= iters < 300
  end
end

@testset "weighted support points (energy kernel)" begin
  @testset "nothing and uniform weights give identical points" begin
    data = randn(MersenneTwister(70), 150, 2)
    a, ca, ia = SPlit.support_points(
      EnergyKernel(),
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(1),
    )
    b, cb, ib = SPlit.support_points(
      EnergyKernel(),
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(1),
      weights = ones(150),
    )
    c, cc, ic = SPlit.support_points(
      EnergyKernel(),
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(1),
      weights = fill(0.37, 150),
    )
    @test a == b == c
    @test (ca, ia) == (cb, ib) == (cc, ic)
    # stochastic mode too, both rules, for uniform weights
    d, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      15;
      kappa = 60,
      max_iterations = 30,
      rng = MersenneTwister(2),
    )
    e, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      15;
      kappa = 60,
      max_iterations = 30,
      rng = MersenneTwister(2),
      weights = ones(150),
    )
    @test d == e
  end

  @testset "one weighted sweep equals one sweep on duplicated rows" begin
    rng = MersenneTwister(71)
    data = randn(rng, 40, 2)
    counts = rand(rng, 1:3, 40)
    dup = vcat([data[i:i, :] for i = 1:40 for _ = 1:counts[i]]...)
    n = 6
    points = data[1:n, :] .+ 0.05
    bounds_w = SPlit._data_bounds(data)
    bounds_d = SPlit._data_bounds(dup)
    new_w = similar(points)
    new_d = similar(points)
    cw = zeros(n)
    cd = zeros(n)
    SPlit._mm_sweep!(
      new_w,
      cw,
      copy(points),
      data,
      SPlit._mean_one_weights(Float64.(counts)),
      zeros(n),
      1.0,
      bounds_w,
      1,
    )
    SPlit._mm_sweep!(
      new_d,
      cd,
      copy(points),
      dup,
      ones(size(dup, 1)),
      zeros(n),
      1.0,
      bounds_d,
      1,
    )
    @test isapprox(new_w, new_d; atol = 1e-10)
  end

  @testset "weighted full-data MM monotonically decreases the weighted objective" begin
    rng = MersenneTwister(72)
    data = randn(rng, 150, 2)
    w = rand(rng, 150) .^ 3
    traj = SPlit._objective_trajectory(
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(3),
      weights = w,
    )
    @test length(traj) >= 2
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-8
    end
  end

  @testset "concentrated weights pull support points toward the heavy cluster" begin
    rng = MersenneTwister(73)
    A = randn(rng, 200, 2) .- 4.0
    B = randn(rng, 200, 2) .+ 4.0
    data = vcat(A, B)
    w = vcat(fill(9.0, 200), fill(1.0, 200))
    in_A(pts) = count(<(0.0), pts[:, 1])
    unweighted, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      40;
      max_iterations = 100,
      rng = MersenneTwister(4),
    )
    weighted, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      40;
      max_iterations = 100,
      rng = MersenneTwister(4),
      weights = w,
    )
    @test in_A(weighted) > in_A(unweighted)
    @test in_A(weighted) >= 30
    for rule in (:uniform, :proportional)
      stoch, _, _ = SPlit.support_points(
        EnergyKernel(),
        data,
        40;
        kappa = 120,
        max_iterations = 100,
        rng = MersenneTwister(5),
        weights = w,
        _subsampling = rule,
      )
      @test in_A(stoch) >= 28
    end
  end

  @testset "validation" begin
    data = randn(MersenneTwister(74), 50, 2)
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      weights = ones(49),
    )
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      weights = ones(50),
      _subsampling = :other,
    )
  end
end
