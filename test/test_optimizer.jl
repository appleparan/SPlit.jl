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
    SPlit._mmd_gradient!(G, k, points, data, ones(40), 1)
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
    SPlit._mmd_gradient!(G4, k, points, data, ones(40), 4)
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

  @testset "all-zero kappa subsample errors" begin
    data = randn(MersenneTwister(90), 60, 2)
    w = zeros(60)
    w[1] = 1.0
    # 59 zero-weight rows and kappa = 10: each draw has probability ≈ 0.83 of
    # drawing only zero-weight rows, so at least one of these seeds must throw.
    @test any(1:20) do seed
      try
        SPlit.support_points(
          EnergyKernel(),
          data,
          5;
          kappa = 10,
          weights = w,
          rng = MersenneTwister(seed),
          max_iterations = 3,
        )
        false
      catch e
        e isa ArgumentError
      end
    end
  end

  @testset ":proportional errors when fewer than kappa rows have positive weight" begin
    data = randn(MersenneTwister(95), 60, 2)
    w = zeros(60)
    w[1:3] .= 1.0
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      kappa = 10,
      weights = w,
      rng = MersenneTwister(1),
      max_iterations = 3,
      _subsampling = :proportional,
    )
  end
end

@testset "weighted support points (Gaussian kernel)" begin
  k = GaussianKernel(1.0)

  @testset "nothing and uniform weights give identical points" begin
    data = randn(MersenneTwister(80), 120, 2)
    a, ca, ia =
      SPlit.support_points(k, data, 12; max_iterations = 40, rng = MersenneTwister(1))
    b, cb, ib = SPlit.support_points(
      k,
      data,
      12;
      max_iterations = 40,
      rng = MersenneTwister(1),
      weights = ones(120),
    )
    @test a == b
    @test (ca, ia) == (cb, ib)
    c, cc, ic = SPlit.support_points(
      k,
      data,
      12;
      max_iterations = 40,
      rng = MersenneTwister(1),
      weights = fill(0.37, 120),
    )
    @test a == c
    @test (ca, ia) == (cc, ic)
  end

  @testset "weighted gradient matches finite differences of the weighted objective" begin
    rng = MersenneTwister(81)
    data = randn(rng, 30, 2)
    w = rand(rng, 30)
    w_bar = w ./ sum(w)
    w_hat = w .* (30 / sum(w))
    points = randn(rng, 5, 2)
    G = similar(points)
    SPlit._mmd_gradient!(G, k, points, data, w_hat, 1)
    h = 1e-6
    for m = 1:5, j = 1:2
      plus = copy(points)
      plus[m, j] += h
      minus = copy(points)
      minus[m, j] -= h
      fd =
        (
          SPlit._mmd_objective(k, plus, data, w_bar) -
          SPlit._mmd_objective(k, minus, data, w_bar)
        ) / (2h)
      @test isapprox(G[m, j], fd; atol = 1e-6)
    end
  end

  @testset "weighted objective never increases across accepted steps" begin
    rng = MersenneTwister(82)
    data = randn(rng, 120, 2)
    w = rand(rng, 120) .^ 2
    traj = SPlit._mmd_trajectory(
      k,
      data,
      12;
      max_iterations = 40,
      rng = MersenneTwister(2),
      weights = w,
    )
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-12
    end
  end

  @testset "concentrated weights pull support points toward the heavy cluster" begin
    rng = MersenneTwister(83)
    A = randn(rng, 200, 2) .- 4.0
    B = randn(rng, 200, 2) .+ 4.0
    data = vcat(A, B)
    w = vcat(fill(9.0, 200), fill(1.0, 200))
    in_A(pts) = count(<(0.0), pts[:, 1])
    unweighted, _, _ =
      SPlit.support_points(k, data, 40; max_iterations = 100, rng = MersenneTwister(4))
    weighted, _, _ = SPlit.support_points(
      k,
      data,
      40;
      max_iterations = 100,
      rng = MersenneTwister(4),
      weights = w,
    )
    @test in_A(weighted) > in_A(unweighted)
  end

  @testset "validation" begin
    data = randn(MersenneTwister(84), 50, 2)
    @test_throws ArgumentError SPlit.support_points(k, data, 5; weights = ones(49))
  end

  @testset "threaded weighted gradient equals serial" begin
    rng = MersenneTwister(85)
    data = randn(rng, 40, 2)
    points = randn(rng, 6, 2)
    w_hat = SPlit._mean_one_weights(rand(MersenneTwister(85), 40))
    G1 = similar(points)
    G4 = similar(points)
    SPlit._mmd_gradient!(G1, k, points, data, w_hat, 1)
    SPlit._mmd_gradient!(G4, k, points, data, w_hat, 4)
    @test G1 == G4
  end
end

@testset "support points toward a target measure" begin
  rng = MersenneTwister(300)
  data = randn(rng, 200, 2)
  R = data[data[:, 1].>0, :]          # a sub-population as the target

  @testset "target = data reproduces the untargeted run exactly" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      a, ca, ia = SPlit.support_points(
        kernel,
        data,
        20;
        max_iterations = 40,
        rng = MersenneTwister(1),
      )
      b, cb, ib = SPlit.support_points(
        kernel,
        data,
        20;
        max_iterations = 40,
        rng = MersenneTwister(1),
        target = data,
      )
      @test a == b
      @test (ca, ia) == (cb, ib)
    end
  end

  @testset "points move toward the target for both kernels" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      plain, _, _ = SPlit.support_points(
        kernel,
        data,
        30;
        max_iterations = 100,
        rng = MersenneTwister(2),
      )
      targeted, _, _ = SPlit.support_points(
        kernel,
        data,
        30;
        max_iterations = 100,
        rng = MersenneTwister(2),
        target = R,
      )
      @test count(>(0.0), targeted[:, 1]) > count(>(0.0), plain[:, 1])
      @test count(>(0.0), targeted[:, 1]) >= 24
      # points stay inside the candidates' bounding box
      for j = 1:2
        lo, hi = extrema(view(data, :, j))
        @test all(lo .<= targeted[:, j] .<= hi)
      end
    end
  end

  @testset "target weights as duplication counts: one MM sweep on duplicated target rows" begin
    Rsmall = R[1:30, :]
    counts = rand(MersenneTwister(301), 1:3, 30)
    Rdup = vcat([Rsmall[i:i, :] for i = 1:30 for _ = 1:counts[i]]...)
    n = 5
    points = data[1:n, :] .+ 0.05
    bounds = SPlit._data_bounds(data)
    new_w = similar(points)
    new_d = similar(points)
    SPlit._mm_sweep!(
      new_w,
      zeros(n),
      copy(points),
      Rsmall,
      SPlit._mean_one_weights(Float64.(counts)),
      zeros(n),
      1.0,
      bounds,
      1,
    )
    SPlit._mm_sweep!(
      new_d,
      zeros(n),
      copy(points),
      Rdup,
      ones(size(Rdup, 1)),
      zeros(n),
      1.0,
      bounds,
      1,
    )
    @test isapprox(new_w, new_d; atol = 1e-10)
  end

  @testset "monotone descent toward the target (energy) and non-increase (Gaussian)" begin
    traj = SPlit._objective_trajectory(
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(3),
      target = R,
    )
    for t = 2:length(traj)
      @test traj[t] <= traj[t-1] + 1e-8
    end
    v = rand(MersenneTwister(302), size(R, 1))
    trajw = SPlit._objective_trajectory(
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(3),
      target = R,
      target_weights = v,
    )
    for t = 2:length(trajw)
      @test trajw[t] <= trajw[t-1] + 1e-8
    end
    trajg = SPlit._mmd_trajectory(
      GaussianKernel(1.0),
      data,
      15;
      max_iterations = 40,
      rng = MersenneTwister(4),
      target = R,
      target_weights = v,
    )
    for t = 2:length(trajg)
      @test trajg[t] <= trajg[t-1] + 1e-12
    end
  end

  @testset "stochastic mode subsamples the target" begin
    big = randn(MersenneTwister(303), 600, 2)
    Rbig = big[big[:, 1].>0, :]
    pts, _, _ = SPlit.support_points(
      EnergyKernel(),
      big,
      40;
      kappa = 100,
      max_iterations = 60,
      rng = MersenneTwister(5),
      target = Rbig,
    )
    @test count(>(0.0), pts[:, 1]) >= 30
  end

  @testset "kappa at or above the target size runs in full-target mode" begin
    big = randn(MersenneTwister(304), 600, 2)
    Rbig = big[big[:, 1].>0, :]
    M = size(Rbig, 1)
    full, _, _ = SPlit.support_points(
      EnergyKernel(),
      big,
      20;
      kappa = M + 10,
      max_iterations = 5,
      rng = MersenneTwister(1),
      target = Rbig,
    )
    @test size(full) == (20, 2)
    plain, _, _ = SPlit.support_points(
      EnergyKernel(),
      big,
      20;
      max_iterations = 5,
      rng = MersenneTwister(1),
      target = Rbig,
    )
    @test full == plain
  end

  @testset "target_weights through support_points" begin
    v = rand(MersenneTwister(305), size(R, 1))
    weighted, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      20;
      max_iterations = 30,
      rng = MersenneTwister(2),
      target = R,
      target_weights = v,
    )
    @test size(weighted) == (20, 2)

    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      constant, _, _ = SPlit.support_points(
        kernel,
        data,
        20;
        max_iterations = 30,
        rng = MersenneTwister(2),
        target = R,
        target_weights = fill(0.3, size(R, 1)),
      )
      uniform, _, _ = SPlit.support_points(
        kernel,
        data,
        20;
        max_iterations = 30,
        rng = MersenneTwister(2),
        target = R,
      )
      @test constant == uniform
    end
  end

  @testset "target with duplicate rows runs (jitter branch)" begin
    Rsmall = randn(MersenneTwister(306), 10, 2)
    Rdup = repeat(Rsmall, 3, 1)   # 30 rows, each 3 times
    pts, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      10;
      max_iterations = 20,
      rng = MersenneTwister(1),
      target = Rdup,
    )
    @test size(pts) == (10, 2)
    @test all(isfinite, pts)
    pts_g, _, _ = SPlit.support_points(
      GaussianKernel(1.0),
      data,
      10;
      max_iterations = 20,
      rng = MersenneTwister(1),
      target = Rdup,
    )
    @test size(pts_g) == (10, 2)
    @test all(isfinite, pts_g)

    # Duplicating the target 3x is the same measure as weighting it 3x, so the
    # two runs should be equivalent up to the jitter (which perturbs the
    # duplicated target by up to 1e-3 of its range) and the different rng
    # consumption (the jitter draw) between the two code paths. The point
    # sets themselves land in different local configurations of comparable
    # quality (observed max abs diff ~2.7, well past 1e-2), so `isapprox` on
    # the raw points is not robust; comparing the energy distance to the
    # underlying target is (observed diff ~3e-3), so use that instead.
    weighted, _, _ = SPlit.support_points(
      EnergyKernel(),
      data,
      10;
      max_iterations = 20,
      rng = MersenneTwister(1),
      target = Rsmall,
      target_weights = fill(3.0, 10),
    )
    @test abs(energydistance(pts, Rsmall) - energydistance(weighted, Rsmall)) < 1e-2
  end

  @testset "validation" begin
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      target = zeros(0, 2),
    )
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      target = R,
      weights = ones(200),
    )
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      target_weights = ones(200),
    )
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      target = R,
      target_weights = ones(3),
    )
    @test_throws ArgumentError SPlit.support_points(
      EnergyKernel(),
      data,
      5;
      target = randn(10, 3),
    )
    @test_throws ArgumentError SPlit.support_points(
      GaussianKernel(1.0),
      data,
      5;
      target = R,
      weights = ones(200),
    )
  end
end
