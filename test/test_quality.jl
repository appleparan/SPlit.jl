using Test
using SPlit
using Random
using Statistics
using DataFrames

@testset "energydistance" begin
  @testset "identical samples give zero" begin
    X = [0.0 0.0; 1.0 1.0; 2.0 0.5]
    @test isapprox(energydistance(X, copy(X)), 0.0; atol = 1e-12)
  end

  @testset "hand-computed 1D values" begin
    # X = {0}, Y = {1}: 2·E|X−Y| − E|X−X'| − E|Y−Y'| = 2·1 − 0 − 0 = 2
    @test isapprox(energydistance([0.0], [1.0]), 2.0; atol = 1e-12)
    # X = {0,1}, Y = {0,1}: identical empirical distributions ⇒ 0
    @test isapprox(energydistance([0.0, 1.0], [0.0, 1.0]), 0.0; atol = 1e-12)
  end

  @testset "nonnegative and sensitive to shift" begin
    rng = MersenneTwister(7)
    X = randn(rng, 200, 3)
    Y = randn(rng, 180, 3)
    Yshift = Y .+ 1.5
    @test energydistance(X, Y) >= -1e-12
    @test energydistance(X, Yshift) > energydistance(X, Y)
  end

  @testset "block accumulation matches naive computation" begin
    rng = MersenneTwister(11)
    X = randn(rng, 130, 2)   # > default block only if block small; force blocks
    Y = randn(rng, 90, 2)
    exact = energydistance(X, Y)
    blocked = SPlit._exact_energydistance(X, Y; block = 17)
    @test isapprox(exact, blocked; atol = 1e-10)
  end

  @testset "subsampled estimate agrees with exact" begin
    rng = MersenneTwister(42)
    X = randn(rng, 600, 2)
    Y = randn(rng, 600, 2) .+ 0.5
    exact = energydistance(X, Y)
    est = energydistance(X, Y; subsample = 300, repeats = 30, rng = MersenneTwister(1))
    @test isapprox(est, exact; rtol = 0.25)
  end

  @testset "dimension mismatch errors" begin
    @test_throws ArgumentError energydistance(randn(5, 2), randn(5, 3))
  end
end

@testset "splitquality" begin
  @testset "works on DataFrame with categoricals" begin
    df = DataFrame(x = randn(MersenneTwister(20), 120), g = repeat(["a", "b", "c"], 40))
    r = datasplit(SupportPointSplitter(max_iterations = 60, rng = MersenneTwister(21)), df)
    q = splitquality(df, r)
    @test q isa Float64
    @test q >= -1e-12
  end

  @testset "support-point split beats a random split on average" begin
    rng = MersenneTwister(22)
    data = randn(rng, 300, 2)
    r =
      datasplit(SupportPointSplitter(max_iterations = 150, rng = MersenneTwister(23)), data)
    q_sp = splitquality(data, r)
    random_qs = map(1:20) do i
      shuffled = randperm(MersenneTwister(100 + i), 300)
      fake = SPlit.SplitResult(shuffled[61:end], shuffled[1:60], true, 0, r.method)
      splitquality(data, fake)
    end
    @test q_sp < Statistics.mean(random_qs)
  end

  @testset "auto-switches to estimation above exact_threshold" begin
    # A skewed split (test = rows with the largest first coordinate) has a large
    # energy distance, so the subsampled estimate — which carries a positive
    # bias of order 1/subsample — can be compared to the exact value with a
    # relative tolerance. Near-zero exact values would make that comparison
    # meaningless.
    data = randn(MersenneTwister(24), 400, 2)
    order = sortperm(data[:, 1])
    method = SupportPointSplitter(max_iterations = 1, rng = MersenneTwister(25))
    skewed = SPlit.SplitResult(order[1:320], order[321:400], true, 1, method)
    exact = splitquality(data, skewed)
    est = splitquality(
      data,
      skewed;
      exact_threshold = 10,
      subsample = 100,
      repeats = 20,
      rng = MersenneTwister(26),
    )
    @test exact > 0.1
    @test isapprox(est, exact; rtol = 0.5)
    @test est != exact   # estimation path actually taken
  end
end

@testset "mmd" begin
  k = GaussianKernel(1.0)

  @testset "identical samples give zero; shift increases it" begin
    X = randn(MersenneTwister(30), 150, 2)
    @test isapprox(mmd(X, copy(X), k), 0.0; atol = 1e-12)
    Y = randn(MersenneTwister(31), 120, 2)
    @test mmd(X, Y, k) >= -1e-12
    @test mmd(X, Y .+ 1.5, k) > mmd(X, Y, k)
    # subsample larger than both samples falls back to the exact computation
    @test mmd(X, Y, k; subsample = 10_000) == mmd(X, Y, k)
  end

  @testset "block accumulation matches naive computation" begin
    X = randn(MersenneTwister(32), 70, 2)
    Y = randn(MersenneTwister(33), 45, 2)
    naive(A, B) =
      sum(SPlit.kernelvalue(k, A[i, :], B[j, :]) for i in axes(A, 1), j in axes(B, 1)) /
      (size(A, 1) * size(B, 1))
    exact = naive(X, X) + naive(Y, Y) - 2 * naive(X, Y)
    @test isapprox(mmd(X, Y, k), exact; atol = 1e-10)
    @test isapprox(SPlit._exact_mmd(k, X, Y; block = 13), exact; atol = 1e-10)
  end

  @testset "EnergyKernel delegates to energydistance; :median resolves on pooled rows" begin
    X = randn(MersenneTwister(34), 80, 2)
    Y = randn(MersenneTwister(35), 60, 2)
    @test mmd(X, Y, EnergyKernel()) == energydistance(X, Y)
    v1 = mmd(X, Y, GaussianKernel(); rng = MersenneTwister(5))
    v2 = mmd(X, Y, GaussianKernel(); rng = MersenneTwister(5))
    @test v1 == v2 && v1 >= 0
  end

  @testset "subsampled estimate agrees with exact on a skewed split" begin
    data = randn(MersenneTwister(36), 400, 2)
    order = sortperm(data[:, 1])
    X = data[order[1:320], :]
    Y = data[order[321:400], :]
    exact = mmd(X, Y, k)
    est = mmd(X, Y, k; subsample = 300, repeats = 20, rng = MersenneTwister(37))
    @test exact > 0.05
    @test isapprox(est, exact; rtol = 0.5)
    @test est != exact
  end

  @testset "splitquality with kernel keyword" begin
    data = randn(MersenneTwister(38), 200, 2)
    r =
      datasplit(SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(39)), data)
    @test splitquality(data, r) == splitquality(data, r; kernel = EnergyKernel())
    q = splitquality(data, r; kernel = k)
    @test q isa Float64 && q >= -1e-12
    @test q == mmd(
      SPlit.preprocess(data)[r.train_indices, :],
      SPlit.preprocess(data)[r.test_indices, :],
      k,
    )
  end
end

@testset "splitquality estimator keyword and threshold" begin
  data = randn(MersenneTwister(60), 300, 2)
  r = datasplit(SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(61)), data)
  q_exact = splitquality(data, r)
  @test q_exact == splitquality(data, r; estimator = Exact())
  @test splitquality(
    data,
    r;
    exact_threshold = 10,
    estimator = nothing,
    rng = MersenneTwister(1),
  ) == splitquality(
    data,
    r;
    estimator = SPlit._fallback_estimator(EnergyKernel()),
    rng = MersenneTwister(1),
  )
  q_sl = splitquality(data, r; estimator = RandomSlices(256), rng = MersenneTwister(2))
  @test isapprox(q_sl, q_exact; rtol = 0.5)
  @test splitquality(
    data,
    r;
    subsample = 100,
    repeats = 3,
    rng = MersenneTwister(4),
    exact_threshold = 10,
  ) == splitquality(data, r; estimator = Subsample(100, 3), rng = MersenneTwister(4))
end

@testset "weighted energydistance and mmd" begin
  rng = MersenneTwister(50)
  X = randn(rng, 30, 2)
  Y = randn(rng, 25, 2) .+ 0.5

  @testset "uniform weights give exactly the unweighted value" begin
    @test energydistance(X, Y; weights_x = ones(30), weights_y = ones(25)) ==
          energydistance(X, Y)
    @test energydistance(X, Y; weights_x = fill(0.2, 30)) == energydistance(X, Y)
    k = GaussianKernel(1.0)
    @test mmd(X, Y, k; weights_x = ones(30), weights_y = ones(25)) == mmd(X, Y, k)
    @test mmd(X, Y, EnergyKernel(); weights_x = ones(30)) == energydistance(X, Y)
  end

  @testset "hand-computed weighted 1-D values" begin
    # X = {0, 1} with weights (3, 1), Y = {1}:
    # 2·E|X−Y| = 2·(0.75·1 + 0.25·0) = 1.5; E|X−X'| = 2·0.75·0.25·1 = 0.375; E|Y−Y'| = 0
    @test isapprox(
      energydistance(
        reshape([0.0, 1.0], :, 1),
        reshape([1.0], :, 1);
        weights_x = [3.0, 1.0],
      ),
      1.5 - 0.375;
      atol = 1e-12,
    )
  end

  @testset "duplication invariance: weights as counts equal duplicated rows" begin
    Xdup = vcat(X[1:1, :], X)               # row 1 twice
    wx = vcat([2.0], ones(29))
    @test isapprox(
      energydistance(X, Y; weights_x = wx),
      energydistance(Xdup, Y);
      atol = 1e-12,
    )
    k = GaussianKernel(0.8)
    @test isapprox(mmd(X, Y, k; weights_x = wx), mmd(Xdup, Y, k); atol = 1e-12)
    # both sides weighted
    Ydup = vcat(Y, Y[end:end, :])
    wy = vcat(ones(24), [2.0])
    @test isapprox(
      energydistance(X, Y; weights_x = wx, weights_y = wy),
      energydistance(Xdup, Ydup);
      atol = 1e-12,
    )
  end

  @testset "block accumulation matches the unblocked weighted value" begin
    wx = rand(MersenneTwister(51), 30)
    wy = rand(MersenneTwister(52), 25)
    a = energydistance(X, Y; weights_x = wx, weights_y = wy)
    b = SPlit._exact_energydistance(X, Y, wx ./ sum(wx), wy ./ sum(wy); block = 7)
    @test isapprox(a, b; atol = 1e-10)
  end

  @testset "Subsample with weights runs, and is exact below m" begin
    wx = rand(MersenneTwister(53), 30)
    exact = energydistance(X, Y; weights_x = wx)
    @test energydistance(X, Y; weights_x = wx, estimator = Subsample(100)) == exact
    big = randn(MersenneTwister(54), 400, 2)
    wbig = rand(MersenneTwister(55), 400)
    est = energydistance(
      big,
      Y;
      weights_x = wbig,
      estimator = Subsample(150, 20),
      rng = MersenneTwister(1),
    )
    @test isapprox(est, energydistance(big, Y; weights_x = wbig); rtol = 0.3)
    k = GaussianKernel(1.0)
    @test isapprox(
      mmd(
        big,
        Y,
        k;
        weights_x = wbig,
        estimator = Subsample(150, 20),
        rng = MersenneTwister(1),
      ),
      mmd(big, Y, k; weights_x = wbig);
      rtol = 0.3,
    )
  end

  @testset "validation" begin
    @test_throws ArgumentError energydistance(X, Y; weights_x = ones(29))
    @test_throws ArgumentError energydistance(X, Y; weights_y = -ones(25))
    @test_throws ArgumentError mmd(X, Y, GaussianKernel(1.0); weights_x = zeros(30))
  end
end
