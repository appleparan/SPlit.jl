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
    est = energydistance(X, Y; subsample = 150, repeats = 30, rng = MersenneTwister(1))
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
