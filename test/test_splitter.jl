using Test
using SPlit
using Random
using DataFrames

@testset "SupportPointSplitter" begin
  @testset "constructor validation" begin
    @test_throws ArgumentError SupportPointSplitter(ratio = 0.0)
    @test_throws ArgumentError SupportPointSplitter(ratio = 1.0)
    @test_throws ArgumentError SupportPointSplitter(max_iterations = 0)
    @test_throws ArgumentError SupportPointSplitter(tolerance = 0.0)
    @test_throws ArgumentError SupportPointSplitter(n_threads = 0)
    @test_throws ArgumentError SupportPointSplitter(kappa = 0)
    @test SupportPointSplitter().kernel isa EnergyKernel
  end

  @testset "accepts non-Float64 Real ratio and tolerance" begin
    s = SupportPointSplitter(ratio = 1 // 5, tolerance = 1.0f-8)
    @test s.ratio === 0.2
    @test s.tolerance === Float64(1.0f-8)
  end

  @testset "matrix split: sizes, partition, honest report" begin
    rng = MersenneTwister(1)
    data = randn(rng, 200, 3)
    s = SupportPointSplitter(ratio = 0.2, max_iterations = 100, rng = MersenneTwister(2))
    r = datasplit(s, data)
    @test length(test_indices(r)) == 40
    @test length(train_indices(r)) == 160
    @test sort(vcat(train_indices(r), test_indices(r))) == collect(1:200)
    @test r.iterations <= 100
    @test r.converged isa Bool
  end

  @testset "ratio > 0.5 puts the larger side in test" begin
    data = randn(MersenneTwister(3), 100, 2)
    s = SupportPointSplitter(ratio = 0.8, max_iterations = 50, rng = MersenneTwister(4))
    r = datasplit(s, data)
    @test length(test_indices(r)) == 80
    @test length(train_indices(r)) == 20
  end

  @testset "DataFrame with categoricals" begin
    df = DataFrame(x = randn(MersenneTwister(5), 90), g = repeat(["a", "b", "c"], 30))
    s = SupportPointSplitter(max_iterations = 50, rng = MersenneTwister(6))
    r = datasplit(s, df)
    @test length(test_indices(r)) == 18
    train_view = df[r, :train]
    @test nrow(train_view) == 72
  end

  @testset "vector input" begin
    v = randn(MersenneTwister(7), 50)
    r = datasplit(SupportPointSplitter(max_iterations = 50, rng = MersenneTwister(8)), v)
    @test length(test_indices(r)) == 10
  end

  @testset "reproducible with rng" begin
    data = randn(MersenneTwister(9), 100, 2)
    r1 =
      datasplit(SupportPointSplitter(max_iterations = 60, rng = MersenneTwister(10)), data)
    r2 =
      datasplit(SupportPointSplitter(max_iterations = 60, rng = MersenneTwister(10)), data)
    @test test_indices(r1) == test_indices(r2)
  end

  @testset "iteration and getindex sugar" begin
    data = randn(MersenneTwister(11), 60, 2)
    r =
      datasplit(SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(12)), data)
    train, test = r
    @test train == train_indices(r)
    @test test == test_indices(r)
    @test size(data[r, :train], 1) == length(train)
    @test_throws ArgumentError data[r, :validation]
  end

  @testset "legacy API is gone" begin
    @test !isdefined(SPlit, :split_data)
    @test !isdefined(SPlit, :format_data)
    @test !isdefined(SPlit, :EnergyDistance)
  end
end

@testset "GaussianKernel through datasplit" begin
  @testset "constructor rejects kappa with GaussianKernel" begin
    @test_throws ArgumentError SupportPointSplitter(kernel = GaussianKernel(), kappa = 50)
    @test SupportPointSplitter(kernel = GaussianKernel()).kernel.bandwidth === :median
  end

  @testset "split works and stores the resolved bandwidth" begin
    data = randn(MersenneTwister(70), 200, 2)
    s = SupportPointSplitter(
      kernel = GaussianKernel(),
      max_iterations = 100,
      rng = MersenneTwister(71),
    )
    r = datasplit(s, data)
    @test length(test_indices(r)) == 40
    @test sort(vcat(train_indices(r), test_indices(r))) == collect(1:200)
    @test r.method.kernel isa GaussianKernel{Float64}
    @test r.method.kernel.bandwidth > 0
    @test r.method.ratio == 0.2
    # the original splitter is untouched
    @test s.kernel.bandwidth === :median
  end

  @testset "reproducible: same rng ⇒ same bandwidth and indices" begin
    data = randn(MersenneTwister(72), 150, 3)
    r1 = datasplit(
      SupportPointSplitter(
        kernel = GaussianKernel(),
        max_iterations = 60,
        rng = MersenneTwister(73),
      ),
      data,
    )
    r2 = datasplit(
      SupportPointSplitter(
        kernel = GaussianKernel(),
        max_iterations = 60,
        rng = MersenneTwister(73),
      ),
      data,
    )
    @test r1.method.kernel.bandwidth == r2.method.kernel.bandwidth
    @test test_indices(r1) == test_indices(r2)
  end

  @testset "DataFrame input with categoricals" begin
    df = DataFrame(x = randn(MersenneTwister(74), 90), g = repeat(["a", "b", "c"], 30))
    r = datasplit(
      SupportPointSplitter(
        kernel = GaussianKernel(1.0),
        max_iterations = 50,
        rng = MersenneTwister(75),
      ),
      df,
    )
    @test length(test_indices(r)) == 18
  end

  @testset "duplicate rows take the jitter path" begin
    data = repeat(randn(MersenneTwister(76), 30, 2), 3)
    r = datasplit(
      SupportPointSplitter(
        kernel = GaussianKernel(1.0),
        max_iterations = 30,
        rng = MersenneTwister(77),
      ),
      data,
    )
    @test length(test_indices(r)) == 18
    @test sort(vcat(train_indices(r), test_indices(r))) == collect(1:90)
  end
end

@testset "splitter hierarchy" begin
  @test SupportPointSplitter <: AbstractSplitter
  data = randn(MersenneTwister(90), 60, 2)
  r = datasplit(SupportPointSplitter(max_iterations = 20, rng = MersenneTwister(91)), data)
  @test r isa SplitResult{<:SupportPointSplitter}
  @test r.method isa AbstractSplitter
end

@testset "datasplit with weights" begin
  data = randn(MersenneTwister(100), 200, 3)

  @testset "uniform weights reproduce the unweighted split exactly" begin
    for kernel in (EnergyKernel(), GaussianKernel(1.0))
      s =
        SupportPointSplitter(kernel = kernel, max_iterations = 60, rng = MersenneTwister(1))
      r0 = datasplit(s, data)
      s =
        SupportPointSplitter(kernel = kernel, max_iterations = 60, rng = MersenneTwister(1))
      r1 = datasplit(s, data; weights = ones(200))
      # weighted standardization matches the unweighted one up to rounding,
      # so the selected rows agree; the optimizer trajectory is compared
      # bit for bit at the `support_points` level in test_optimizer.jl
      @test r1.test_indices == r0.test_indices
      @test r1.train_indices == r0.train_indices
    end
  end

  @testset "a :median bandwidth resolves to a numeric kernel with weights" begin
    s = SupportPointSplitter(
      kernel = GaussianKernel(),
      max_iterations = 5,
      rng = MersenneTwister(2),
    )
    r = datasplit(s, data; weights = rand(MersenneTwister(3), 200))
    @test r.method.kernel isa GaussianKernel{Float64}
  end

  @testset "heavy cluster gets more test rows" begin
    rng = MersenneTwister(101)
    A = randn(rng, 200, 2) .- 4.0
    B = randn(rng, 200, 2) .+ 4.0
    X = vcat(A, B)
    w = vcat(fill(9.0, 200), fill(1.0, 200))
    s = SupportPointSplitter(ratio = 0.2, max_iterations = 100, rng = MersenneTwister(4))
    r_u = datasplit(s, X)
    s = SupportPointSplitter(ratio = 0.2, max_iterations = 100, rng = MersenneTwister(4))
    r_w = datasplit(s, X; weights = w)
    @test count(<=(200), r_w.test_indices) > count(<=(200), r_u.test_indices)
  end

  @testset "DataFrame input with weights" begin
    df = DataFrame(x = randn(MersenneTwister(102), 90), g = repeat(["a", "b", "c"], 30))
    s = SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(5))
    r = datasplit(s, df; weights = rand(MersenneTwister(6), 90))
    @test length(test_indices(r)) == 18
  end

  @testset "validation" begin
    s = SupportPointSplitter(max_iterations = 5)
    @test_throws ArgumentError datasplit(s, data; weights = ones(199))
    @test_throws ArgumentError datasplit(s, data; weights = -ones(200))
  end
end

@testset "splitquality with weights" begin
  data = randn(MersenneTwister(103), 150, 2)
  s = SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(7))
  r = datasplit(s, data)
  @test splitquality(data, r; weights = ones(150)) == splitquality(data, r)
  w = rand(MersenneTwister(8), 150)
  q = splitquality(data, r; weights = w)
  @test q isa Float64
  @test q >= -1e-12
  # equals the weighted discrepancy between the weighted train and test rows
  X = SPlit.preprocess(data, w)
  wn = w ./ sum(w)
  expected = energydistance(
    X[r.train_indices, :],
    X[r.test_indices, :];
    weights_x = wn[r.train_indices],
    weights_y = wn[r.test_indices],
  )
  @test isapprox(q, expected; atol = 1e-12)
  @test_throws ArgumentError splitquality(data, r; weights = ones(10))
end
