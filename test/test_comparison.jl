using Test
using SPlit
using Random
using DataFrames

@testset "compare" begin
  data = randn(MersenneTwister(1), 150, 2)
  splitters = [
    SupportPointSplitter(ratio = 0.2, max_iterations = 60, rng = MersenneTwister(2)),
    SupportPointSplitter(ratio = 0.3, max_iterations = 60, rng = MersenneTwister(3)),
  ]
  c = compare(splitters, data)

  @testset "structure" begin
    @test c isa SplitComparison
    @test length(c.results) == 2
    @test all(q -> q isa Float64, c.qualities)
  end

  @testset "DataFrame conversion" begin
    df = DataFrame(c)
    @test nrow(df) == 2
    @test Set(names(df)) == Set([
      "method",
      "kernel",
      "ratio",
      "train",
      "test",
      "converged",
      "iterations",
      "energy_distance",
    ])
    @test all(==("SupportPointSplitter"), df.method)
    @test df.test == [30, 45]
  end

  @testset "best picks the lowest energy distance" begin
    m, r = best(c)
    i = argmin(c.qualities)
    @test m === c.methods[i]
    @test r === c.results[i]
  end

  @testset "no Base.summary method (contract kept)" begin
    @test !any(
      m -> m.sig <: Tuple{typeof(Base.summary),SplitComparison},
      methods(Base.summary),
    )
  end
end

@testset "compare with a quality kernel" begin
  data = randn(MersenneTwister(50), 120, 2)
  splitters = [
    SupportPointSplitter(ratio = 0.2, max_iterations = 40, rng = MersenneTwister(51)),
    SupportPointSplitter(ratio = 0.3, max_iterations = 40, rng = MersenneTwister(52)),
  ]
  c = compare(splitters, data; kernel = GaussianKernel(1.0))
  @test c.kernel == GaussianKernel(1.0)
  df = DataFrame(c)
  @test "mmd" in names(df)
  @test !("energy_distance" in names(df))
  c0 = compare(splitters, data)
  @test c0.kernel == EnergyKernel()
  @test "energy_distance" in names(DataFrame(c0))
end

@testset "compare stores fitted splitters" begin
  data = randn(MersenneTwister(53), 120, 2)
  splitters = [
    SupportPointSplitter(
      kernel = GaussianKernel(),
      max_iterations = 40,
      rng = MersenneTwister(54),
    ),
    SupportPointSplitter(ratio = 0.3, max_iterations = 40, rng = MersenneTwister(55)),
  ]
  c = compare(splitters, data)
  @test all(c.methods[i] === c.results[i].method for i in eachindex(c.methods))
  @test c.methods[1].kernel isa GaussianKernel{Float64}
  @test splitters[1].kernel.bandwidth === :median
  m, r = best(c)
  @test m === r.method
end

@testset "compare resolves a :median scoring kernel once" begin
  data = randn(MersenneTwister(60), 1_500, 2)
  splitters = [
    SupportPointSplitter(ratio = 0.2, max_iterations = 20, rng = MersenneTwister(61)),
    SupportPointSplitter(ratio = 0.3, max_iterations = 20, rng = MersenneTwister(62)),
  ]
  c = compare(splitters, data; kernel = GaussianKernel(), rng = MersenneTwister(63))
  @test c.kernel isa GaussianKernel{Float64}
  X = SPlit.preprocess(data)
  expected = [mmd(X[r.train_indices, :], X[r.test_indices, :], c.kernel) for r in c.results]
  @test c.qualities == expected
  m, r = best(c)
  @test r === c.results[argmin(c.qualities)]
end

@testset "SplitComparison accepts any AbstractSplitter" begin
  @test fieldtype(SplitComparison, :methods) == Vector{AbstractSplitter}
end

@testset "compare mixes splitter types" begin
  data = randn(MersenneTwister(70), 150, 2)
  splitters = AbstractSplitter[
    SupportPointSplitter(max_iterations = 40, rng = MersenneTwister(71)),
    HerdingSplitter(kernel = EnergyKernel()),
    HerdingSplitter(kernel = GaussianKernel(1.0)),
  ]
  c = compare(splitters, data)
  df = DataFrame(c)
  @test df.method == ["SupportPointSplitter", "HerdingSplitter", "HerdingSplitter"]
  @test df.kernel == ["EnergyKernel", "EnergyKernel", "GaussianKernel"]
  m, r = best(c)
  @test m === r.method
end

@testset "compare forwards weights" begin
  data = randn(MersenneTwister(110), 120, 2)
  w = rand(MersenneTwister(111), 120)
  methods = [
    SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(1)),
    HerdingSplitter(kernel = GaussianKernel(1.0)),
  ]
  c = compare(methods, data; weights = w)
  @test length(c.qualities) == 2
  @test all(isfinite, c.qualities)
  @test isapprox(
    c.qualities[1],
    splitquality(data, c.results[1]; weights = w);
    atol = 1e-12,
  )
end

@testset "compare with a reference" begin
  data = randn(MersenneTwister(610), 150, 2)
  R = data[data[:, 1].>0, :]
  methods = [
    SupportPointSplitter(max_iterations = 30, rng = MersenneTwister(1)),
    HerdingSplitter(kernel = GaussianKernel(1.0)),
  ]
  c = compare(methods, data; reference = R)
  @test length(c.qualities) == 2
  @test all(isfinite, c.qualities)
  @test isapprox(
    c.qualities[2],
    splitquality(data, c.results[2]; reference = R);
    atol = 1e-12,
  )
  cg = compare(methods, data; reference = R, kernel = GaussianKernel())
  @test cg.kernel isa GaussianKernel{Float64}
end
