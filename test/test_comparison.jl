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
      "kernel",
      "ratio",
      "train",
      "test",
      "converged",
      "iterations",
      "energy_distance",
    ])
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
