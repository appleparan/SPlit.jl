using Test
using SPlit
using Random
using Statistics
using DataFrames
using LinearAlgebra

@testset "standardize = false" begin
  E = let M = randn(MersenneTwister(1), 300, 8)
    M ./ norm.(eachrow(M))            # cosine-normalized rows
  end
  E[:, 8] .= 0.25                      # a constant column: kept when standardize = false

  @testset "raw rows reach the splitter unchanged" begin
    rows = selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 60; standardize = false)
    direct = SPlit.herd(EnergyKernel(), Matrix{Float64}(E), 60)
    @test rows == direct
    @test rows != selectrows(HerdingSplitter(kernel = EnergyKernel()), E[:, 1:7], 60)  # standardized path differs
    r = datasplit(TwinningSplitter(), E; standardize = false)
    @test sort(vcat(train_indices(r), test_indices(r))) == 1:300
    v = randn(MersenneTwister(2), 100)
    @test selectrows(
      HerdingSplitter(kernel = EnergyKernel()),
      v,
      20;
      standardize = false,
    ) == SPlit.herd(EnergyKernel(), reshape(Matrix{Float64}(reshape(v, :, 1)), :, 1), 20)
  end

  @testset "weights and a reference on raw rows" begin
    w = rand(MersenneTwister(3), 300)
    @test selectrows(
      HerdingSplitter(kernel = EnergyKernel()),
      E,
      60;
      weights = w,
      standardize = false,
    ) == SPlit.herd(EnergyKernel(), Matrix{Float64}(E), 60; weights = w)
    R = E[1:40, :]
    @test selectrows(
      HerdingSplitter(kernel = EnergyKernel()),
      E,
      60;
      reference = R,
      standardize = false,
    ) == SPlit.herd(EnergyKernel(), Matrix{Float64}(E), 60; target = Matrix{Float64}(R))
    @test_throws ArgumentError selectrows(
      HerdingSplitter(),
      E,
      60;
      reference = E[1:40, 1:7],
      standardize = false,
    )
    @test_throws ArgumentError selectrows(
      HerdingSplitter(),
      E,
      60;
      weights = w,
      reference = R,
      standardize = false,
    )
    @test_throws ArgumentError selectrows(
      HerdingSplitter(),
      E,
      60;
      reference_weights = rand(40),
      standardize = false,
    )
  end

  @testset "DataFrames are rejected; the default is unchanged" begin
    df = DataFrame(x = randn(MersenneTwister(4), 50), g = repeat(["a", "b"], 25))
    @test_throws ArgumentError datasplit(HerdingSplitter(), df; standardize = false)
    @test_throws ArgumentError selectrows(
      HerdingSplitter(),
      E,
      10;
      reference = df[1:10, [:x]],
      standardize = false,
    )
    data = randn(MersenneTwister(5), 200, 3)
    @test test_indices(datasplit(HerdingSplitter(kernel = EnergyKernel()), data)) ==
          test_indices(
      datasplit(HerdingSplitter(kernel = EnergyKernel()), data; standardize = true),
    )
  end

  @testset "splitquality, compare, and multiplet on raw rows" begin
    r = datasplit(HerdingSplitter(kernel = EnergyKernel()), E; standardize = false)
    q = splitquality(E, r; standardize = false)
    @test q ≈ energydistance(E[train_indices(r), :], E[test_indices(r), :])
    @test q != splitquality(E[:, 1:7], r)
    qr = splitquality(E, r; reference = E[1:40, :], standardize = false)
    @test qr ≈ energydistance(E[test_indices(r), :], E[1:40, :])
    c = compare(
      [HerdingSplitter(kernel = EnergyKernel()), TwinningSplitter()],
      E;
      standardize = false,
    )
    @test c.qualities[1] ≈ q
    folds = multiplet(TwinningSplitter(), E, 4; standardize = false)
    @test sort(reduce(vcat, folds)) == 1:300
    @test_throws ArgumentError multiplet(
      TwinningSplitter(),
      DataFrame(x = randn(40)),
      4;
      standardize = false,
    )
  end
end
