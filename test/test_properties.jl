using Test
using SPlit
using Random
using Statistics

@testset "paper properties (integration)" begin
  @testset "support-point splits beat random splits across datasets" begin
    for (seed, n, p) in ((1, 200, 2), (2, 300, 4))
      data = randn(MersenneTwister(seed), n, p)
      s = SupportPointSplitter(max_iterations = 200, rng = MersenneTwister(seed + 50))
      r = datasplit(s, data)
      q_sp = splitquality(data, r)
      n_test = length(test_indices(r))
      random_qs = map(1:25) do i
        perm = randperm(MersenneTwister(1_000 * seed + i), n)
        fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
        splitquality(data, fake)
      end
      @test q_sp < mean(random_qs)
    end
  end

  @testset "stochastic mode approximates full-data mode" begin
    data = randn(MersenneTwister(7), 800, 2)
    full =
      datasplit(SupportPointSplitter(max_iterations = 150, rng = MersenneTwister(8)), data)
    stoch = datasplit(
      SupportPointSplitter(kappa = 200, max_iterations = 150, rng = MersenneTwister(9)),
      data,
    )
    q_full = splitquality(data, full)
    q_stoch = splitquality(data, stoch)
    # stochastic optimization trades a bounded amount of quality for speed
    @test q_stoch < 3 * max(q_full, 1e-4)
  end
end
