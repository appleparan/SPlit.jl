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

  @testset "Gaussian-kernel splits beat random splits under MMD and energy distance" begin
    data = randn(MersenneTwister(80), 250, 2)
    k = GaussianKernel(1.0)
    s = SupportPointSplitter(kernel = k, max_iterations = 150, rng = MersenneTwister(81))
    r = datasplit(s, data)
    q_mmd = splitquality(data, r; kernel = k)
    q_ed = splitquality(data, r)
    n_test = length(test_indices(r))
    rand_mmd = Float64[]
    rand_ed = Float64[]
    for i = 1:25
      perm = randperm(MersenneTwister(2_000 + i), 250)
      fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
      push!(rand_mmd, splitquality(data, fake; kernel = k))
      push!(rand_ed, splitquality(data, fake))
    end
    @test q_mmd < mean(rand_mmd)
    @test q_ed < mean(rand_ed)
  end

  @testset "herding splits beat random splits under MMD and energy distance" begin
    data = randn(MersenneTwister(110), 250, 2)
    for k in (GaussianKernel(1.0), EnergyKernel())
      r = datasplit(HerdingSplitter(kernel = k), data)
      q_k = splitquality(data, r; kernel = k)
      q_ed = splitquality(data, r)
      n_test = length(test_indices(r))
      rand_k = Float64[]
      rand_ed = Float64[]
      for i = 1:25
        perm = randperm(MersenneTwister(3_000 + i), 250)
        fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, r.method)
        push!(rand_k, splitquality(data, fake; kernel = k))
        push!(rand_ed, splitquality(data, fake))
      end
      @test q_k < mean(rand_k)
      @test q_ed < mean(rand_ed)
    end
  end

  @testset "weighted support-point splits beat random splits under the weighted energy distance" begin
    rng = MersenneTwister(120)
    data = randn(rng, 300, 3)
    w = exp.(randn(rng, 300))        # log-normal, heavy-tailed weights
    s = SupportPointSplitter(max_iterations = 200, rng = MersenneTwister(121))
    r = datasplit(s, data; weights = w)
    q_sp = splitquality(data, r; weights = w)
    n_test = length(test_indices(r))
    random_qs = map(1:25) do i
      perm = randperm(MersenneTwister(5_000 + i), 300)
      fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
      splitquality(data, fake; weights = w)
    end
    @test q_sp < mean(random_qs)
  end
end
