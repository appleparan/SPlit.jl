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

  @testset "reference-targeted splits beat random subsets against the reference" begin
    rng = MersenneTwister(620)
    data = randn(rng, 300, 3)
    R = data[data[:, 1].>0.3, :]
    for s in (
      SupportPointSplitter(max_iterations = 200, rng = MersenneTwister(621)),
      HerdingSplitter(kernel = GaussianKernel(1.0)),
    )
      r = datasplit(s, data; reference = R)
      q = splitquality(data, r; reference = R)
      n_test = length(test_indices(r))
      random_qs = map(1:25) do i
        perm = randperm(MersenneTwister(6_000 + i), 300)
        fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, r.method)
        splitquality(data, fake; reference = R)
      end
      @test q < mean(random_qs)
    end
  end

  @testset "a reference that is not a subset of the data still pulls the selection" begin
    data = randn(MersenneTwister(630), 400, 2)
    ref = randn(MersenneTwister(631), 150, 2) .+ [2.0 0.0]
    for s in (
      SupportPointSplitter(max_iterations = 150, rng = MersenneTwister(632)),
      HerdingSplitter(kernel = GaussianKernel(1.0)),
    )
      idx = selectrows(deepcopy(s), data, 60; reference = ref)
      @test mean(data[idx, 1]) > mean(data[:, 1]) + 0.5
    end
  end
  @testset "twinning splits beat random splits under the energy distance" begin
    mixture = let rng = MersenneTwister(300), N = 400
      c = rand(rng, 1:4, N)
      centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
      centers[c, :] .+ randn(rng, N, 2)
    end
    for (seed, data) in ((301, mixture), (302, randn(MersenneTwister(302), 400, 4)))
      s = TwinningSplitter()
      r = datasplit(s, data)
      q = splitquality(data, r)
      n_test = length(test_indices(r))
      random_qs = map(1:25) do i
        perm = randperm(MersenneTwister(1_000 * seed + i), size(data, 1))
        fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
        splitquality(data, fake)
      end
      @test q < mean(random_qs)
    end
  end

  @testset "twinning multiplets beat random folds on the worst fold's energy distance" begin
    data = randn(MersenneTwister(310), 400, 3)
    X = SPlit.preprocess(data)
    worst(folds) = maximum(energydistance(X[f, :], X) for f in folds)
    random_worst = map(1:20) do i
      perm = randperm(MersenneTwister(3_000 + i), 400)
      worst([perm[(100*(j-1)+1):(100*j)] for j = 1:4])
    end
    for strategy in (:sequential, :halving, :single)
      folds = multiplet(TwinningSplitter(), data, 4; strategy)
      @test worst(folds) < mean(random_worst)
    end
  end

  @testset "kernel-thinning splits beat random splits under energy distance and MMD" begin
    mixture = let rng = MersenneTwister(400), N = 400
      c = rand(rng, 1:4, N)
      centers = [-3.0 -3.0; 3.0 -3.0; -3.0 3.0; 3.0 3.0]
      centers[c, :] .+ randn(rng, N, 2)
    end
    for (seed, data) in ((401, mixture), (402, randn(MersenneTwister(402), 400, 4)))
      for kernel in (EnergyKernel(), GaussianKernel(1.0))
        s = KernelThinningSplitter(kernel = kernel, rng = MersenneTwister(seed))
        r = datasplit(s, data)
        q = splitquality(data, r; kernel)
        n_test = length(test_indices(r))
        random_qs = map(1:25) do i
          perm = randperm(MersenneTwister(1_000 * seed + i), size(data, 1))
          fake = SPlit.SplitResult(perm[(n_test+1):end], perm[1:n_test], true, 0, s)
          splitquality(data, fake; kernel)
        end
        @test q < mean(random_qs)
      end
    end
  end
end
