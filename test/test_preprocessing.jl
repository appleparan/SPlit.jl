using Test
using SPlit
using DataFrames
using CategoricalArrays
using Statistics

@testset "preprocess" begin
  @testset "matrix: standardizes columns" begin
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0]
    P = SPlit.preprocess(X)
    @test size(P) == (4, 2)
    for j in axes(P, 2)
      @test isapprox(mean(P[:, j]), 0.0; atol = 1e-12)
      @test isapprox(std(P[:, j]), 1.0; atol = 1e-12)
    end
  end

  @testset "matrix: drops constant columns" begin
    X = [1.0 5.0; 2.0 5.0; 3.0 5.0]
    P = SPlit.preprocess(X)
    @test size(P) == (3, 1)
  end

  @testset "matrix: all-constant errors" begin
    X = fill(2.0, 4, 2)
    @test_throws ArgumentError SPlit.preprocess(X)
  end

  @testset "missing values error" begin
    X = Matrix{Union{Float64,Missing}}([1.0 2.0; missing 3.0])
    @test_throws ArgumentError SPlit.preprocess(X)
    df = DataFrame(a = [1.0, missing, 3.0])
    @test_throws ArgumentError SPlit.preprocess(df)
  end

  @testset "vector input" begin
    P = SPlit.preprocess([1.0, 2.0, 3.0, 4.0])
    @test size(P) == (4, 1)
    @test isapprox(std(P[:, 1]), 1.0; atol = 1e-12)
  end

  @testset "dataframe: numeric + categorical (Helmert)" begin
    df = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], g = ["a", "b", "c", "a", "b", "c"])
    P = SPlit.preprocess(df)
    # 1 numeric + (3 levels - 1) Helmert columns = 3 columns
    @test size(P) == (6, 3)
    for j in axes(P, 2)
      @test isapprox(mean(P[:, j]), 0.0; atol = 1e-12)
    end
    # CategoricalVector behaves the same
    df2 = DataFrame(x = df.x, g = categorical(df.g))
    @test SPlit.preprocess(df2) == P
  end

  @testset "dataframe: unsupported column type errors" begin
    df = DataFrame(a = [1.0, 2.0], b = [[1], [2]])
    @test_throws ArgumentError SPlit.preprocess(df)
  end

  @testset "Union{Missing,T} columns without missing values are accepted" begin
    plain =
      DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], g = ["a", "b", "c", "a", "b", "c"])
    unioned = DataFrame(
      x = Vector{Union{Missing,Float64}}(plain.x),
      g = Vector{Union{Missing,String}}(plain.g),
    )
    @test SPlit.preprocess(unioned) == SPlit.preprocess(plain)
  end

  @testset "canonical Helmert level order is independent of row order" begin
    df = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], g = ["c", "a", "b", "c", "a", "b"])
    P = SPlit.preprocess(df)
    perm = [4, 1, 6, 2, 5, 3]
    shuffled = df[perm, :]
    P_shuffled = SPlit.preprocess(shuffled)
    @test P_shuffled[invperm(perm), :] == P
  end

  @testset "helmert_matrix" begin
    H = SPlit.helmert_matrix(3)
    @test size(H) == (3, 2)
    # each contrast column sums to zero
    @test all(isapprox.(sum(H; dims = 1), 0.0; atol = 1e-12))
    @test size(SPlit.helmert_matrix(1)) == (1, 0)
  end
end

@testset "weighted preprocess" begin
  using Random

  @testset "nothing dispatches to the unweighted method" begin
    data = randn(MersenneTwister(30), 50, 3)
    @test SPlit.preprocess(data, nothing) == SPlit.preprocess(data)
  end

  @testset "weighted mean 0 and weighted variance 1 per column" begin
    rng = MersenneTwister(31)
    data = randn(rng, 80, 3) .* [1.0 5.0 0.1] .+ [2.0 -1.0 0.0]
    w = rand(rng, 80)
    X = SPlit.preprocess(data, w)
    wn = w ./ sum(w)
    for j = 1:3
      μ = sum(wn .* X[:, j])
      σ2 = sum(wn .* (X[:, j] .- μ) .^ 2) / (1 - sum(abs2, wn))
      @test isapprox(μ, 0.0; atol = 1e-12)
      @test isapprox(σ2, 1.0; atol = 1e-12)
    end
  end

  @testset "uniform weights match the unweighted result up to rounding" begin
    data = randn(MersenneTwister(32), 60, 2)
    X_w = SPlit.apply_preprocessor(
      SPlit.fit_preprocessor(data; weights = fill(1 / 60, 60)),
      data,
    )
    @test isapprox(X_w, SPlit.preprocess(data); atol = 1e-12)
  end

  @testset "constant weight vector is treated as nothing and matches exactly" begin
    data = randn(MersenneTwister(32), 60, 2)
    @test SPlit.preprocess(data, ones(60)) == SPlit.preprocess(data)
  end

  @testset "all weight on one row errors" begin
    w = zeros(10)
    w[3] = 1.0
    @test_throws ArgumentError SPlit.preprocess(randn(10, 2), w)
  end

  @testset "column constant on positive-weight rows errors" begin
    @test_throws ArgumentError SPlit.preprocess(
      [5.0 1.0; 5.0 2.0; 7.0 3.0],
      [1.0, 1.0, 0.0],
    )
  end

  @testset "DataFrame with categoricals accepts weights" begin
    df = DataFrame(x = randn(MersenneTwister(33), 30), g = repeat(["a", "b", "c"], 10))
    X = SPlit.preprocess(df, ones(30))
    @test size(X) == (30, 3)
  end

  @testset "wrong length errors" begin
    @test_throws ArgumentError SPlit.preprocess(randn(10, 2), ones(9))
  end
end

@testset "Preprocessor fit/apply" begin
  @testset "preprocess is unchanged: matrix, weighted, DataFrame" begin
    rng = MersenneTwister(200)
    data = randn(rng, 70, 3) .* [1.0 4.0 0.2] .+ [1.0 -2.0 0.0]
    # reference values computed with the pre-M2 formulas, inline
    expected = copy(data)
    for j = 1:3
      μ = mean(expected[:, j])
      σ = std(expected[:, j])
      expected[:, j] .= (expected[:, j] .- μ) ./ σ
    end
    @test SPlit.preprocess(data) == expected
    w = rand(rng, 70)
    wn = w ./ sum(w)
    correction = 1 - sum(abs2, wn)
    expected_w = copy(data)
    for j = 1:3
      col = expected_w[:, j]
      μ = sum(wn .* col)
      σ = sqrt(sum(wn .* (col .- μ) .^ 2) / correction)
      expected_w[:, j] .= (col .- μ) ./ σ
    end
    @test SPlit.preprocess(data, w) == expected_w
    df = DataFrame(x = randn(MersenneTwister(201), 30), g = repeat(["a", "b", "c"], 10))
    X = SPlit.preprocess(df)
    @test size(X) == (30, 3)
    H = SPlit.helmert_matrix(3)
    idx = Dict("a" => 1, "b" => 2, "c" => 3)
    raw = hcat(df.x, [H[idx[v], 1] for v in df.g], [H[idx[v], 2] for v in df.g])
    for j = 1:3
      raw[:, j] .= (raw[:, j] .- mean(raw[:, j])) ./ std(raw[:, j])
    end
    @test X == raw
    # constant column dropped, all-constant errors
    @test size(SPlit.preprocess(hcat(ones(10), randn(MersenneTwister(202), 10))), 2) == 1
    @test_throws ArgumentError SPlit.preprocess(ones(10, 2))
  end

  @testset "apply uses the fitted μ and σ" begin
    R = randn(MersenneTwister(203), 100, 2)
    prep = SPlit.fit_preprocessor(R)
    Y = randn(MersenneTwister(204), 40, 2) .+ 5.0
    Ya = SPlit.apply_preprocessor(prep, Y)
    @test SPlit.apply_preprocessor(prep, R) == SPlit.preprocess(R)
    @test all(abs.(mean(Ya; dims = 1)) .> 3.0)     # not re-centered
    @test isapprox(Ya, (Y .- mean(R; dims = 1)) ./ std(R; dims = 1); atol = 1e-12)
  end

  @testset "weighted fit uses the weighted moments" begin
    R = randn(MersenneTwister(205), 80, 2)
    w = rand(MersenneTwister(206), 80)
    prep = SPlit.fit_preprocessor(R; weights = w)
    @test SPlit.apply_preprocessor(prep, R) == SPlit.preprocess(R, w)
  end

  @testset "columns constant on the fit set are dropped for both sets" begin
    R = hcat(ones(20), randn(MersenneTwister(207), 20))
    X = randn(MersenneTwister(208), 15, 2)
    prep = SPlit.fit_preprocessor(R; extra = X)
    @test size(SPlit.apply_preprocessor(prep, X), 2) == 1
    @test_throws ArgumentError SPlit.fit_preprocessor(ones(20, 2); extra = X)
  end

  @testset "categorical levels are the union, in canonical order" begin
    R = DataFrame(x = randn(MersenneTwister(209), 12), g = repeat(["a", "b"], 6))
    X = DataFrame(x = randn(MersenneTwister(210), 9), g = repeat(["a", "b", "c"], 3))
    prep = SPlit.fit_preprocessor(R; extra = X)
    spec = prep.specs[2]
    @test spec isa SPlit.CategoricalColumn
    @test spec.levels == ["a", "b", "c"]
    XR = SPlit.apply_preprocessor(prep, R)
    XX = SPlit.apply_preprocessor(prep, X)
    # the (a,b) vs c contrast is constant on R and is dropped: one Helmert column survives
    @test size(XR, 2) == 2
    @test size(XX, 2) == 2
    # level c is unknown when the preprocessor was fit without X
    prep_r = SPlit.fit_preprocessor(R)
    @test_throws ArgumentError SPlit.apply_preprocessor(prep_r, X)
    # CategoricalVector keeps the declared order, then data-only levels
    Rc = DataFrame(
      g = categorical(repeat(["z", "y"], 5); levels = ["z", "y", "w"]),
      x = randn(MersenneTwister(211), 10),
    )
    Xc = DataFrame(
      g = categorical(repeat(["q", "z"], 3); levels = ["q", "z"]),
      x = randn(MersenneTwister(212), 6),
    )
    prepc = SPlit.fit_preprocessor(Rc; extra = Xc)
    @test prepc.specs[1].levels == ["z", "y", "q"]
  end

  @testset "shape and column mismatches error" begin
    R = randn(MersenneTwister(213), 20, 3)
    prep = SPlit.fit_preprocessor(R)
    @test_throws ArgumentError SPlit.apply_preprocessor(prep, randn(5, 2))
    Rd = DataFrame(x = randn(10), g = repeat(["a", "b"], 5))
    prepd = SPlit.fit_preprocessor(Rd)
    @test_throws ArgumentError SPlit.apply_preprocessor(
      prepd,
      DataFrame(g = repeat(["a", "b"], 5), x = randn(10)),
    )
    @test_throws ArgumentError SPlit.apply_preprocessor(
      prepd,
      DataFrame(x = randn(10), g = randn(10)),
    )
    @test_throws ArgumentError SPlit.fit_preprocessor(R; extra = randn(5, 2))
    @test_throws ArgumentError SPlit.fit_preprocessor(R; extra = fill("a", 5, 3))
    @test_throws ArgumentError SPlit.apply_preprocessor(prep, [1.0, missing, 2.0][:, :])
  end
end
