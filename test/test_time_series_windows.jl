using Test
using SPlit
using Random
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "time_series_windows_helpers.jl"))

@testset "time series window flattening" begin
  @testset "fixture from the plan" begin
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0; 5.0 50.0]
    L, stride = 2, 2
    @test window_count(size(X, 1), L, stride) == 2
    Z, starts = windows(X, L; stride = stride)
    @test starts == [1, 3]
    @test Z == [1.0 2.0 10.0 20.0; 3.0 4.0 30.0 40.0]
    @test size(Z) == (2, 4)
  end

  @testset "N < L gives zero windows, not an error" begin
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0]
    @test window_count(size(X, 1), 5, 1) == 0
    Z, starts = windows(X, 5; stride = 1)
    @test size(Z) == (0, 10)
    @test isempty(starts)
  end

  @testset "invalid L or stride throw ArgumentError" begin
    X = randn(MersenneTwister(1), 10, 2)
    @test_throws ArgumentError window_count(10, 0, 1)
    @test_throws ArgumentError window_count(10, 2, 0)
    @test_throws ArgumentError window_count(10, -1, 1)
    @test_throws ArgumentError windows(X, 0)
    @test_throws ArgumentError windows(X, 2; stride = 0)
  end

  @testset "stride = L windows share no observations" begin
    X = randn(MersenneTwister(2), 20, 3)
    L = 4
    Z, starts = windows(X, L; stride = L)
    @test length(starts) == window_count(20, L, L) == 5
    for i = 1:(length(starts)-1)
      @test starts[i+1] - starts[i] == L
      range_i = starts[i]:(starts[i]+L-1)
      range_next = starts[i+1]:(starts[i+1]+L-1)
      @test isempty(intersect(range_i, range_next))
    end
  end

  @testset "recover_window reproduces the original slice" begin
    X = randn(MersenneTwister(3), 17, 2)
    L, stride = 3, 2
    Z, starts = windows(X, L; stride = stride)
    p = size(X, 2)
    for i = 1:length(starts)
      @test recover_window(X, starts[i], L) == X[starts[i]:(starts[i]+L-1), :]
      @test reshape(Z[i, :], L, p) == recover_window(X, starts[i], L)
    end
  end

  @testset "standardize_by_variable: mean 0, std 1 on the fit set, applied elsewhere" begin
    L, p = 4, 2
    Z = randn(MersenneTwister(4), 50, L * p)
    Zs = standardize_by_variable(Z, L, p)
    for v = 1:p
      cols = ((v-1)*L+1):(v*L)
      @test mean(view(Zs, :, cols)) ≈ 0.0 atol = 1e-10
      @test std(view(Zs, :, cols)) ≈ 1.0
    end
    # applying the fit-set shift/scale to another matrix matches doing it by hand
    Z2 = randn(MersenneTwister(5), 10, L * p)
    Zs2 = standardize_by_variable(Z2, L, p; fit = Z)
    expected = similar(Zs2)
    for v = 1:p
      cols = ((v-1)*L+1):(v*L)
      m = mean(view(Z, :, cols))
      s = std(view(Z, :, cols))
      expected[:, cols] = (view(Z2, :, cols) .- m) ./ s
    end
    @test Zs2 ≈ expected
  end

  @testset "lag1_autocorrelation on hand-computable sequences" begin
    # single variable, arbitrary short sequence: hand-computed ratio
    z = [2.0, 4.0, 1.0, 8.0]
    xm = mean(z)
    d = z .- xm
    expected = sum(d[1:(end-1)] .* d[2:end]) / sum(abs2, d)
    @test lag1_autocorrelation(z, 4, 1) ≈ expected

    # slowly varying (arithmetic ramp), L = 10: hand-computed ratio is 0.7
    ramp = collect(1.0:10.0)
    @test lag1_autocorrelation(ramp, 10, 1) ≈ 0.7

    # alternating, L = 5: hand-computed ratio is -0.8
    alt = [1.0, -1.0, 1.0, -1.0, 1.0]
    @test lag1_autocorrelation(alt, 5, 1) ≈ -0.8

    # two variables average: 0.4 (ramp of length 5) and -0.8 (alternating) -> -0.2
    z2 = vcat(collect(1.0:5.0), alt)
    @test lag1_autocorrelation(z2, 5, 2) ≈ -0.2

    # a constant variable contributes 0.0, not NaN
    const_var = fill(5.0, 5)
    z3 = vcat(const_var, alt)
    @test lag1_autocorrelation(z3, 5, 2) ≈ -0.4
    @test !isnan(lag1_autocorrelation(z3, 5, 2))
  end

  @testset "two_regime_series: shape, labels, reproducibility" begin
    M, L, p = 30, 8, 3
    X, labels = two_regime_series(MersenneTwister(6); M = M, L = L, p = p)
    @test size(X) == (M * L, p)
    @test length(labels) == M
    @test all(l -> l in (:A, :B), labels)

    X2, labels2 = two_regime_series(MersenneTwister(6); M = M, L = L, p = p)
    @test X == X2
    @test labels == labels2

    X3, labels3 = two_regime_series(MersenneTwister(7); M = M, L = L, p = p)
    @test X != X3 || labels != labels3
  end

  @testset "selectrows(TwinningSplitter()) on flattened windows" begin
    rng = MersenneTwister(8)
    M, L, p = 40, 6, 3
    X, _ = two_regime_series(rng; M = M, L = L, p = p)
    Z, starts = windows(X, L; stride = L)
    @test size(Z, 1) == M
    Zs = standardize_by_variable(Z, L, p)
    n = 10
    sel = selectrows(TwinningSplitter(), Zs, n; standardize = false)
    @test length(sel) == n
    @test length(unique(sel)) == n
    @test all(i -> 1 <= i <= M, sel)
    for i in sel
      @test recover_window(X, starts[i], L) == reshape(Z[i, :], L, p)
    end
  end
end
