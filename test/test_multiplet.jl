using Test
using SPlit
using Random
using Statistics
using DataFrames

function check_partition(folds, N, k)
  @test length(folds) == k
  @test sort(reduce(vcat, folds)) == 1:N
  @test all(issorted, folds)
  sizes = length.(folds)
  @test maximum(sizes) - minimum(sizes) <= 1
end

@testset "multiplet" begin
  @testset "twinning: every strategy partitions with balanced, sorted folds" begin
    for (N, k) in ((200, 4), (203, 4), (97, 5), (64, 8))
      data = randn(MersenneTwister(N + k), N, 3)
      for strategy in (:sequential, :halving, :single)
        strategy === :halving && !ispow2(k) && continue
        folds = multiplet(TwinningSplitter(), data, k; strategy)
        check_partition(folds, N, k)
      end
    end
  end

  @testset ":single follows the neighbor rank of one twinning run" begin
    data = randn(MersenneTwister(20), 120, 2)
    X = SPlit.preprocess(data)
    groups = SPlit._twin_groups(X, 30, :farthest, MersenneTwister(0))   # N = 4·30: r = 4 exactly
    folds = multiplet(TwinningSplitter(), data, 4; strategy = :single)
    for j = 1:4
      @test folds[j] == sort([g[j] for g in groups])
    end
    # N mod k ≠ 0: the members above rank k go one each to the first folds
    data2 = randn(MersenneTwister(21), 123, 2)
    folds2 = multiplet(TwinningSplitter(), data2, 4; strategy = :single)
    @test length.(folds2) == [31, 31, 31, 30]
    # r > k: N = 11, k = 4 gives n = 2 groups of 5 and 6 rows; the 3 rows above rank 4 go to folds 1..3
    data3 = randn(MersenneTwister(27), 11, 2)
    folds3 = multiplet(TwinningSplitter(), data3, 4; strategy = :single)
    @test sort(reduce(vcat, folds3)) == 1:11
    @test length.(folds3) == [3, 3, 3, 2]
  end

  @testset ":sequential and :halving work with other splitters" begin
    data = randn(MersenneTwister(22), 160, 2)
    h = HerdingSplitter(kernel = EnergyKernel())
    check_partition(multiplet(h, data, 4), 160, 4)
    check_partition(multiplet(h, data, 4; strategy = :halving), 160, 4)
    sp = SupportPointSplitter(max_iterations = 20, rng = MersenneTwister(1))
    check_partition(multiplet(sp, data, 3), 160, 3)
    # weights forwarded (herding accepts them)
    check_partition(multiplet(h, data, 4; weights = rand(MersenneTwister(2), 160)), 160, 4)
  end

  @testset ":sequential reproduces repeated selectrows on the remaining rows" begin
    data = randn(MersenneTwister(23), 100, 2)
    s = TwinningSplitter()
    f1 = selectrows(s, data, 33)
    rest = setdiff(1:100, f1)
    f2 = rest[selectrows(s, data[rest, :], 33)]
    folds = multiplet(s, data, 3)
    @test folds[1] == sort(f1) && folds[2] == sort(f2)
    @test folds[3] == sort(setdiff(rest, f2))
  end

  @testset "DataFrame and vector inputs" begin
    df = DataFrame(x = randn(MersenneTwister(24), 90), g = repeat(["a", "b", "c"], 30))
    check_partition(multiplet(TwinningSplitter(), df, 3), 90, 3)
    check_partition(multiplet(TwinningSplitter(), randn(MersenneTwister(25), 50), 5), 50, 5)
  end

  @testset "validation" begin
    data = randn(MersenneTwister(26), 40, 2)
    @test_throws ArgumentError multiplet(TwinningSplitter(), data, 1)
    @test_throws ArgumentError multiplet(TwinningSplitter(), data, 41)
    @test_throws ArgumentError multiplet(TwinningSplitter(), data, 3; strategy = :halving)
    @test_throws ArgumentError multiplet(HerdingSplitter(), data, 4; strategy = :single)
    @test_throws ArgumentError multiplet(TwinningSplitter(), data, 4; strategy = :other)
    @test_throws ArgumentError multiplet(TwinningSplitter(), data, 4; weights = rand(40))
    @test_throws ArgumentError multiplet(
      TwinningSplitter(),
      data,
      4;
      strategy = :single,
      weights = rand(40),
    )
  end
end
