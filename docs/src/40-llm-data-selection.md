# [Selecting LLM training data](@id llm-data-selection)

Given an embedding matrix (one row per document) and a budget of ``n``
rows, the splitters choose rows whose empirical distribution stays close
to a target measure: the whole corpus, a quality-weighted corpus, or a
smaller target set. This page is the workflow and a decision table; the
numbers come from `examples/llm_data_selection.jl` on 5,000 arXiv
abstracts embedded with MiniLM (384 dimensions, CC0).

## Workflow

1. Embed and cosine-normalize (``x / \|x\|``), then pass `standardize = false`
   so the angles are preserved.
2. Pick the measure: nothing (match the corpus), `weights` (a quality
   score per row; the selection matches the weighted corpus), or
   `reference` (a target sample; the selection matches it while drawing
   from the corpus).
3. Pick the method from the table below, call `selectrows`, and check
   with `energydistance` (or `splitquality` for a split).

```julia
using SPlit, Random
idx = selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 500; standardize = false)
idx_w = selectrows(HerdingSplitter(kernel = EnergyKernel()), E, 500; weights = quality, standardize = false)
idx_t = selectrows(KernelThinningSplitter(rng = MersenneTwister(1)), E, 500; reference = E_target, standardize = false)
few = selectrows(KernelThinningSplitter(), E, 100; standardize = false)   # Compress++ engages at n ≪ N
energydistance(E[idx, :], E)
```

```python
from splitiq import select_rows, energydistance
idx = select_rows(E, 500, method='herding', kernel='energy', standardize=False)
few = select_rows(E, 100, method='kernel_thinning', standardize=False)     # compress='auto'
```

## Which method

| N | n / N | weights or reference? | method | why |
|---|---|---|---|---|
| ≤ 10⁴ | any | any | `HerdingSplitter(EnergyKernel())` | lowest energy distance under all three measures in the example, and 1.5 s at N = 5,000; `KernelThinningSplitter` when MMD is the criterion |
| 10⁵–10⁶ | split ratio | no | `TwinningSplitter` | `O(pN log N)`, and no kernel matrix: 140 s at N = 10⁶ against 1,900–2,100 s for herding ([Benchmarks](@ref benchmarks)) |
| 10⁵–10⁶ | split ratio | yes | `HerdingSplitter` or `KernelThinningSplitter` | the only methods that take `weights`/`reference`; both `O(N²)`, with kernel thinning 8.6–11x herding's time at N = 10⁴ |
| ≥ 10⁵ | ≤ a few % | no | `KernelThinningSplitter(compress = :auto)` | [Compress++](@ref compress) removes both `O(N²)` terms; it pays off when `4^g (4 log₄ N + 1) < 1.5 N`, at N = 10⁶ up to a few 10⁴ rows |
| ≥ 10⁵ | ≤ a few % | yes | `HerdingSplitter` | Compress++ is not defined for `weights`/`reference`, so kernel thinning stays `O(N²)` there; herding's weighted data term is exact |

The N ≥ 10⁵ rows carry over the [Benchmarks](@ref benchmarks) timings,
which were recorded at p = 10. Embedding dimensions raise every constant,
and twinning is the only method measured there: 833 s at N = 10⁵, p = 768
([Design experiments](@ref twinning-trees)). Read the rows as an ordering
between methods, not as a time budget.

`SupportPointSplitter` is missing from the table on purpose. It optimizes
continuous points and then rounds each one to its nearest unclaimed data
row, and in 384 dimensions the optimizer moves the points less than the
spacing between rows, so `select_nearest` returns the initial random
sample — the rounding effect documented in section 3 of the
[Benchmarks](@ref benchmarks) page. Its energy distance to the data is
0.00231 under the plain and weighted measures and 0.00243 with a
reference: the same number every time, which is what that failure looks
like.

## What the example measures

Full table: [`assets/examples/llm_selection.md`](assets/examples/llm_selection.md).
At n = 500 out of N = 5,000, matching the corpus itself, herding reaches
an energy distance of 0.00107 and kernel thinning 0.00123 against 0.0024
for a uniform random subset, with twinning at 0.00166 in 0.28 s — the
fastest of the four. K-center greedy (Sener & Savarese, 2018) reaches
0.0228, 9.5x *worse* than random: farthest-first traversal maximizes
coverage, which pulls the selection toward the boundary of the cloud
rather than matching its distribution. Under `weights` (abstract length)
the gap widens: herding 0.00103 and kernel thinning 0.00109 against
0.00504 for random, while support points land at 0.00632, worse than
random. Under `reference` (the 250 `cs` rows of the corpus) random is
0.147 and k-center greedy 0.0909, while herding reaches 0.00818 and
kernel thinning 0.00812 — the setting where targeted selection earns the
most.

Compress++ pays for itself at this size only marginally: at n = 250,
`compress = :never` scores 0.00245 in 4.5 s and `compress = :always`
0.00261 in 3.1 s, both well below random's 0.00485. The 6% quality cost
buys a 1.5x speedup at N = 5,000; the ratio grows with N, which is why
`:auto` fires only when the cost rule says so.

## What this does not settle

Combining a quality score with distribution matching through a weighted
empirical measure is natural but not validated in the literature; the
example only shows that the weighted selections track the weighted corpus
better than unweighted ones do. If downstream results disagree,
stratified selection by quality quantile is the alternative to compare
against.

## References

- Sener, O., & Savarese, S. (2018). Active Learning for Convolutional Neural Networks: A Core-Set Approach. *ICLR*.
- Shetty, A., Dwivedi, R., & Mackey, L. (2022). Distribution Compression in Near-Linear Time. *ICLR*.
- Dataset: `sondalex/arxiv-abstracts-2021-embeddings-10000` on Hugging Face (arXiv abstracts with MiniLM embeddings, CC0).
