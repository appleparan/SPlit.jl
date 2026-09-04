# Examples

`llm_data_selection.jl` selects LLM training data from a public arXiv-abstract
embedding matrix with every `SPlit` splitter, under plain/weighted/targeted
settings, and times Compress++ against plain kernel thinning.

Run it from the repository root (`path="."` is the checkout — SPlit is a
registered package, so a plain `Pkg.instantiate()` resolves the registry
version instead):

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia -t auto --project=examples examples/llm_data_selection.jl
```

The resulting table is written to `docs/src/assets/examples/llm_selection.md`.
