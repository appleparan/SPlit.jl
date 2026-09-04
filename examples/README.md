# Examples

`llm_data_selection.jl` selects LLM training data from a public arXiv-abstract
embedding matrix with every `SPlit` splitter, under plain/weighted/targeted
settings, and times Compress++ against plain kernel thinning.

Run it with `julia -t auto --project=examples examples/llm_data_selection.jl`
(instantiate the environment first: `julia --project=examples -e 'using Pkg; Pkg.instantiate()'`).

The resulting table is written to `docs/src/assets/examples/llm_selection.md`.
