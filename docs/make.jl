using SPlit
using Documenter

DocMeta.setdocmeta!(SPlit, :DocTestSetup, :(using SPlit); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers

makedocs(;
  modules = [SPlit],
  authors = "Jongsu Kim <jongsukim8@gmail.com> and contributors",
  repo = "https://github.com/appleparan/SPlit.jl/blob/{commit}{path}#{line}",
  sitename = "SPlit.jl",
  format = Documenter.HTML(; canonical = "https://appleparan.github.io/SPlit.jl"),
  pages = [
    "index.md"
    [
      file for file in readdir(joinpath(@__DIR__, "src")) if
      file != "index.md" && splitext(file)[2] == ".md"
    ]
  ],
)

deploydocs(; repo = "github.com/appleparan/SPlit.jl")
