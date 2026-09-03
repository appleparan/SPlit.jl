"""
Optimal splitting ratio from Joseph (2022), "Optimal Ratio for Data Splitting"
(Statistical Analysis and Data Mining 15(4), 531-538). The optimal testing
fraction is γ = 1/(√p + 1) (Eq. 11), where p is the number of parameters of
the linear regression model f(x)'β expected to explain the data well,
including the intercept (Section 2.2).
"""

"""
    optimal_split_ratio(x, y; method = :simple, degree = 2) -> Float64

Optimal fraction of data to assign to the test set, following the
γ* = 1/(√p + 1) result of Joseph (2022, Eq. 11).

- `method = :simple`: p is taken as the number of encoded predictor columns
  of `x` (after [`preprocess`](@ref): categorical columns Helmert-encoded,
  constant columns dropped) plus one, for the intercept.
- `method = :regression`: the paper's practical strategy (Section 3) for
  when the model is unknown — expand `x` into a larger feature set (e.g.
  polynomial terms up to `degree`), fit a linear regression on the full
  data, and select p via a model-selection criterion such as Cp or AIC.
  Not implemented in this release; calling with this method raises an
  error rather than silently falling back to `:simple`.

`x` may be an `AbstractVector`, `AbstractMatrix`, or `DataFrame`; `y` must
have the same number of observations as `x`.
"""
function optimal_split_ratio(x, y; method::Symbol = :simple, degree::Int = 2)
  method in (:simple, :regression) ||
    throw(ArgumentError("method must be :simple or :regression, got :$method"))
  nobs = x isa AbstractVector ? length(x) : size(x, 1)
  length(y) == nobs ||
    throw(ArgumentError("x and y must have the same number of observations"))

  if method === :regression
    error(
      "method = :regression is not implemented yet; the model-selection " *
      "based estimation procedure of p from Joseph (2022, Section 3) lands " *
      "in a later release",
    )
  end

  p = size(preprocess(x), 2) + 1
  return 1 / (sqrt(p) + 1)
end
