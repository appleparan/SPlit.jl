"""Property-style tests for splitiq.optimal_split_ratio."""

from __future__ import annotations

import math

import numpy as np
import pytest

from splitiq import optimal_split_ratio


def test_gamma_formula_for_numeric_predictors() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 3))
    y = rng.standard_normal(100)
    p = 3 + 1  # 3 numeric predictors + intercept
    assert optimal_split_ratio(x, y) == pytest.approx(1 / (math.sqrt(p) + 1))


def test_constant_column_is_dropped_before_counting_p() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.standard_normal(80), np.full(80, 5.0)])
    y = rng.standard_normal(80)
    p = 1 + 1  # the constant column is dropped, leaving 1 predictor + intercept
    assert optimal_split_ratio(x, y) == pytest.approx(1 / (math.sqrt(p) + 1))


def test_regression_method_is_not_implemented() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 2))
    y = rng.standard_normal(40)
    with pytest.raises(NotImplementedError):
        optimal_split_ratio(x, y, method='regression')


def test_unknown_method_raises_value_error() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40, 2))
    y = rng.standard_normal(40)
    with pytest.raises(ValueError, match='method'):
        optimal_split_ratio(x, y, method='other')
