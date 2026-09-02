"""Property-style tests for splitiq.datasplit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import datasplit


def _data(n: int, p: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, p))


def test_indices_disjoint_and_complete() -> None:
    n = 150
    data = _data(n, seed=1)
    result = datasplit(data, ratio=0.2, seed=1, max_iterations=20)
    train = set(result.train_indices.tolist())
    test = set(result.test_indices.tolist())
    assert train.isdisjoint(test)
    assert train | test == set(range(n))


def test_ratio_below_half_puts_smaller_side_in_test() -> None:
    n = 150
    data = _data(n, seed=2)
    result = datasplit(data, ratio=0.2, seed=1, max_iterations=20)
    assert len(result.test_indices) == round(0.2 * n)
    assert len(result.train_indices) == n - round(0.2 * n)


def test_ratio_above_half_puts_larger_side_in_test() -> None:
    n = 150
    data = _data(n, seed=3)
    result = datasplit(data, ratio=0.8, seed=1, max_iterations=20)
    assert len(result.test_indices) == round(0.8 * n)
    assert len(result.train_indices) == n - round(0.8 * n)


def test_seeded_split_is_reproducible() -> None:
    data = _data(150, seed=4)
    first = datasplit(data, ratio=0.2, seed=42, max_iterations=50)
    second = datasplit(data, ratio=0.2, seed=42, max_iterations=50)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)


def test_different_seed_gives_different_split() -> None:
    data = _data(150, seed=5)
    first = datasplit(data, ratio=0.2, seed=42, max_iterations=50)
    second = datasplit(data, ratio=0.2, seed=7, max_iterations=50)
    assert not np.array_equal(first.train_indices, second.train_indices)


def test_apply_returns_numpy_subsets_of_the_right_shape() -> None:
    data = _data(100, seed=6)
    result = datasplit(data, ratio=0.2, seed=1, max_iterations=20)
    train, test = result.apply(data)
    assert train.shape == (len(result.train_indices), data.shape[1])
    assert test.shape == (len(result.test_indices), data.shape[1])


def test_apply_returns_pandas_subsets_of_the_right_shape() -> None:
    n = 100
    rng = np.random.default_rng(7)
    df = pd.DataFrame({'x': rng.standard_normal(n), 'y': rng.standard_normal(n)})
    result = datasplit(df, ratio=0.2, seed=1, max_iterations=20)
    train, test = result.apply(df)
    assert len(train) == len(result.train_indices)
    assert len(test) == len(result.test_indices)


def test_herding_is_deterministic_without_a_seed() -> None:
    data = _data(100, seed=8)
    first = datasplit(data, ratio=0.2, method='herding')
    second = datasplit(data, ratio=0.2, method='herding')
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert first.converged is True


def test_gaussian_kernel_resolves_a_positive_bandwidth() -> None:
    data = _data(100, seed=9)
    result = datasplit(data, ratio=0.2, kernel='gaussian', seed=1, max_iterations=20)
    assert result.bandwidth is not None
    assert result.bandwidth > 0


def test_gaussian_kernel_stores_a_fixed_bandwidth() -> None:
    data = _data(100, seed=10)
    result = datasplit(data, ratio=0.2, kernel='gaussian', bandwidth=1.0, seed=1, max_iterations=20)
    assert result.bandwidth == pytest.approx(1.0)


def test_kappa_runs_stochastic_mode() -> None:
    n = 300
    data = _data(n, seed=11)
    result = datasplit(data, ratio=0.2, kappa=50, seed=1, max_iterations=20)
    assert len(result.train_indices) + len(result.test_indices) == n


def test_ratio_outside_unit_interval_raises_value_error() -> None:
    data = _data(50, seed=12)
    with pytest.raises(ValueError, match='ratio'):
        datasplit(data, ratio=1.5)


def test_3d_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match='1-D or 2-D'):
        datasplit(np.zeros((5, 2, 2)))


def test_dataframe_with_missing_values_raises_value_error() -> None:
    df = pd.DataFrame({'x': [1.0, np.nan, 3.0], 'y': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='missing value'):
        datasplit(df)


def test_dataframe_with_category_column_splits() -> None:
    n = 90
    rng = np.random.default_rng(13)
    df = pd.DataFrame(
        {'x': rng.standard_normal(n), 'g': pd.Categorical(['a', 'b', 'c'] * (n // 3))}
    )
    result = datasplit(df, ratio=0.2, seed=1, max_iterations=20)
    assert len(result.train_indices) + len(result.test_indices) == n


def test_dataframe_with_string_column_splits() -> None:
    n = 90
    rng = np.random.default_rng(14)
    df = pd.DataFrame({'x': rng.standard_normal(n), 'g': ['a', 'b', 'c'] * (n // 3)})
    result = datasplit(df, ratio=0.2, seed=1, max_iterations=20)
    assert len(result.train_indices) + len(result.test_indices) == n


def test_herding_with_kappa_raises_value_error() -> None:
    data = _data(60, seed=15)
    with pytest.raises(ValueError, match='herding'):
        datasplit(data, method='herding', kappa=10)


def test_unknown_kernel_raises_value_error() -> None:
    data = _data(60, seed=16)
    with pytest.raises(ValueError, match='kernel'):
        datasplit(data, kernel='unknown')  # ty: ignore[invalid-argument-type]


def test_unknown_method_raises_value_error() -> None:
    data = _data(60, seed=17)
    with pytest.raises(ValueError, match='method'):
        datasplit(data, method='unknown')  # ty: ignore[invalid-argument-type]


def test_iter_unpacks_into_train_and_test() -> None:
    data = _data(60, seed=18)
    result = datasplit(data, ratio=0.2, seed=1, max_iterations=20)
    train, test = result
    np.testing.assert_array_equal(train, result.train_indices)
    np.testing.assert_array_equal(test, result.test_indices)
