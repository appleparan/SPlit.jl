"""Parity tests for sample weights in splitiq."""

from __future__ import annotations

import numpy as np
import pytest

from splitiq import datasplit, energydistance, mmd, splitquality


def _clusters(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((150, 2)) - 4.0
    b = rng.standard_normal((150, 2)) + 4.0
    weights = np.concatenate([np.full(150, 9.0), np.full(150, 1.0)])
    return np.vstack([a, b]), weights


def test_uniform_weights_reproduce_the_unweighted_split() -> None:
    data = np.random.default_rng(1).standard_normal((120, 2))
    plain = datasplit(data, ratio=0.2, seed=3, max_iterations=40)
    weighted = datasplit(data, ratio=0.2, seed=3, max_iterations=40, weights=np.ones(120))
    np.testing.assert_array_equal(plain.test_indices, weighted.test_indices)


def test_heavy_cluster_gets_more_test_rows() -> None:
    data, weights = _clusters()
    plain = datasplit(data, ratio=0.2, seed=4, max_iterations=100)
    weighted = datasplit(data, ratio=0.2, seed=4, max_iterations=100, weights=weights)
    assert np.sum(weighted.test_indices < 150) > np.sum(plain.test_indices < 150)


def test_herding_accepts_weights() -> None:
    data, weights = _clusters(seed=2)
    plain = datasplit(data, ratio=0.2, method='herding', kernel='gaussian', bandwidth=1.0)
    weighted = datasplit(
        data, ratio=0.2, method='herding', kernel='gaussian', bandwidth=1.0, weights=weights
    )
    assert np.sum(weighted.test_indices < 150) > np.sum(plain.test_indices < 150)


def test_energydistance_duplication_invariance() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((30, 2))
    y = rng.standard_normal((25, 2)) + 0.5
    x_dup = np.vstack([x[:1], x])
    weights_x = np.concatenate([[2.0], np.ones(29)])
    assert energydistance(x, y, weights_x=weights_x) == pytest.approx(
        energydistance(x_dup, y), abs=1e-10
    )
    assert mmd(x, y, bandwidth=0.8, weights_x=weights_x) == pytest.approx(
        mmd(x_dup, y, bandwidth=0.8), abs=1e-10
    )


def test_splitquality_accepts_weights() -> None:
    data, weights = _clusters(seed=6)
    result = datasplit(data, ratio=0.2, seed=7, max_iterations=40, weights=weights)
    assert splitquality(data, result, weights=weights) >= -1e-12


def test_weights_accept_lists_and_series_like_inputs() -> None:
    data = np.random.default_rng(8).standard_normal((40, 2))
    result = datasplit(data, ratio=0.25, seed=1, max_iterations=10, weights=[1.0] * 40)
    assert len(result.test_indices) == 10


@pytest.mark.parametrize(
    'bad',
    [np.ones(39), -np.ones(40), np.zeros(40), np.ones((40, 1))],
    ids=['wrong-length', 'negative', 'all-zero', 'two-dimensional'],
)
def test_bad_weights_raise_value_error(bad: np.ndarray) -> None:
    data = np.random.default_rng(9).standard_normal((40, 2))
    with pytest.raises(ValueError, match='weights'):
        datasplit(data, ratio=0.25, seed=1, max_iterations=10, weights=bad)
