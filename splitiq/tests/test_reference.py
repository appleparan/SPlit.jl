"""Parity tests for select and the reference distribution in splitiq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import datasplit, select_rows, splitquality


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    data = np.random.default_rng(seed).standard_normal((300, 2))
    return data, data[data[:, 0] > 0]


def test_select_returns_n_distinct_zero_based_indices() -> None:
    data, _ = _data(1)
    idx = select_rows(data, 60, seed=3, max_iterations=40)
    assert idx.shape == (60,)
    assert len(set(idx.tolist())) == 60
    assert idx.min() >= 0
    assert idx.max() < 300


def test_select_matches_the_selected_side_of_datasplit() -> None:
    data, _ = _data(2)
    result = datasplit(data, ratio=0.2, seed=4, max_iterations=40)
    idx = select_rows(data, 60, seed=4, max_iterations=40)
    assert result.selected == 'test'
    assert sorted(idx.tolist()) == sorted(result.test_indices.tolist())


def test_selected_side_follows_ratio() -> None:
    data, _ = _data(3)
    assert datasplit(data, ratio=0.8, seed=1, max_iterations=10).selected == 'train'


def test_reference_concentrates_the_selection() -> None:
    data, ref = _data(5)
    plain = select_rows(data, 60, seed=6, max_iterations=100)
    targeted = select_rows(data, 60, seed=6, max_iterations=100, reference=ref)
    assert np.sum(data[targeted, 0] > 0) > np.sum(data[plain, 0] > 0)


def test_herding_and_gaussian_accept_reference() -> None:
    data, ref = _data(7)
    idx = select_rows(data, 40, method='herding', kernel='gaussian', bandwidth=1.0, reference=ref)
    assert np.sum(data[idx, 0] > 0) >= 35
    result = datasplit(data, ratio=0.2, kernel='gaussian', seed=8, max_iterations=60, reference=ref)
    assert result.bandwidth is not None


def test_reference_weights_and_dataframes() -> None:
    data, _ = _data(9)
    df = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1], 'g': ['a', 'b', 'c'] * 100})
    ref = df[df.x > 0]
    weights = np.random.default_rng(10).random(len(ref))
    idx = select_rows(df, 60, seed=11, max_iterations=40, reference=ref, reference_weights=weights)
    assert idx.shape == (60,)


def test_splitquality_against_reference_is_lower_for_the_targeted_split() -> None:
    data, ref = _data(12)
    targeted = datasplit(data, ratio=0.2, seed=13, max_iterations=150, reference=ref)
    plain = datasplit(data, ratio=0.2, seed=13, max_iterations=150)
    assert splitquality(data, targeted, reference=ref) < splitquality(data, plain, reference=ref)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'reference': np.ones((10, 3))},
        {'reference_weights': np.ones(10)},
        {'reference': np.ones((10, 2)), 'weights': np.ones(300)},
    ],
    ids=['column-mismatch', 'weights-without-reference', 'weights-and-reference'],
)
def test_bad_reference_arguments_raise_value_error(kwargs: dict) -> None:
    data, _ = _data(14)
    with pytest.raises(ValueError, match=r'reference|n must'):
        select_rows(data, 10, seed=1, max_iterations=5, **kwargs)
