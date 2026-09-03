"""Parity tests for method='twinning' and multiplet in splitiq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import datasplit, energydistance, multiplet, select_rows


def _data(seed: int = 0, n: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 3))


def test_twinning_datasplit_partitions_and_reports() -> None:
    data = _data(1)
    result = datasplit(data, ratio=0.2, method='twinning')
    assert result.method == 'twinning'
    assert result.kernel == 'energy'
    assert result.bandwidth is None
    assert result.converged
    assert result.iterations == 60
    assert result.selected == 'test'
    assert sorted([*result.train_indices.tolist(), *result.test_indices.tolist()]) == list(
        range(300)
    )
    again = datasplit(data, ratio=0.2, method='twinning')
    assert np.array_equal(again.test_indices, result.test_indices)


def test_twinning_beats_random_under_energy_distance() -> None:
    data = _data(2, 400)  # already standardized in distribution, so no preprocessing needed
    result = datasplit(data, ratio=0.2, method='twinning')
    q = energydistance(data[result.test_indices], data[result.train_indices])
    rng = np.random.default_rng(3)
    random_qs = []
    for _ in range(10):
        perm = rng.permutation(400)
        random_qs.append(energydistance(data[perm[:80]], data[perm[80:]]))
    assert q < float(np.mean(random_qs))


def test_select_rows_start_index_is_zero_based() -> None:
    data = _data(4)
    idx = select_rows(data, 30, method='twinning', start=17)
    assert idx[0] == 17
    assert len(set(idx.tolist())) == 30
    random_a = select_rows(data, 30, method='twinning', start='random', seed=5)
    random_b = select_rows(data, 30, method='twinning', start='random', seed=5)
    assert np.array_equal(random_a, random_b)


@pytest.mark.parametrize('strategy', ['sequential', 'halving', 'single'])
def test_multiplet_partitions_into_balanced_folds(strategy: str) -> None:
    data = _data(6, 203)
    folds = multiplet(data, 4, strategy=strategy)  # ty: ignore[invalid-argument-type]
    assert len(folds) == 4
    sizes = sorted(len(f) for f in folds)
    assert max(sizes) - min(sizes) <= 1
    assert sorted(np.concatenate(folds).tolist()) == list(range(203))
    assert all(f.min() >= 0 for f in folds)


def test_multiplet_with_other_methods_and_pandas() -> None:
    df = pd.DataFrame({'x': _data(7, 90)[:, 0], 'g': ['a', 'b', 'c'] * 30})
    folds = multiplet(df, 3, method='herding', kernel='energy')
    assert sorted(np.concatenate(folds).tolist()) == list(range(90))
    folds_sp = multiplet(df, 3, method='support_points', max_iterations=10, seed=1)
    assert len(folds_sp) == 3


def test_twinning_option_errors() -> None:
    data = _data(8)
    with pytest.raises(ValueError, match='kernel'):
        datasplit(data, method='twinning', kernel='gaussian')
    with pytest.raises(ValueError, match='kappa'):
        datasplit(data, method='twinning', kappa=50)
    with pytest.raises(ValueError, match='n_threads'):
        select_rows(data, 20, method='twinning', n_threads=2)
    with pytest.raises(ValueError, match='start'):
        datasplit(data, method='herding', start='random')
    with pytest.raises(ValueError, match='start'):
        select_rows(data, 20, method='twinning', start=-1)
    with pytest.raises(ValueError, match='weighted'):
        datasplit(data, method='twinning', weights=np.ones(300))
    with pytest.raises(ValueError, match='single'):
        multiplet(data, 4, method='herding', strategy='single')
    with pytest.raises(ValueError, match='power of two'):
        multiplet(data, 3, strategy='halving')
    with pytest.raises(ValueError, match='strategy'):
        multiplet(data, 4, strategy='other')  # ty: ignore[invalid-argument-type]
