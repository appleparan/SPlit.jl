"""Parity tests for method='kernel_thinning' in splitiq."""

from __future__ import annotations

import numpy as np
import pytest

from splitiq import datasplit, energydistance, multiplet, select_rows


def _data(seed: int = 0, n: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 3))


def test_kernel_thinning_datasplit_partitions_and_reports() -> None:
    data = _data(1)
    result = datasplit(data, ratio=0.2, method='kernel_thinning', seed=1)
    assert result.method == 'kernel_thinning'
    assert result.kernel == 'energy'
    assert result.bandwidth is None
    assert result.converged
    assert result.iterations >= 0
    assert result.selected == 'test'
    all_indices = [*result.train_indices.tolist(), *result.test_indices.tolist()]
    assert sorted(all_indices) == list(range(300))
    again = datasplit(data, ratio=0.2, method='kernel_thinning', seed=1)
    assert np.array_equal(again.test_indices, result.test_indices)
    gauss = datasplit(data, ratio=0.25, method='kernel_thinning', kernel='gaussian', seed=2)
    assert gauss.bandwidth is not None
    assert len(gauss.test_indices) == 75


def test_kernel_thinning_beats_random_under_energy_distance() -> None:
    data = _data(2, 400)
    result = datasplit(data, ratio=0.2, method='kernel_thinning', seed=3)
    q = energydistance(data[result.test_indices], data[result.train_indices])
    rng = np.random.default_rng(4)
    random_qs = []
    for _ in range(10):
        perm = rng.permutation(400)
        random_qs.append(energydistance(data[perm[:80]], data[perm[80:]]))
    assert q < float(np.mean(random_qs))


def test_select_rows_delta_and_multiplet() -> None:
    data = _data(5)
    idx = select_rows(data, 60, method='kernel_thinning', delta=0.1, seed=6)
    assert len(set(idx.tolist())) == 60
    folds = multiplet(data, 4, method='kernel_thinning', seed=7)
    assert sorted(np.concatenate(folds).tolist()) == list(range(300))
    # more than half: the complement of a kernel-thinning selection of the other side
    idx200 = select_rows(data, 200, method='kernel_thinning', seed=9)
    assert len(set(idx200.tolist())) == 200


def test_kernel_thinning_ratio_half_regression() -> None:
    data = _data(10, 203)
    result = datasplit(data, ratio=0.5, method='kernel_thinning', seed=11)
    all_indices = [*result.train_indices.tolist(), *result.test_indices.tolist()]
    assert sorted(all_indices) == list(range(203))


def test_kernel_thinning_option_errors() -> None:
    data = _data(8)
    with pytest.raises(ValueError, match='delta'):
        datasplit(data, method='herding', delta=0.1)
    with pytest.raises(ValueError, match='delta'):
        datasplit(data, method='kernel_thinning', delta=1.5)
    with pytest.raises(ValueError, match='kappa'):
        datasplit(data, method='kernel_thinning', kappa=50)
    with pytest.raises(ValueError, match='start'):
        datasplit(data, method='kernel_thinning', start='random')
    with pytest.raises(ValueError, match='method'):
        datasplit(data, method='thinning')  # ty: ignore[invalid-argument-type]
