"""Property-style tests for splitiq's split-quality diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import (
    RandomFeatures,
    RandomSlices,
    Subsample,
    datasplit,
    energydistance,
    mmd,
    splitquality,
)


def test_energydistance_is_zero_for_identical_samples() -> None:
    x = np.random.default_rng(0).standard_normal((50, 2))
    assert energydistance(x, x.copy()) == pytest.approx(0.0, abs=1e-8)


def test_energydistance_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 2))
    y = rng.standard_normal((45, 2)) + 1.0
    assert energydistance(x, y) == pytest.approx(energydistance(y, x))


def test_splitquality_works_on_dataframe_with_categoricals() -> None:
    n = 120
    rng = np.random.default_rng(2)
    df = pd.DataFrame({'x': rng.standard_normal(n), 'g': ['a', 'b', 'c'] * (n // 3)})
    result = datasplit(df, ratio=0.2, seed=1, max_iterations=60)
    quality = splitquality(df, result)
    assert quality >= -1e-12


def test_support_point_split_beats_random_splits_on_average() -> None:
    # Mirrors the guarantee support points make (Mak & Joseph 2018): the
    # energy distance between the chosen train/test partition should beat
    # the average over random partitions of the same size.
    n = 300
    data = np.random.default_rng(3).standard_normal((n, 2))
    result = datasplit(data, ratio=0.2, seed=4, max_iterations=150)
    train, test = result.apply(data)
    support_quality = energydistance(train, test)

    n_test = len(result.test_indices)
    random_qualities = [
        energydistance(data[perm[n_test:]], data[perm[:n_test]])
        for perm in (np.random.default_rng(100 + i).permutation(n) for i in range(10))
    ]

    assert support_quality < np.mean(random_qualities)


def test_mmd_gaussian_is_nonnegative() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((40, 2))
    y = rng.standard_normal((40, 2)) + 2.0
    assert mmd(x, y, kernel='gaussian') >= -1e-10


def test_mmd_gaussian_is_zero_for_identical_samples() -> None:
    x = np.random.default_rng(6).standard_normal((40, 2))
    assert mmd(x, x.copy(), kernel='gaussian') == pytest.approx(0.0, abs=1e-8)


def test_subsample_estimator_runs_for_energydistance() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, 2))
    y = rng.standard_normal((200, 2)) + 0.3
    value = energydistance(x, y, estimator=Subsample(50), seed=1)
    assert value > 0


def test_random_slices_estimator_runs_for_energydistance() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal((200, 2))
    y = rng.standard_normal((200, 2)) + 0.3
    value = energydistance(x, y, estimator=RandomSlices(16), seed=1)
    assert value > 0


def test_random_features_estimator_runs_for_gaussian_mmd() -> None:
    rng = np.random.default_rng(9)
    x = rng.standard_normal((200, 2))
    y = rng.standard_normal((200, 2)) + 0.3
    value = mmd(x, y, kernel='gaussian', estimator=RandomFeatures(128), seed=1)
    assert value >= 0


def test_random_features_with_energy_kernel_raises_value_error() -> None:
    x = np.random.default_rng(10).standard_normal((50, 2))
    y = np.random.default_rng(11).standard_normal((50, 2))
    with pytest.raises(ValueError, match='RandomFeatures'):
        energydistance(x, y, estimator=RandomFeatures(10))


def test_mismatched_column_counts_raise_value_error() -> None:
    x = np.random.default_rng(12).standard_normal((30, 2))
    y = np.random.default_rng(13).standard_normal((30, 3))
    with pytest.raises(ValueError, match='columns'):
        energydistance(x, y)
