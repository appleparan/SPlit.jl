"""Parity tests for standardize=False in splitiq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splitiq import datasplit, energydistance, multiplet, select_rows, splitquality


def _embeddings(seed: int = 0, n: int = 300, p: int = 8) -> np.ndarray:
    m = np.random.default_rng(seed).standard_normal((n, p))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_standardize_false_changes_the_selection_and_keeps_a_partition() -> None:
    data = _embeddings(1)
    raw = datasplit(data, ratio=0.2, method='herding', kernel='energy', standardize=False)
    std = datasplit(data, ratio=0.2, method='herding', kernel='energy')
    assert sorted([*raw.train_indices.tolist(), *raw.test_indices.tolist()]) == list(range(300))
    assert not np.array_equal(raw.test_indices, std.test_indices)
    idx = select_rows(data, 60, method='twinning', standardize=False)
    assert len(set(idx.tolist())) == 60
    folds = multiplet(data, 3, method='twinning', standardize=False)
    assert sorted(np.concatenate(folds).tolist()) == list(range(300))


def test_splitquality_scores_raw_rows() -> None:
    # splitiq does not expose SPlit.jl's `compare`; this covers `splitquality`
    # scoring the raw (unstandardized) rows the brief's `compare` case intended.
    data = _embeddings(2)
    result = datasplit(data, ratio=0.2, method='herding', kernel='energy', standardize=False)
    q = splitquality(data, result, standardize=False)
    assert q == pytest.approx(energydistance(data[result.train_indices], data[result.test_indices]))


def test_dataframes_are_rejected_without_standardization() -> None:
    df = pd.DataFrame({'x': np.random.default_rng(3).standard_normal(50), 'g': ['a', 'b'] * 25})
    with pytest.raises(ValueError, match='standardize'):
        datasplit(df, method='herding', standardize=False)
