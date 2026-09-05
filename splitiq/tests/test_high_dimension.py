"""Parity check for high-dimensional selection.

12,288 columns matches a common LLM embedding width, where the underlying search
structure switches away from a k-d tree.
"""

from __future__ import annotations

import numpy as np

from splitiq import select_rows


def test_twinning_selects_distinct_rows_at_high_dimension() -> None:
    x = np.random.default_rng(0).standard_normal((60, 12_288))
    idx = select_rows(x, 6, method='twinning', standardize=False)
    assert len(set(idx.tolist())) == 6
    assert all(i in range(60) for i in idx.tolist())


def test_support_points_selects_distinct_rows_at_high_dimension() -> None:
    x = np.random.default_rng(0).standard_normal((60, 12_288))
    idx = select_rows(
        x,
        6,
        method='support_points',
        kappa=20,
        max_iterations=3,
        seed=1,
        standardize=False,
    )
    assert len(set(idx.tolist())) == 6
    assert all(i in range(60) for i in idx.tolist())
