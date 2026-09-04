"""Tests for the time-series window-flattening example helpers.

The helpers live in `examples/time_series_windows.py`, which is not an
installed module (examples are not packaged), so it is imported by file
path. Most tests are deterministic and need no Julia; one smoke test
exercises `select_rows` and needs the dev Julia project wired by
`conftest.py`.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from splitiq import select_rows

if TYPE_CHECKING:
    from types import ModuleType

_MODULE_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'time_series_windows.py'


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('time_series_windows', _MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        message = f'could not load module spec from {_MODULE_PATH}'
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tsw = _load_module()

# Shared fixture (same literals as the Julia test and the design doc):
#   X = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]  # N = 5, p = 2
#   L = 2, stride = 2 -> 2 windows, starts = [0, 2], row index 4 dropped
FIXTURE_X = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
FIXTURE_L = 2
FIXTURE_STRIDE = 2
FIXTURE_Z = [[1, 2, 10, 20], [3, 4, 30, 40]]
FIXTURE_STARTS = [0, 2]


def test_window_count_matches_fixture() -> None:
    assert tsw.window_count(5, FIXTURE_L, FIXTURE_STRIDE) == 2


def test_window_count_zero_when_n_rows_below_length() -> None:
    assert tsw.window_count(3, 5, 1) == 0


def test_window_count_invalid_length_raises() -> None:
    with pytest.raises(ValueError, match='length'):
        tsw.window_count(5, 0, 1)


def test_window_count_invalid_stride_raises() -> None:
    with pytest.raises(ValueError, match='stride'):
        tsw.window_count(5, 2, 0)


def test_windows_fixture_flattens_variable_major() -> None:
    z, starts = tsw.windows(FIXTURE_X, FIXTURE_L, FIXTURE_STRIDE)
    assert np.array_equal(z, np.array(FIXTURE_Z, dtype=float))
    assert np.array_equal(starts, np.array(FIXTURE_STARTS))
    assert starts.dtype.kind in {'i', 'u'}


def test_windows_default_stride_equals_length() -> None:
    z, starts = tsw.windows(FIXTURE_X, FIXTURE_L)
    assert np.array_equal(z, np.array(FIXTURE_Z, dtype=float))
    assert np.array_equal(starts, np.array(FIXTURE_STARTS))


def test_windows_n_rows_less_than_length_returns_empty() -> None:
    x = [[1, 10], [2, 20], [3, 30]]
    z, starts = tsw.windows(x, 5, 1)
    assert z.shape == (0, 10)
    assert starts.shape == (0,)


def test_windows_invalid_length_raises() -> None:
    with pytest.raises(ValueError, match='length'):
        tsw.windows(FIXTURE_X, 0, 1)


def test_windows_invalid_stride_raises() -> None:
    with pytest.raises(ValueError, match='stride'):
        tsw.windows(FIXTURE_X, 2, 0)


def test_windows_stride_equal_length_shares_no_observations() -> None:
    x = np.arange(8, dtype=float).reshape(8, 1)
    _z, starts = tsw.windows(x, 4, 4)
    assert starts.tolist() == [0, 4]
    # Consecutive windows must not overlap in source rows.
    for start, next_start in itertools.pairwise(starts):
        assert next_start - start >= 4


def test_recover_window_matches_slice_and_reshape() -> None:
    x = np.array(FIXTURE_X, dtype=float)
    z, starts = tsw.windows(FIXTURE_X, FIXTURE_L, FIXTURE_STRIDE)
    for i, start in enumerate(starts):
        recovered = tsw.recover_window(x, int(start), FIXTURE_L)
        assert np.array_equal(recovered, x[start : start + FIXTURE_L])
        reshaped = z[i].reshape((FIXTURE_L, 2), order='F')
        assert np.array_equal(reshaped, recovered)


def test_standardize_by_variable_fit_stats_and_reapply() -> None:
    rng = np.random.default_rng(0)
    length, p = 4, 2
    z = rng.standard_normal((50, length * p)) * np.array([3.0, 5.0]).repeat(length) + np.array(
        [10.0, -2.0]
    ).repeat(length)
    zs, fit = tsw.standardize_by_variable(z, length, p)
    for v in range(p):
        block = zs[:, v * length : (v + 1) * length]
        assert block.mean() == pytest.approx(0.0, abs=1e-8)
        assert block.std() == pytest.approx(1.0, abs=1e-8)

    # Applying the same fit elsewhere must match a manual computation.
    subset = z[:5]
    zs_subset, fit_again = tsw.standardize_by_variable(subset, length, p, fit=fit)
    assert fit_again == fit
    means, stds = fit
    for v in range(p):
        sl = slice(v * length, (v + 1) * length)
        expected = (subset[:, sl] - means[v]) / stds[v]
        assert np.allclose(zs_subset[:, sl], expected)


def test_lag1_autocorrelation_constant_is_zero() -> None:
    z_row = np.array([5.0, 5.0, 5.0, 5.0])
    assert tsw.lag1_autocorrelation(z_row, 4, 1) == 0.0


def test_lag1_autocorrelation_exact_value() -> None:
    # y = [1, 2, 3, 4]; ybar = 2.5, dev = [-1.5, -0.5, 0.5, 1.5]
    # denom = sum(dev**2) = 5.0; numer = sum(dev[:-1] * dev[1:]) = 1.25
    # r1 = 1.25 / 5.0 = 0.25
    z_row = np.array([1.0, 2.0, 3.0, 4.0])
    assert tsw.lag1_autocorrelation(z_row, 4, 1) == pytest.approx(0.25)


def test_lag1_autocorrelation_averages_over_variables() -> None:
    # Variable 0 constant -> 0.0; variable 1 is the exact sequence above -> 0.25.
    z_row = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0])
    assert tsw.lag1_autocorrelation(z_row, 4, 2) == pytest.approx(0.125)


def test_two_regime_series_shape_and_labels() -> None:
    rng = np.random.default_rng(1)
    m, length, p = 10, 5, 3
    x, labels = tsw.two_regime_series(rng, m, length, p=p)
    assert x.shape == (m * length, p)
    assert len(labels) == m
    assert set(np.asarray(labels).tolist()) <= {'A', 'B'}


def test_two_regime_series_bit_identical_for_same_seed() -> None:
    x1, labels1 = tsw.two_regime_series(np.random.default_rng(42), 8, 6, p=2)
    x2, labels2 = tsw.two_regime_series(np.random.default_rng(42), 8, 6, p=2)
    assert np.array_equal(x1, x2)
    assert np.array_equal(np.asarray(labels1), np.asarray(labels2))


def test_select_rows_smoke_on_windows() -> None:
    """`select_rows` recovers a distinct-index selection over flattened windows."""
    rng = np.random.default_rng(7)
    m, length, p, n = 20, 4, 2, 5
    x, _labels = tsw.two_regime_series(rng, m, length, p=p)
    z, starts = tsw.windows(x, length, length)
    zs, _fit = tsw.standardize_by_variable(z, length, p)

    idx = select_rows(zs, n, method='twinning', standardize=False)

    assert len(set(idx.tolist())) == n
    for i in idx.tolist():
        recovered = tsw.recover_window(x, int(starts[i]), length)
        assert np.array_equal(recovered, x[starts[i] : starts[i] + length])
