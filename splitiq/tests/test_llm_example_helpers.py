"""Tests for the LLM data-selection example's pure helpers.

The helpers live in `examples/llm_data_selection.py`, which is not an
installed module (examples are not packaged), so it is imported by file
path, following the same pattern as `test_time_series_windows.py`. These
tests need neither Julia nor network access: they exercise cosine
normalization, quality-proxy clipping, `cs`-category detection, and
k-center greedy, all pure-NumPy helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

_MODULE_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'llm_data_selection.py'


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('llm_data_selection', _MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        message = f'could not load module spec from {_MODULE_PATH}'
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llm = _load_module()


# ---------------------------------------------------------------------------
# cosine_normalize
# ---------------------------------------------------------------------------


def test_cosine_normalize_unit_rows() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
    normalized = llm.cosine_normalize(x)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)
    assert np.allclose(normalized[0], [0.6, 0.8])


def test_cosine_normalize_zero_row_unchanged() -> None:
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    normalized = llm.cosine_normalize(x)
    assert np.array_equal(normalized[0], [0.0, 0.0])
    assert np.allclose(normalized[1], [0.6, 0.8])


def test_cosine_normalize_rejects_1d() -> None:
    with pytest.raises(ValueError, match='2-D'):
        llm.cosine_normalize(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# clip_quantile
# ---------------------------------------------------------------------------


def test_clip_quantile_caps_upper_tail() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    clipped = llm.clip_quantile(values, 0.8)
    threshold = float(np.quantile(values, 0.8))
    assert clipped.max() == pytest.approx(threshold)
    assert np.array_equal(clipped[:4], values[:4])


def test_clip_quantile_full_quantile_is_max() -> None:
    values = np.array([5.0, 1.0, 9.0, 3.0])
    clipped = llm.clip_quantile(values, 1.0)
    assert np.array_equal(clipped, values)


def test_clip_quantile_invalid_q_raises() -> None:
    with pytest.raises(ValueError, match='q'):
        llm.clip_quantile(np.array([1.0, 2.0]), 1.5)


# ---------------------------------------------------------------------------
# has_category / category_mask
# ---------------------------------------------------------------------------


def test_has_category_true_and_false() -> None:
    assert llm.has_category(['cs', 'stat'], 'cs') is True
    assert llm.has_category(['math'], 'cs') is False


def test_has_category_none_is_false() -> None:
    assert llm.has_category(None, 'cs') is False


def test_category_mask_matches_expected_rows() -> None:
    column = [['cs', 'stat'], ['math'], None, ['cs']]
    mask = llm.category_mask(column, 'cs')
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, False, True]


# ---------------------------------------------------------------------------
# kcenter_greedy
# ---------------------------------------------------------------------------


def test_kcenter_greedy_known_answer_on_tiny_fixture() -> None:
    # np.random.default_rng(0).integers(4) == 3, so the start is row 3
    # (value 6.0). Step 1: distances to 6.0 are [6, 5, 1, 0] -> farthest is
    # row 0. Step 2: min(prev, distances to 0.0) = [0, 1, 1, 0] -> the first
    # tied max is row 1.
    x = np.array([[0.0], [1.0], [5.0], [6.0]])
    sel = llm.kcenter_greedy(x, 3, np.random.default_rng(0))
    assert sel.tolist() == [3, 0, 1]
    assert len(set(sel.tolist())) == 3


def test_kcenter_greedy_selects_all_rows_when_n_equals_n_rows() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    sel = llm.kcenter_greedy(x, 3, np.random.default_rng(1))
    assert sorted(sel.tolist()) == [0, 1, 2]


def test_kcenter_greedy_invalid_n_raises() -> None:
    x = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match='n must be in'):
        llm.kcenter_greedy(x, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match='n must be in'):
        llm.kcenter_greedy(x, 3, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# build_embedding_matrix / content_lengths (pure, easy to check alongside)
# ---------------------------------------------------------------------------


def test_build_embedding_matrix_fills_missing_with_zero() -> None:
    rows = [[1.0, 2.0], None, [0.0, None]]
    matrix = llm.build_embedding_matrix(rows, p=2)
    assert np.array_equal(matrix, np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]))


def test_content_lengths_matches_python_len() -> None:
    column = ['abc', '', 'abcdefgh']
    assert llm.content_lengths(column).tolist() == [3.0, 0.0, 8.0]
