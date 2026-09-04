"""Parity tests for `compare`/`SplitComparison` in splitiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from splitiq import compare, splitquality

if TYPE_CHECKING:
    from splitiq.split import SplitResult


def _data(seed: int = 0, n: int = 300, p: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, p))


def _selected_indices(result: SplitResult) -> np.ndarray:
    return result.test_indices if result.selected == 'test' else result.train_indices


def test_compare_two_methods_returns_index_aligned_results_and_qualities() -> None:
    data = _data(1)
    comparison = compare(data, ['support_points', 'herding'], ratio=0.2, seed=2)
    assert len(comparison.results) == 2
    assert len(comparison.qualities) == 2
    assert comparison.kernel == 'energy'
    for result in comparison.results:
        assert sorted([*result.train_indices.tolist(), *result.test_indices.tolist()]) == list(
            range(300)
        )
    assert all(np.isfinite(q) for q in comparison.qualities)
    assert comparison.results[0].method == 'support_points'
    assert comparison.results[1].method == 'herding'


def test_best_returns_the_argmin() -> None:
    data = _data(3)
    comparison = compare(data, ['support_points', 'herding', 'twinning'], ratio=0.2, seed=4)
    index, result = comparison.best()
    assert comparison.qualities[index] == min(comparison.qualities)
    assert result is comparison.results[index]


def test_mapping_spec_with_kernel_thinning_compress_never() -> None:
    data = _data(5)
    comparison = compare(
        data,
        ['herding', {'method': 'kernel_thinning', 'compress': 'never'}],
        ratio=0.2,
        seed=6,
    )
    assert comparison.results[1].method == 'kernel_thinning'


def test_qualities_equal_splitquality_under_the_same_scoring_kernel() -> None:
    data = _data(7)
    comparison = compare(data, ['support_points', 'herding'], ratio=0.2, kernel='energy', seed=8)
    for result, quality in zip(comparison.results, comparison.qualities, strict=True):
        assert quality == pytest.approx(splitquality(data, result, kernel='energy'))


def test_standardize_false_changes_qualities_for_cosine_normalized_rows() -> None:
    m = np.random.default_rng(9).standard_normal((300, 8))
    embeddings = m / np.linalg.norm(m, axis=1, keepdims=True)
    standardized = compare(embeddings, ['herding'], ratio=0.2, seed=10)
    raw = compare(embeddings, ['herding'], ratio=0.2, seed=10, standardize=False)
    assert standardized.qualities[0] != pytest.approx(raw.qualities[0])


def test_weights_are_forwarded_to_every_splitter() -> None:
    data = _data(11)
    weights = np.random.default_rng(12).random(len(data))
    comparison = compare(data, ['support_points', 'herding'], ratio=0.2, seed=13, weights=weights)
    for result in comparison.results:
        assert sorted([*result.train_indices.tolist(), *result.test_indices.tolist()]) == list(
            range(300)
        )


def test_reference_concentrates_the_selection() -> None:
    data = _data(14)
    ref = data[data[:, 0] > 0]
    plain = compare(data, ['herding'], ratio=0.2, seed=15)
    targeted = compare(data, ['herding'], ratio=0.2, seed=15, reference=ref)
    plain_idx = _selected_indices(plain.results[0])
    targeted_idx = _selected_indices(targeted.results[0])
    assert np.sum(data[targeted_idx, 0] > 0) > np.sum(data[plain_idx, 0] > 0)


def test_unknown_method_name_raises_value_error() -> None:
    data = _data(16)
    with pytest.raises(ValueError, match='method'):
        compare(data, ['not_a_method'])  # ty: ignore[invalid-argument-type]


def test_mapping_without_method_key_raises_value_error() -> None:
    data = _data(17)
    with pytest.raises(ValueError, match='method'):
        compare(data, [{'compress': 'never'}])


def test_weights_together_with_reference_raises_value_error() -> None:
    data = _data(18)
    ref = data[data[:, 0] > 0]
    with pytest.raises(ValueError, match='reference'):
        compare(data, ['herding'], weights=np.ones(len(data)), reference=ref)
