"""
Tests for splitiq core functionality
"""

import numpy as np
import pytest

import splitiq
from splitiq import InputValidationError, JuliaComputationError


class TestSplitData:
    """Test split_data function"""

    def test_basic_split(self):
        """Test basic data splitting"""
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)

        # Split data
        train_idx, test_idx = splitiq.split_data(X, split_ratio=0.2)

        # Check types
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

        # Check sizes
        assert len(train_idx) + len(test_idx) == len(X)
        assert len(test_idx) == pytest.approx(0.2 * len(X), abs=2)

        # Check no overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        # Check valid indices
        assert np.all(train_idx >= 0) and np.all(train_idx < len(X))
        assert np.all(test_idx >= 0) and np.all(test_idx < len(X))

    def test_different_split_ratios(self):
        """Test different split ratios"""
        X = np.random.randn(100, 3)

        for ratio in [0.1, 0.3, 0.5]:
            train_idx, test_idx = splitiq.split_data(X, split_ratio=ratio)
            expected_test_size = int(ratio * len(X))
            assert len(test_idx) == pytest.approx(expected_test_size, abs=2)

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        X = np.random.randn(50, 3)

        train1, test1 = splitiq.split_data(X, random_seed=123)
        train2, test2 = splitiq.split_data(X, random_seed=123)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Non-2D array
        with pytest.raises(InputValidationError):
            splitiq.split_data(np.array([1, 2, 3]))

        # Empty array
        with pytest.raises(InputValidationError):
            splitiq.split_data(np.empty((0, 5)))

        # Array with NaN
        X_nan = np.random.randn(10, 3)
        X_nan[0, 0] = np.nan
        with pytest.raises(InputValidationError):
            splitiq.split_data(X_nan)

    def test_edge_cases(self):
        """Test edge cases"""
        # Very small dataset
        X_small = np.random.randn(5, 2)
        train_idx, test_idx = splitiq.split_data(X_small, split_ratio=0.4)

        assert len(train_idx) + len(test_idx) == 5
        assert len(test_idx) <= 2  # 40% of 5 is 2

        # Single feature
        X_single = np.random.randn(20, 1)
        train_idx, test_idx = splitiq.split_data(X_single)

        assert len(train_idx) + len(test_idx) == 20


class TestOptimalSplitRatio:
    """Test optimal_split_ratio function"""

    def test_basic_ratio_computation(self):
        """Test basic ratio computation"""
        X = np.random.randn(100, 3)
        ratio = splitiq.optimal_split_ratio(X)

        assert isinstance(ratio, float)
        assert 0 < ratio < 0.5  # Should be reasonable

    def test_different_methods(self):
        """Test different methods"""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        ratio_simple = splitiq.optimal_split_ratio(X, method="simple")
        ratio_regression = splitiq.optimal_split_ratio(X, y, method="regression")

        assert isinstance(ratio_simple, float)
        assert isinstance(ratio_regression, float)
        assert 0 < ratio_simple < 1
        assert 0 < ratio_regression < 1


class TestComputeSupportPoints:
    """Test compute_support_points function"""

    def test_basic_support_points(self):
        """Test basic support points computation"""
        X = np.random.randn(50, 3)
        points = splitiq.compute_support_points(X, n_points=10)

        assert isinstance(points, np.ndarray)
        assert points.shape[0] == 10  # Number of support points
        assert points.shape[1] == X.shape[1]  # Same number of features

    def test_automatic_n_points(self):
        """Test automatic determination of number of support points"""
        X = np.random.randn(30, 3)
        points = splitiq.compute_support_points(X)

        assert isinstance(points, np.ndarray)
        assert points.shape[1] == X.shape[1]
