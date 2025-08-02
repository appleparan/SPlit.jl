"""
Basic usage examples for splitiq package
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
import splitiq


def example_classification_data():
    """Example with classification dataset"""
    print("=== Classification Data Example ===")

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )

    print(f"Dataset shape: {X.shape}")

    # Split data using optimal support points
    train_idx, test_idx = splitiq.split_data(X, split_ratio=0.2, random_seed=42)

    print(f"Training set size: {len(train_idx)}")
    print(f"Test set size: {len(test_idx)}")

    # Get the actual data splits
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def example_regression_data():
    """Example with regression dataset"""
    print("\n=== Regression Data Example ===")

    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=500,
        n_features=3,
        noise=0.1,
        random_state=42
    )

    print(f"Dataset shape: {X.shape}")

    # Find optimal split ratio
    optimal_ratio = splitiq.optimal_split_ratio(X, y, method="simple")
    print(f"Optimal split ratio: {optimal_ratio:.3f}")

    # Split using optimal ratio
    train_idx, test_idx = splitiq.split_data(
        X,
        split_ratio=optimal_ratio,
        random_seed=42
    )

    print(f"Training set size: {len(train_idx)}")
    print(f"Test set size: {len(test_idx)}")

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def example_support_points():
    """Example of computing support points"""
    print("\n=== Support Points Example ===")

    # Generate 2D data for visualization
    np.random.seed(42)
    X = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, 0.5], [0.5, 1]],
        size=200
    )

    print(f"Data shape: {X.shape}")

    # Compute support points
    support_points = splitiq.compute_support_points(
        X,
        n_points=20,
        max_iterations=100
    )

    print(f"Support points shape: {support_points.shape}")

    # Plot if matplotlib is available
    try:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Original data')
        plt.title('Original Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Original data')
        plt.scatter(support_points[:, 0], support_points[:, 1],
                   c='red', s=100, alpha=0.8, label='Support points')
        plt.title('Data with Support Points')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        plt.tight_layout()
        plt.savefig('support_points_example.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'support_points_example.png'")

    except ImportError:
        print("Matplotlib not available, skipping plot")

    return X, support_points


def example_stochastic_optimization():
    """Example with large dataset using stochastic optimization"""
    print("\n=== Stochastic Optimization Example ===")

    # Generate large dataset
    np.random.seed(42)
    X = np.random.randn(5000, 10)

    print(f"Large dataset shape: {X.shape}")

    # Use stochastic optimization with kappa parameter
    train_idx, test_idx = splitiq.split_data(
        X,
        split_ratio=0.15,
        kappa=1000,  # Use subset of 1000 for optimization
        max_iterations=200,
        random_seed=42
    )

    print(f"Training set size: {len(train_idx)}")
    print(f"Test set size: {len(test_idx)}")
    print(f"Used stochastic optimization with kappa=1000")

    return X[train_idx], X[test_idx]


def compare_with_random_split():
    """Compare SPlit with random splitting"""
    print("\n=== Comparison with Random Split ===")

    # Generate data with clear structure
    np.random.seed(42)
    X1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 300)
    X2 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], 300)
    X = np.vstack([X1, X2])

    # SPlit method
    train_idx_split, test_idx_split = splitiq.split_data(
        X, split_ratio=0.2, random_seed=42
    )

    # Random split for comparison
    np.random.seed(42)
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    n_test = int(0.2 * len(X))
    test_idx_random = all_indices[:n_test]
    train_idx_random = all_indices[n_test:]

    print(f"SPlit - Train: {len(train_idx_split)}, Test: {len(test_idx_split)}")
    print(f"Random - Train: {len(train_idx_random)}, Test: {len(test_idx_random)}")

    # Calculate means for comparison
    X_train_split = X[train_idx_split]
    X_test_split = X[test_idx_split]
    X_train_random = X[train_idx_random]
    X_test_random = X[test_idx_random]

    print("\nMean differences (train vs test):")
    print(f"SPlit method: {np.abs(X_train_split.mean(axis=0) - X_test_split.mean(axis=0))}")
    print(f"Random split: {np.abs(X_train_random.mean(axis=0) - X_test_random.mean(axis=0))}")


if __name__ == "__main__":
    print("Running splitiq examples...")

    try:
        example_classification_data()
        example_regression_data()
        example_support_points()
        example_stochastic_optimization()
        compare_with_random_split()

        print("\n=== All examples completed successfully! ===")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure Julia and the SPlit.jl package are properly installed.")
