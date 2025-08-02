# splitiq

Python bindings for SPlit.jl - Optimal data splitting using support points method.

## Overview

splitiq provides Python access to the SPlit.jl package, which implements optimal data splitting procedures based on the method of support points as described in Joseph and Vakayil (2021).

## Installation

### Prerequisites

- Python e 3.9
- Julia e 1.10

### Install from PyPI

```bash
pip install splitiq
```

### Development Installation

```bash
git clone https://github.com/yourusername/SPlit.jl.git
cd SPlit.jl/splitiq
uv sync --dev
```

## Quick Start

```python
import numpy as np
import splitiq

# Generate sample data
X = np.random.randn(1000, 5)

# Split data using optimal support points method
train_idx, test_idx = splitiq.split_data(X, split_ratio=0.2)

# Use the indices to split your data
X_train, X_test = X[train_idx], X[test_idx]

# Find optimal split ratio
optimal_ratio = splitiq.optimal_split_ratio(X)
print(f"Optimal split ratio: {optimal_ratio}")

# Compute support points
support_points = splitiq.compute_support_points(X, n_points=50)
```

## API Reference

### `split_data(X, split_ratio=0.2, **kwargs)`

Split data using the optimal support points method.

**Parameters:**

- `X` (np.ndarray): Input data matrix of shape (n_samples, n_features)
- `split_ratio` (float): Proportion for test set (default: 0.2)
- `kappa` (int, optional): Subsample size for stochastic optimization
- `max_iterations` (int): Maximum iterations (default: 500)
- `tolerance` (float): Convergence tolerance (default: 1e-10)
- `n_threads` (int, optional): Number of parallel threads
- `random_seed` (int, optional): Random seed for reproducibility

**Returns:**

- `train_indices` (np.ndarray): Training set indices
- `test_indices` (np.ndarray): Test set indices

### `optimal_split_ratio(X, method="simple", **kwargs)`

Determine the optimal split ratio for a dataset.

**Parameters:**

- `X` (np.ndarray): Input data matrix
- `method` (str): Method to use ("simple" or "regression")
- `max_ratio` (float): Maximum ratio to consider (default: 0.5)
- `step_size` (float): Step size for search (default: 0.01)

**Returns:**

- `optimal_ratio` (float): Optimal split ratio

### `compute_support_points(X, n_points=None, **kwargs)`

Compute support points for the dataset.

**Parameters:**

- `X` (np.ndarray): Input data matrix
- `n_points` (int, optional): Number of support points
- `kappa` (int, optional): Subsample size for stochastic optimization
- `max_iterations` (int): Maximum iterations (default: 500)
- `tolerance` (float): Convergence tolerance (default: 1e-10)

**Returns:**

- `support_points` (np.ndarray): Computed support points

## How It Works

The SPlit method uses support points to create optimal data splits that:

1. **Preserve data distribution**: The training and test sets maintain similar statistical properties
2. **Minimize energy distance**: Reduces the energy distance between training and test distributions
3. **Scale efficiently**: Handles large datasets through stochastic optimization

## References

- Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. *Technometrics*, 63(4), 492-502.
- Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of Statistics*, 46(6A), 2562-2592.

## License

Apache-2.0

## Contributing

Contributions are welcome! Please see the [contributing guidelines](https://github.com/yourusername/SPlit.jl/blob/main/CONTRIBUTING.md).
