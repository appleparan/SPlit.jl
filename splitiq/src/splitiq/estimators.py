"""Discrepancy estimators for `energydistance`, `mmd`, and `splitquality`.

Each estimator is a thin, immutable description of how the underlying
Julia computation should approximate (or evaluate exactly) a pairwise-kernel
quantity; `_to_julia` builds the corresponding Julia value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from splitiq._julia import JuliaValue


@dataclass(frozen=True)
class Exact:
    """Evaluate every pairwise term exactly (block-wise, threaded)."""

    def _to_julia(self, jl: JuliaValue) -> JuliaValue:
        """Build the corresponding Julia ``Exact`` value.

        Args:
            jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.

        Returns:
            A Julia ``Exact()`` value.
        """
        return jl.Exact()


@dataclass(frozen=True)
class Subsample:
    """Average the exact statistic over repeated random row subsets.

    Carries a positive bias of order ``1/m``; meant for comparing splits,
    not as an absolute value.

    Attributes:
        m: Subsample size drawn on each repeat.
        repeats: Number of random subsets to average over.
    """

    m: int
    repeats: int = 8

    def _to_julia(self, jl: JuliaValue) -> JuliaValue:
        """Build the corresponding Julia ``Subsample`` value.

        Args:
            jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.

        Returns:
            A Julia ``Subsample(m, repeats)`` value.
        """
        return jl.Subsample(self.m, self.repeats)


@dataclass(frozen=True)
class RandomSlices:
    """Sliced energy-distance estimator using random 1-D projections.

    Unbiased; defined for the energy kernel only.

    Attributes:
        k: Number of random projections.
    """

    k: int

    def _to_julia(self, jl: JuliaValue) -> JuliaValue:
        """Build the corresponding Julia ``RandomSlices`` value.

        Args:
            jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.

        Returns:
            A Julia ``RandomSlices(k)`` value.
        """
        return jl.RandomSlices(self.k)


@dataclass(frozen=True)
class RandomFeatures:
    """Random Fourier feature estimator for the Gaussian-kernel MMD.

    Unbiased; defined for the Gaussian kernel only.

    Attributes:
        D: Number of random Fourier features.
    """

    D: int

    def _to_julia(self, jl: JuliaValue) -> JuliaValue:
        """Build the corresponding Julia ``RandomFeatures`` value.

        Args:
            jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.

        Returns:
            A Julia ``RandomFeatures(D)`` value.
        """
        return jl.RandomFeatures(self.D)


DiscrepancyEstimator = Exact | Subsample | RandomSlices | RandomFeatures
