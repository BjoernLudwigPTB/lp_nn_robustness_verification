"""This module contains type aliases for type hints and thus more convenient coding"""

__all__ = [
    "ActivationFunc",
    "Intervals",
    "IntervalCollection",
    "NNParams",
    "RealMatrix",
    "RealVector",
    "RealScalarFunction",
    "UncertainArray",
    "VectorOfRealMatrices",
    "VectorOfRealVectors",
]

from typing import Callable, cast, Iterator, NamedTuple, TypeAlias

import numpy as np
from interval import interval
from numpy._typing import NDArray

RealScalarFunction: TypeAlias = Callable[[float], float]
"""A real-valued function with one real argument"""
RealMatrix: TypeAlias = NDArray[np.float64]
"""A real matrix represented by a :class:`np.ndarray <numpy.ndarray>`"""
RealVector = NDArray[np.float64]
"""A real vector represented by a :class:`np.ndarray <numpy.ndarray>`"""
VectorOfRealVectors: TypeAlias = NDArray[np.float64]
"""A vector of real vectors represented by a :class:`np.ndarray <numpy.ndarray>`"""
VectorOfRealMatrices: TypeAlias = NDArray[np.float64]
"""A vector of real matrices represented by a :class:`np.ndarray <numpy.ndarray>`"""
Intervals: TypeAlias = tuple[interval, ...]
"""A tuple of intervals on the real number line each enabled for interval arithmetics"""
IntervalCollection: TypeAlias = tuple[Intervals, ...]
"""Tuple of tuples of intervals on the real number line interval arithmetics enabled"""


class UncertainArray(NamedTuple):
    """A tuple of a tensor of values with a tensor of associated uncertainties"""

    values: RealVector
    """the corresponding values"""
    uncertainties: RealMatrix | RealVector
    """... and their associated uncertainties"""


class ActivationFunc(NamedTuple):
    """A representation of a real-valued, scalar function with its first derivative"""

    func: RealScalarFunction = lambda x: x
    """the function itself"""
    deriv: RealScalarFunction = lambda x: 1
    """the function's derivative"""


class NNParams(NamedTuple):
    """A representation of a neural network's parameters"""

    biases: VectorOfRealVectors = np.array([[0.0, 0.0]])
    """The bias vectors of the neural network"""
    weights: VectorOfRealMatrices = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    """The weights matrices of the neural network"""

    def __iter__(  # type: ignore[override]
        self,
    ) -> Iterator[tuple[RealVector, RealMatrix]]:
        """Return an iterator over the biases and weights

        Examples
        --------
        nn_params = NNParams(
            np.array([[0.0, 0.0]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])
        )
        for biases, weights in nn_params:
            print(biases, weights)
        """
        return cast(
            Iterator[tuple[RealVector, RealMatrix]], zip(self.biases, self.weights)
        )
