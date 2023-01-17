"""This module contains type aliases for type hints and thus more convenient coding"""

__all__ = [
    "ActivationFunc",
    "Intervals",
    "IntervalCollection",
    "LayerIdx",
    "NeuronIdx",
    "NNParams",
    "RealMatrix",
    "RealVars",
    "RealVector",
    "RealScalarFunction",
    "UncertainArray",
    "ValidCombinationForZeMA",
    "VectorOfRealMatrices",
    "VectorOfRealVectors",
]

from dataclasses import dataclass
from typing import Callable, cast, Iterator, NamedTuple, TypeAlias

import numpy as np
from interval import interval
from numpy._typing import NDArray
from pyscipopt.scip import Variable  # type: ignore[import]

RealMatrix: TypeAlias = NDArray[np.float64]
"""A real matrix represented by a :class:`np.ndarray <numpy.ndarray>`"""
RealVector: TypeAlias = NDArray[np.float64]
"""A real vector represented by a :class:`np.ndarray <numpy.ndarray>`"""
VectorOfRealVectors: TypeAlias = tuple[NDArray[np.float64], ...]
"""A vector of real vectors represented by a :class:`np.ndarray <numpy.ndarray>`"""
VectorOfRealMatrices: TypeAlias = tuple[NDArray[np.float64], ...]
"""A vector of real matrices represented by a :class:`np.ndarray <numpy.ndarray>`"""
RealScalarFunction: TypeAlias = Callable[
    [np.float64 | RealVector], np.float64 | RealVector
]
"""A real-valued function with one real argument"""
Intervals: TypeAlias = tuple[interval, ...]
"""A tuple of intervals on the real number line each enabled for interval arithmetics"""
IntervalCollection: TypeAlias = tuple[Intervals, ...]
"""Tuple of tuples of intervals on the real number line interval arithmetics enabled"""
LayerIdx: TypeAlias = int
"""Index of a layer of a neural network"""
NeuronIdx: TypeAlias = int
"""Index of a neuron in a layer of a neural network"""
RealVars: TypeAlias = dict[tuple[LayerIdx, NeuronIdx], Variable]
"""The real variables in the linear optimization problem"""


class UncertainArray(NamedTuple):
    """A tuple of a tensor of values with a tensor of associated uncertainties"""

    values: RealVector
    """the corresponding values"""
    uncertainties: RealMatrix | RealVector
    """... and their associated uncertainties"""


class ActivationFunc(NamedTuple):
    """A representation of a real-valued, scalar function with its first derivative"""

    func: RealScalarFunction = lambda x: np.full(1, x) if isinstance(x, float) else x
    """the function itself"""
    deriv: RealScalarFunction = lambda x: np.ones(1)
    """the function's derivative"""


@dataclass
class NNParams:
    """A representation of a neural network's parameters"""

    biases: VectorOfRealVectors
    """The bias vectors of the neural network"""
    weights: VectorOfRealMatrices
    """The weights matrices of the neural network"""

    def __init__(
        self,
        biases: VectorOfRealVectors = (np.array([0.0, 0.0]),),
        weights: VectorOfRealMatrices = (np.array([[1.0, 0.0], [0.0, 1.0]]),),
    ):
        assert len(biases) == len(weights), (
            f"Somehow there are {len(biases)} of bias vectors and {len(weights)} "
            f"weight matrices but these numbers should be equal, one for each layer"
        )
        for bias_vector, weight_matrix in zip(biases, weights):
            assert len(weight_matrix.shape) == 2, (
                f"Somehow there is a weight array, that is not of the shape of a "
                f"matrix but has shape {weight_matrix.shape}"
            )
            assert len(bias_vector) == weight_matrix.shape[0], (
                f"Somehow one of the bias vectors' dimension ({bias_vector}), does "
                f"not match the dimension of its weight matrix ({weight_matrix})"
            )
        self.biases = biases
        self.weights = weights

    def __iter__(
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


class ValidCombinationForZeMA(NamedTuple):
    """A key tuple to mark collections of NumPy seeds known to produce feasible sets"""

    size_scaler: int
    """the size_scaler for extracting the ZeMA samples"""
    depth: int
    """the network depth used to create layer sizes

    the layer sizes are created using
    :func:`.generate_nn_params.construct_out_features_counts`
    """
    sample: int
    """the index of the sample to be used between 0 and 4765"""
