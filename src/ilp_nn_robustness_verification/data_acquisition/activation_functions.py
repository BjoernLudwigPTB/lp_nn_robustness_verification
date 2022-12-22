"""The activation functions ready to be used in the optimization problem"""

__all__ = ["sigmoid", "Sigmoid", "sigmoid_prime"]

import numpy as np

from ..data_types import ActivationFunc


def sigmoid(val: float) -> float:
    r"""Real-valued implementation of :math:`\sigma (x) := \frac{1}{1 + e^{-x}}`"""
    return float(1 / (1 + np.exp(-val)))


def sigmoid_prime(val: float) -> float:
    r"""Real-valued implementation of :math:`\sigma'(x):=\frac{e^x}{(1+e^{x})^2}`"""
    return float(sigmoid(val) * (1 - sigmoid(val)))


Sigmoid = ActivationFunc(sigmoid, sigmoid_prime)
"""Provides an interface to the sigmoid activation function and its derivative

func : :data:`~ilp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the real-valued :func:`sigmoid` activation function
deriv : :data:`~ilp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the first derivative :func:`sigmoid_prime` of the real-valued activation function
"""
