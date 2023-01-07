"""The activation functions ready to be used in the optimization problem"""

__all__ = ["sigmoid", "Sigmoid", "sigmoid_prime", "quadlu", "QuadLU", "quadlu_prime"]

import numpy as np

from ..data_types import ActivationFunc, RealVector


def sigmoid(val: np.float64 | RealVector) -> np.float64 | RealVector:
    r"""Real-valued implementation of :math:`\sigma (x) := \frac{1}{1 + e^{-x}}`"""
    return 1.0 / (1.0 + np.exp(-val))


def sigmoid_prime(val: np.float64 | RealVector) -> np.float64 | RealVector:
    r"""Real-valued implementation of :math:`\sigma'(x):=\frac{e^x}{(1+e^{x})^2}`"""
    return sigmoid(val) * (1.0 - sigmoid(val))


Sigmoid = ActivationFunc(sigmoid, sigmoid_prime)
"""Provides an interface to the sigmoid activation function and its derivative

func : :data:`~lp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the real-valued :func:`sigmoid` activation function
deriv : :data:`~lp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the first derivative :func:`sigmoid_prime` of the real-valued activation function
"""


def quadlu(
    val: np.float64 | RealVector, alpha: float = 0.25
) -> np.float64 | RealVector:
    r"""Real-valued implementation of :math:`\operatorname{QuaLU} (x)`

        :math:`\operatorname{QuadLU}_\alpha \colon \mathbb{R} \to \mathbb{R}` is defined
        as

        .. math::

            \operatorname{QuadLU}_\alpha (x) :=
            \begin{cases}
              0, &\quad \text{for } x \leq -\alpha \\
              (x + \alpha)^2, &\quad \text{for } -\alpha < x < \alpha \\
              4\alpha x, &\quad \text{for } x \geq \alpha \\
            \end{cases}

        with :math:`\alpha \in \mathbb{R}_+`.
    """
    less_or_equal_mask = val <= -alpha
    greater_or_equal_mask = val >= alpha
    if isinstance(val, float):
        if less_or_equal_mask:
            return 0.0
        if greater_or_equal_mask:
            return 4.0 * alpha * val
        return np.square(val + alpha)
    assert isinstance(val, np.ndarray)
    result = np.zeros_like(val)
    result[greater_or_equal_mask] = 4.0 * alpha * val[greater_or_equal_mask]
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result[in_between_mask] = np.square(val[in_between_mask] + alpha)
    return result


def quadlu_prime(
    val: np.float64 | RealVector, alpha: float = 0.25
) -> np.float64 | RealVector:
    r"""Real-valued implementation of :math:`\operatorname{QuaLU}' (x)"""
    if isinstance(val, float):
        if val <= -alpha:
            return 0
        if val >= alpha:
            return 4.0 * alpha
        return 2 * (val + alpha)
    assert isinstance(val, np.ndarray)
    result = np.zeros_like(val)
    less_or_equal_mask = val <= -alpha
    greater_or_equal_mask = val >= alpha
    result[greater_or_equal_mask] = 4.0 * alpha
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result[in_between_mask] = 2 * (val[in_between_mask] + alpha)
    return result


Sigmoid = ActivationFunc(sigmoid, sigmoid_prime)
"""Provides an interface to the sigmoid activation function and its derivative

func : :data:`~lp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the real-valued :func:`sigmoid` activation function
deriv : :data:`~lp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the first derivative :func:`sigmoid_prime` of the real-valued activation function
"""


QuadLU = ActivationFunc(quadlu, quadlu_prime)
"""Provides an interface to the QuadLU activation function and its derivative

func : :data:`~lp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the real-valued :func:`quadlu` activation function
deriv : :data:`~lp_nn_robustness_verification.type_aliases.RealScalarFunction`
    the first derivative :func:`quadlu_prime` of the real-valued activation function
"""
