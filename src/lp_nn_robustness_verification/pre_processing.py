"""Performs all computations needed prior the solving the optimization problem"""

__all__ = ["LinearInclusion"]

from dataclasses import dataclass

import numpy as np
from interval import interval

from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import (
    ActivationFunc,
    IntervalCollection,
    Intervals,
    NNParams,
    VectorOfRealVectors,
)


@dataclass
class LinearInclusion:
    """Instances provide convenient access to the method of linear inclusion

    Parameters
    ----------
    uncertain_inputs: UncertainInputs, optional
        the values with associated uncertainties and resulting
        intervals, defaults to the default
        :class:`~.data_acquisition.uncertain_inputs.UncertainInputs` instance
    activation : ActivationFunc, optional
         the activation function and its derivative, defaults to the default
         :class:`~.type_aliases.ActivationFunc` instance
    nn_params : NNParams, optional
        the neural networks parameters, i.e. a tuple of bias vectors and weight
        matrices, defaults to the default
        :class:`~.type_aliases.NNParams` instance
    """

    uncertain_inputs: UncertainInputs
    activation: ActivationFunc
    nn_params: NNParams
    z_is: IntervalCollection
    theta: IntervalCollection
    xi_is: VectorOfRealVectors
    r_is: IntervalCollection

    def __init__(
        self,
        uncertain_inputs: UncertainInputs = UncertainInputs(),
        activation: ActivationFunc = ActivationFunc(),
        nn_params: NNParams = NNParams(),
    ):
        """Instantiate linear inclusion"""
        assert (
            len(uncertain_inputs.uncertain_values.values)
            == nn_params.weights[0].shape[1]
        ), (
            "Somehow the input values' and the first weight matrix' dimensions do not "
            "match"
        )
        self.uncertain_inputs = uncertain_inputs
        self.activation = activation
        self.nn_params = nn_params
        self._compute_z_is()
        self._compute_xi_is()
        self.theta = IntervalCollection(
            (uncertain_inputs.intervals,),
        )
        self._compute_and_append_theta_is()
        self._compute_r_is()

    def _compute_z_is(self) -> None:
        """Compute the result of the interval extension of the linear transformation

        For details see Equation 3.7 of Definition 3.2.10 in [Ludwig2023]_.
        """
        z_is = []
        for biases, weight_matrix in self.nn_params:
            z_i = []
            for k_idx, (bias, weight_vector) in enumerate(zip(biases, weight_matrix)):
                z_i.append(float(bias))
                for j_idx, weight in enumerate(weight_vector):
                    z_i[k_idx] += float(weight) * self.uncertain_inputs.intervals[j_idx]
            z_is.append(
                Intervals(
                    z_i,
                )
            )
        self.z_is = IntervalCollection(
            z_is,
        )

    def _compute_and_append_theta_is(self) -> None:
        r"""Compute the :math:`\Theta^{(i)}, i = 0, \ldots, \ell`

        For details see Equation 3.8 of Definition 3.2.10 in [Ludwig2023]_"""
        theta = list(self.theta)
        for z_i in self.z_is:
            theta_i = [
                interval[
                    self.activation.func(z_k[0].inf),
                    self.activation.func(z_k[0].sup),
                ]
                for z_k in z_i
            ]
            assert isinstance(theta_i, list), (
                f"Somehow one of the theta^(i)s ended up not being a list of "
                f"intervals but a {type(theta_i)} of intervals"
            )
            for element in theta_i:
                assert isinstance(element, interval), (
                    f"Somehow one of the components of theta ended up not being an "
                    f"interval but a {type(element)}"
                )
            theta.append(
                Intervals(
                    theta_i,
                )
            )
        self.theta = IntervalCollection(theta)

    def _compute_xi_is(self) -> None:
        """Compute the midpoints of the intervals of the linear transformation's results

        For details see Equation 3.9 of Definition 3.2.10 in [Ludwig2023]_.
        """
        xi_is = []
        for z_i in self.z_is:
            xi_ks = []
            for z_k in z_i:
                xi_ks.append(z_k.midpoint[0].inf)
            xi_is.append(np.array(xi_ks))
            assert len(xi_ks) == len(z_i), (
                f"Somehow there is not one xi_k^(i) for every of the {len(z_i)}, "
                f"z_k^(i), but only {len(xi_ks)}"
            )
        assert len(xi_is) == len(self.z_is), (
            f"Somehow there is not one xi^(i) for every of the {len(self.z_is)} z^(i), "
            f"but only {len(xi_is)}"
        )
        self.xi_is = tuple(xi_is)

    def _compute_r_is(self) -> None:
        """Compute the residual terms the linear inclusions linear approximation

        For details see Equation 3.10 of Definition 3.2.10 in [Ludwig2023]_.
        """
        r_is = []
        for xi_i, z_i in zip(self.xi_is, self.z_is):
            r_i = []
            for xi_k, z_k in zip(xi_i, z_i):
                r_i.append(self._taylors_residual(xi_k, z_k))
            assert isinstance(
                r_i[-1], interval
            ), f"Somehow r^(i) ended up not being an interval but a {type(r_i[-1])}"
            r_is.append(
                Intervals(
                    r_i,
                )
            )
        self.r_is = IntervalCollection(r_is)

    def _taylors_residual(self, xi_k: float, z_k: interval) -> interval:
        """Compute the residual term of the taylor approximation at the midpoint of z_k

        For details see Equation 3.10 of Definition 3.2.10 in [Ludwig2023]_.
        """
        return (
            interval[
                self.activation.func(z_k[0].inf),
                self.activation.func(z_k[0].sup),
            ]
            - float(self.activation.func(xi_k))  # type: ignore[arg-type]
            - float(self.activation.deriv(xi_k))  # type: ignore[arg-type]
            * (z_k - xi_k)
        )


def compute_values_label(
    uncertain_inputs: UncertainInputs = UncertainInputs(),
    activation: ActivationFunc = ActivationFunc(),
    nn_params: NNParams = NNParams(),
) -> int:
    """Compute the index of the maximum of the result vector of the values' forward pass

    Parameters
    ----------
    uncertain_inputs: UncertainInputs, optional
        the values with associated uncertainties and resulting
        intervals, defaults to the default
        :class:`~.data_acquisition.uncertain_inputs.UncertainInputs` instance
    activation : ActivationFunc, optional
         the activation function and its derivative, defaults to the default
         :class:`~.type_aliases.ActivationFunc` instance
    nn_params : NNParams, optional
        the neural networks parameters, i.e. a tuple of bias vectors and weight
        matrices, defaults to the default
        :class:`~.type_aliases.NNParams` instance
    """
    x_i = uncertain_inputs.uncertain_values.values
    for biases, weight_matrix in nn_params:
        x_i = activation.func(weight_matrix @ x_i + biases)  # type: ignore[assignment]
    return int(x_i.argmax())
