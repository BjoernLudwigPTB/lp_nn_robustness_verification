"""The actual implementation of the linear optimization problem."""
from itertools import chain

from numpy.testing import assert_equal
from pyscipopt import Model, quicksum  # type: ignore[import]

from lp_nn_robustness_verification.data_types import RealVars
from lp_nn_robustness_verification.pre_processing import (
    compute_values_label,
    LinearInclusion,
)


class RobustnessVerification:
    """Instances of this class represent instances of the linear optimization problem

    For details see chapter 3 in [Ludwig2023]_.

    Parameters
    ----------
    linear_inclusion : LinearInclusion
        all parameters for the linear constraints and the actual values with
        uncertainties
    """

    linear_inclusion: LinearInclusion
    x_is: RealVars
    z_is: RealVars
    model: Model

    def __init__(self, linear_inclusion: LinearInclusion):
        """Crate an instance of the optimization problem"""
        self.linear_inclusion = linear_inclusion
        self.label = compute_values_label(
            linear_inclusion.uncertain_inputs,
            linear_inclusion.activation,
            linear_inclusion.nn_params,
        )
        self.x_is = {}
        self.z_is = {}
        self.model = Model("Robustness Verification")
        self._set_up_model()

    def _set_up_model(self) -> None:
        """Set up all the variables for the SCIP model"""
        self.auxiliary_t = self.model.addVar(name="t", lb=None)
        self._add_vars_x_i_in_theta_i()
        self._add_vars_z_i()
        self._add_linear_cons()
        self._add_auxiliary_cons()
        self._add_objective()

    def _add_vars_x_i_in_theta_i(self) -> None:
        for i_idx, theta_i in enumerate(self.linear_inclusion.theta):
            for k_idx, theta_i_k in enumerate(theta_i):
                self.x_is[i_idx, k_idx] = self.model.addVar(
                    name=f"x_{k_idx}^({i_idx})",
                    vtype="C",
                    lb=theta_i_k[0].inf,
                    ub=theta_i_k[0].sup,
                )
        assert_equal(
            len(self.x_is),
            len(self.linear_inclusion.uncertain_inputs.uncertain_values.values)
            + sum(
                weight_matrix.shape[0]
                for weight_matrix in self.linear_inclusion.nn_params.weights
            ),
        )

    def _add_vars_z_i(self) -> None:
        for i_idx, (biases, weight_matrix) in enumerate(
            self.linear_inclusion.nn_params, start=1
        ):
            for k_idx, (bias, weight_vector) in enumerate(zip(biases, weight_matrix)):
                self.z_is[i_idx, k_idx] = self.model.addVar(
                    name=f"z_{k_idx}^({i_idx})", vtype="C"
                )
                self.model.addCons(
                    self.z_is[i_idx, k_idx]
                    == quicksum(
                        chain(
                            (
                                weight * self.x_is[i_idx - 1, j_idx]
                                for j_idx, weight in enumerate(weight_vector)
                            ),
                            (bias,),
                        )
                    ),
                    f"z_{k_idx}(x^({k_idx - 1}))",
                )

    def _add_linear_cons(self) -> None:
        for layer_idx, (xi_i, r_i) in enumerate(
            zip(self.linear_inclusion.xi_is, self.linear_inclusion.r_is), start=1
        ):
            for neuron_idx, (xi_i_k, r_i_k) in enumerate(zip(xi_i, r_i)):
                self.model.addCons(
                    self.x_is[layer_idx, neuron_idx]
                    - float(self.linear_inclusion.activation.func(xi_i_k))
                    - float(self.linear_inclusion.activation.deriv(xi_i_k))
                    * (self.z_is[layer_idx, neuron_idx] - xi_i_k)
                    - r_i_k[0].inf
                    >= 0
                )
                self.model.addCons(
                    self.x_is[layer_idx, neuron_idx]
                    - float(self.linear_inclusion.activation.func(xi_i_k))
                    - float(self.linear_inclusion.activation.deriv(xi_i_k))
                    * (self.z_is[layer_idx, neuron_idx] - xi_i_k)
                    - r_i_k[0].sup
                    <= 0
                )

    def _add_auxiliary_cons(self) -> None:
        for neuron_idx in range(len(self.linear_inclusion.theta[-1])):
            if neuron_idx != self.label:
                self.model.addCons(
                    self.x_is[len(self.linear_inclusion.theta) - 1, self.label]
                    - self.x_is[len(self.linear_inclusion.theta) - 1, neuron_idx]
                    >= self.auxiliary_t
                )

    def _add_objective(self) -> None:
        self.model.setObjective(self.auxiliary_t, "maximize")

    def solve(self, visualize: bool = True) -> str | None:
        """Actually solve the optimization problem

        Parameters
        ----------
        visualize : bool, optional
            If True (default), the result will be shown after finishing
        """
        self.model.optimize()
        if visualize and self.model.getSols():
            return self.visualize_solution()
        return None

    def visualize_solution(self) -> str:
        """Rudimentary visualize the optimization result on the console"""
        solution_assignments = []
        for (layer_idx, neuron_idx) in self.x_is:
            solution_assignments.append(
                f"x_{neuron_idx}^({layer_idx}): "
                f"{self.model.getVal(self.x_is[layer_idx, neuron_idx])}"
            )
        return str(solution_assignments)
