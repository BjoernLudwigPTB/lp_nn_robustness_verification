"""The actual implementation of the linear optimization problem."""

from numpy.testing import assert_equal
from pyscipopt import Model, quicksum  # type: ignore[import]

from lp_nn_robustness_verification.data_types import RealVars
from lp_nn_robustness_verification.pre_processing import (
    compute_values_label,
    LinearInclusion,
)


class RobustVerifier:
    """Instances of this class represent instances of the linear optimization problem

    For details see chapter 3 in [Ludwig2023]_.

    Parameters
    ----------
    linear_inclusion : LinearInclusion
        all parameters for the linear constraints and the input regions
    """

    linear_inclusion: LinearInclusion
    x_is: RealVars
    z_is: RealVars
    model: Model

    def __init__(self, linear_inclusion: LinearInclusion):
        """Crate instance of the optimization problem without considering remainders"""
        self.linear_inclusion = linear_inclusion
        self.x_is = {}
        self.z_is = {}
        self.model = Model("Robustness Verification (abstract base)")
        self._set_up_model()

    def _set_up_model(self) -> None:
        """Set up all the variables for the SCIP model"""
        self._add_vars_x_i_in_theta_i()
        self._add_vars_z_i()
        self._add_linear_cons()
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
        """Introduce z_is to the SCIP model"""
        for i_idx, (biases, weight_matrix) in enumerate(
            self.linear_inclusion.nn_params, start=1
        ):
            for k_idx, (bias, weight_vector) in enumerate(zip(biases, weight_matrix)):
                self.z_is[i_idx, k_idx] = self.model.addVar(
                    name=f"z_{k_idx}^({i_idx})", vtype="C", lb=None
                )
                self.model.addCons(
                    self.z_is[i_idx, k_idx]
                    == quicksum(
                        weight * self.x_is[i_idx - 1, j_idx]
                        for j_idx, weight in enumerate(weight_vector)
                    )
                    + bias,
                    f"z_{k_idx}^({i_idx})(x^({i_idx - 1}))",
                )

    def _add_linear_cons(self) -> None:
        """Introduce linear constraints to the SCIP model"""
        for layer_idx, (xi_i, r_i) in enumerate(
            zip(self.linear_inclusion.xi_is, self.linear_inclusion.r_is), start=1
        ):
            for neuron_idx, (xi_i_k, r_i_k) in enumerate(zip(xi_i, r_i)):
                linear_approximation = (
                    self.x_is[layer_idx, neuron_idx]
                    - float(self.linear_inclusion.activation.func(xi_i_k))
                    - float(self.linear_inclusion.activation.deriv(xi_i_k))
                    * (self.z_is[layer_idx, neuron_idx] - xi_i_k)
                )
                self.model.addCons(
                    linear_approximation - r_i_k[0].inf >= 0,
                    f"x_{neuron_idx}^({layer_idx}) upper half-space",
                )
                self.model.addCons(
                    linear_approximation - r_i_k[0].sup <= 0,
                    f"x_{neuron_idx}^({layer_idx}) lower half-space",
                )

    def _add_objective(self) -> None:
        """Introduce objective function to the SCIP model"""
        label = compute_values_label(
            self.linear_inclusion.uncertain_inputs,
            self.linear_inclusion.activation,
            self.linear_inclusion.nn_params,
        )
        for neuron_idx in range(len(self.linear_inclusion.theta[-1])):
            if neuron_idx != label:
                self.model.setObjective(
                    self.x_is[len(self.linear_inclusion.theta) - 1, label]
                    - self.x_is[len(self.linear_inclusion.theta) - 1, neuron_idx],
                    "minimize",
                )

    def solve(self) -> None:
        """Actually solve the optimization problem"""
        self.model.optimize()

    def visualize_solution(self) -> str:
        """Rudimentary visualize the optimization result on the console"""
        solution_assignments = []
        for x_i in self.x_is.values():
            solution_assignments.append(f"{x_i.name}: " f"{self.model.getVal(x_i)}")
        for layer_idx, r_i in enumerate(self.linear_inclusion.r_is, start=1):
            for neuron_idx, r_i_k in enumerate(r_i):
                solution_assignments.append(
                    f"inf r_{neuron_idx}^({layer_idx}): {r_i_k[0].inf}"
                )
                solution_assignments.append(
                    f"sup r_{neuron_idx}^({layer_idx}): {r_i_k[0].sup}"
                )
        solution_assignments.append("\n")
        return str(solution_assignments)
