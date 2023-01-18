"""The actual implementation of the linear optimization problem."""

from numpy.testing import assert_equal
from pyscipopt import Model, quicksum  # type: ignore[import]

from lp_nn_robustness_verification.data_types import RealVars, RI
from lp_nn_robustness_verification.pre_processing import (
    compute_values_label,
    LinearInclusion,
)


class GenericRobustnessVerification:
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

    def _add_objective(self) -> None:
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

    def visualize_solution(
        self, inf_r_is: list[RI] | None = None, sup_r_is: list[RI] | None = None
    ) -> str:
        """Rudimentary visualize the optimization result on the console"""
        solution_assignments = []
        for x_i in self.x_is.values():
            solution_assignments.append(f"{x_i.name}: " f"{self.model.getVal(x_i)}")
        for inf_r_i, sup_r_i in zip(inf_r_is, sup_r_is):
            solution_assignments.append(f"{inf_r_i.name}: " f"{inf_r_i.value}")
            solution_assignments.append(f"{sup_r_i.name}: " f"{sup_r_i.value}")
        solution_assignments.append("\n")
        return str(solution_assignments)


class RobustnessVerification(GenericRobustnessVerification):
    """This is the adapted version as described in chapter 3.2 in [Ludwig2023]_

    Parameters
    ----------
    linear_inclusion : LinearInclusion
        all parameters for the linear constraints and the actual values with
        uncertainties
    """

    inf_r_is: RealVars
    sup_r_is: RealVars

    def __init__(self, linear_inclusion: LinearInclusion):
        """Crate an instance of the optimization problem"""
        super().__init__(linear_inclusion)
        self.inf_r_is = {}
        self.sup_r_is = {}
        self.model.setProbName("Robustness Verification (adapted)")
        self._add_linear_cons()

    def _add_linear_cons(self) -> None:
        for layer_idx, xi_i in enumerate(self.linear_inclusion.xi_is, start=1):
            for neuron_idx, xi_i_k in enumerate(xi_i):
                self.inf_r_is[layer_idx, neuron_idx] = self.model.addVar(
                    name=f"inf r_{neuron_idx}^({layer_idx})", vtype="C", lb=None
                )
                self.sup_r_is[layer_idx, neuron_idx] = self.model.addVar(
                    name=f"sup r_{neuron_idx}^({layer_idx})", vtype="C", lb=None
                )
                linear_approximation = (
                    self.x_is[layer_idx, neuron_idx]
                    - float(self.linear_inclusion.activation.func(xi_i_k))
                    - float(self.linear_inclusion.activation.deriv(xi_i_k))
                    * (self.z_is[layer_idx, neuron_idx] - xi_i_k)
                )
                self.model.addCons(
                    linear_approximation - self.inf_r_is[layer_idx, neuron_idx] >= 0,
                    f"x_{neuron_idx}^({layer_idx}) upper half-space",
                )
                self.model.addCons(
                    linear_approximation - self.sup_r_is[layer_idx, neuron_idx] <= 0,
                    f"x_{neuron_idx}^({layer_idx}) lower half-space",
                )

    def visualize_solution(
        self, inf_r_is: list[RI] | None = None, sup_r_is: list[RI] | None = None
    ) -> str:
        """Rudimentary visualize the optimization result on the console"""
        if inf_r_is is None:
            inf_r_is: list[RI] = []
            for inf_r_i in self.inf_r_is.values():
                inf_r_is.append(RI(inf_r_i.name, self.model.getVal(inf_r_i)))
        if sup_r_is is None:
            sup_r_is: list[RI] = []
            for sup_r_i in self.sup_r_is.values():
                sup_r_is.append(RI(sup_r_i.name, self.model.getVal(sup_r_i)))
        return super().visualize_solution(inf_r_is, sup_r_is)


class RobustLU(GenericRobustnessVerification):
    """This is the original version as described in chapter 3.2 in [Ludwig2023]_

    Parameters
    ----------
    linear_inclusion : LinearInclusion
        all parameters for the linear constraints and the actual values with
        uncertainties
    """

    def __init__(self, linear_inclusion: LinearInclusion):
        """Crate an instance of the optimization problem"""
        super().__init__(linear_inclusion)
        self.model.setProbName("Robustness Verification (original)")
        self._add_linear_cons()

    def _add_linear_cons(self) -> None:
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

    def visualize_solution(
        self, inf_r_is: list[RI] | None = None, sup_r_is: list[RI] | None = None
    ) -> str:
        """Rudimentary visualize the optimization result on the console"""
        if inf_r_is is None or sup_r_is is None:
            inf_r_is = []
            sup_r_is = []
        for layer_idx, r_i in enumerate(self.linear_inclusion.r_is, start=1):
            for neuron_idx, r_i_k in enumerate(r_i):
                inf_r_is.append(RI(f"inf r_{neuron_idx}^({layer_idx})", r_i_k[0].inf))
                sup_r_is.append(RI(f"sup r_{neuron_idx}^({layer_idx})", r_i_k[0].sup))
        return super().visualize_solution(inf_r_is, sup_r_is)
