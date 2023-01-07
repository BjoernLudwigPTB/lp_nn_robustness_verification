import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from numpy.testing import assert_equal

from lp_nn_robustness_verification.data_acquisition.activation_functions import Sigmoid
from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import NNParams, UncertainArray
from lp_nn_robustness_verification.linear_program import RobustnessVerification
from lp_nn_robustness_verification.pre_processing import LinearInclusion


@pytest.fixture(scope="session")
def custom_linear_inclusion() -> LinearInclusion:
    return LinearInclusion(
        uncertain_inputs=UncertainInputs(
            UncertainArray(np.array([0.0, 0.5, 1.0]), np.array([0.5, 0.5, 0.5]))
        ),
        activation=Sigmoid,
        nn_params=NNParams(
            (np.array([0.0, 0.0]),),
            (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),),
        ),
    )


def test_robustness_verification_init(custom_linear_inclusion: LinearInclusion) -> None:
    assert RobustnessVerification(custom_linear_inclusion)


def test_robustness_verification_solve_runs(
    custom_linear_inclusion: LinearInclusion, capfd: CaptureFixture[str]
) -> None:
    RobustnessVerification(custom_linear_inclusion).solve()
    assert "SCIP Status        : problem" in capfd.readouterr().out


def test_robustness_verification_solve_solves_example(
    custom_linear_inclusion: LinearInclusion, capfd: CaptureFixture[str]
) -> None:
    RobustnessVerification(custom_linear_inclusion).solve()
    assert (
        "SCIP Status        : problem is solved [optimal solution found]"
        in capfd.readouterr().out
    )


def test_robustness_verification_solves_to_known_value(
    custom_linear_inclusion: LinearInclusion,
) -> None:
    optimization = RobustnessVerification(custom_linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_equal(optimization.model.getObjVal(), 0.2310585786300049)
