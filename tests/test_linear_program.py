from typing import Callable, NamedTuple

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from numpy.ma.testutils import assert_almost_equal
from numpy.testing import assert_equal
from zema_emc_annotated.data_types import SampleSize  # type: ignore[import]
from zema_emc_annotated.dataset import ZeMASamples  # type: ignore[import]

from lp_nn_robustness_verification.data_acquisition.activation_functions import (
    Identity,
    QuadLU,
    Sigmoid,
)
from lp_nn_robustness_verification.data_acquisition.generate_nn_params import (
    construct_out_features_counts,
    generate_weights_and_biases,
)
from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import NNParams, UncertainArray
from lp_nn_robustness_verification.linear_program import RobustVerifier
from lp_nn_robustness_verification.pre_processing import LinearInclusion


class ScalerAndLayers(NamedTuple):
    """A key tuple to mark collections of NumPy seeds known to produce feasible sets"""

    size_scaler: int
    """the size_scaler for extracting the ZeMA samples"""
    depth: int
    """the network depth used to create layer sizes

    the layer sizes are created using
    :func:`.generate_nn_params.construct_out_features_counts`
    """


class IndexAndSeed(NamedTuple):
    """A value tuple of a sample index and NumPy seed known to produce a feasible set"""

    sample_idx: int
    """the index of the sample in the ZeMA dataset"""
    seed: int
    """the NumPy seed to be used when creating the weight and bias matrices"""


@pytest.fixture(scope="session")
def custom_linear_inclusion() -> LinearInclusion:
    return LinearInclusion(
        uncertain_inputs=UncertainInputs(
            UncertainArray(np.array([6.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        ),
        activation=Identity,
        nn_params=NNParams(
            (np.array([0.0, 0.0]),),
            (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),),
        ),
    )


@pytest.fixture(scope="session")
def compute_linear_inclusion_for_instance() -> Callable[
    [int, int, int, int], LinearInclusion
]:
    def compute_linear_inclusion(
        size_scaler: int, depth: int, sample_idx: int, seed: int
    ) -> LinearInclusion:
        zema_data = ZeMASamples(
            SampleSize(n_cycles=4766, datapoints_per_cycle=size_scaler), normalize=True
        )
        uncertain_inputs = UncertainInputs(
            UncertainArray(
                zema_data.values[sample_idx], zema_data.uncertainties[sample_idx]
            )
        )
        linear_inclusion = LinearInclusion(
            uncertain_inputs,
            Sigmoid,
            generate_weights_and_biases(
                len(uncertain_inputs.values),
                construct_out_features_counts(
                    len(uncertain_inputs.values),
                    out_features=size_scaler * 11 - depth,
                    depth=depth,
                ),
                seed,
            ),
        )
        return linear_inclusion

    return compute_linear_inclusion


@pytest.fixture(scope="session")
def solvable_instances() -> list[tuple[ScalerAndLayers, IndexAndSeed]]:
    instances = []
    for depth in (1, 3):
        for sample_idx in range(10):
            for seed in range(10):
                instances.append(
                    (ScalerAndLayers(1, depth), IndexAndSeed(sample_idx, seed))
                )
    return instances


def test_robust_verifier_init(custom_linear_inclusion: LinearInclusion) -> None:
    assert RobustVerifier(custom_linear_inclusion)


def test_robust_verifier_solve_runs(
    custom_linear_inclusion: LinearInclusion, capfd: CaptureFixture[str]
) -> None:
    RobustVerifier(custom_linear_inclusion).solve()
    assert "SCIP Status        : problem" in capfd.readouterr().out


def test_robust_verifier_solve_solves_example(
    custom_linear_inclusion: LinearInclusion, capfd: CaptureFixture[str]
) -> None:
    RobustVerifier(custom_linear_inclusion).solve()
    assert (
        "SCIP Status        : problem is solved [optimal solution found]"
        in capfd.readouterr().out
    )


def test_robust_verifier_solves_to_known_value_for_fixture(
    custom_linear_inclusion: LinearInclusion,
) -> None:
    optimization = RobustVerifier(custom_linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_equal(optimization.model.getObjVal(), 1.0)


def test_robust_verifier_solves_to_known_value_for_custom_problem() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([1.5, 0.5]), np.array([0.5, 0.6]))),
        Identity,
        NNParams(),
    )
    optimization = RobustVerifier(linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), -0.1
    )


def test_robust_verifier_solves_to_known_negative_value_with_sigmoid() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([-1.2, 0.2]), np.array([0.5, 1.0]))),
        Sigmoid,
        NNParams(),
    )
    optimization = RobustVerifier(linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), -0.021786708959446344
    )


def test_robust_verifier_solves_to_known_value_with_sigmoid() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([1.0, 0.5]), np.array([0.2, 0.1]))),
        Sigmoid,
        NNParams(),
    )
    optimization = RobustVerifier(linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), 0.04431817490181711
    )


def test_robust_verifier_solves_to_known_value_with_quadlu() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([2.0, 1.5]), np.array([0.2, 0.1]))),
        QuadLU,
        NNParams(),
    )
    optimization = RobustVerifier(linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), 0.2
    )


def test_robust_verifier_solves_to_known_negative_value_with_quadlu() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([0.0, 0.25]), np.array([0.1, 0.25]))),
        QuadLU,
        NNParams(),
    )
    optimization = RobustVerifier(linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), -0.06
    )


def test_robust_verifier_solve_solves_known_to_work_instances(
    solvable_instances: list[tuple[ScalerAndLayers, IndexAndSeed]],
    compute_linear_inclusion_for_instance: Callable[
        [int, int, int, int], LinearInclusion
    ],
    capfd: CaptureFixture[str],
) -> None:
    for (size_scaler, depth), (sample_idx, seed) in solvable_instances:
        linear_inclusion = compute_linear_inclusion_for_instance(
            size_scaler, depth, sample_idx, seed
        )
        RobustVerifier(linear_inclusion).solve()
        assert (
            "SCIP Status        : problem is solved [optimal solution found]"
            in capfd.readouterr().out
        )
