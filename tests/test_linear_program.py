from typing import Callable

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from numpy.ma.testutils import assert_almost_equal
from numpy.testing import assert_equal
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
from lp_nn_robustness_verification.examples.data_types import (
    IndexAndSeed,
    ScalerAndLayers,
)
from lp_nn_robustness_verification.linear_program import (
    RobustLU,
    RobustnessVerification,
)
from lp_nn_robustness_verification.pre_processing import LinearInclusion


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
    [int, int, int, int], tuple[UncertainInputs, LinearInclusion]
]:
    def values_and_linear_inclusion(
        size_scaler: int, depth: int, sample_idx: int, seed: int
    ) -> tuple[UncertainInputs, LinearInclusion]:
        zema_data = ZeMASamples(4766, size_scaler, True)
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
        return uncertain_inputs, linear_inclusion

    return values_and_linear_inclusion


@pytest.fixture(scope="session")
def solvable_instances() -> tuple[tuple[ScalerAndLayers, IndexAndSeed], ...]:
    return (
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 17937)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 31593)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 9216)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 40871)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 67664)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 49229)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 120758)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 111888)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 36800)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 94922)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 89693)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 27315)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 14371)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 6836)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 107744)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 221)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 76086)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 58572)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 116665)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 45325)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 531236)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 544667)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 576150)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 500317)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 509270)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 593765)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 98315)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 71508)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 103056)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 504794)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 585294)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 54127)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 522316)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 80986)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 602888)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 513645)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 580590)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 22764)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 85338)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 567132)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 612284)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 549300)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 63877)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 553780)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 620725)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 562595)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 536153)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 616796)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 527349)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 541222)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 589629)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 518503)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 608252)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 559116)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 572797)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 599560)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 450878)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 424190)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 477785)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 411004)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 482296)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 486593)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 419805)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 469012)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 379604)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 455779)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 446401)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 428576)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 392898)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 374977)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 442269)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 437676)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 495732)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 415370)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 465018)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 433890)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 406920)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 402615)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 384790)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 473924)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 492118)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 389318)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 460770)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 399065)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 799321)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 776911)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 758937)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 861623)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 803530)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 808067)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 870756)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 825935)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 790286)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 785814)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 857367)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 754645)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 768074)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 812739)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 839977)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 844343)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 795019)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 835088)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 848581)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 781840)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 830783)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 764033)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 853404)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 772881)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 751218)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 822636)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 669737)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 687509)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 867358)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 660820)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 718780)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 818806)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 692178)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 674144)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 683147)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 638829)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 714633)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 741168)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 710014)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 678677)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 732572)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 629969)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 723881)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 728136)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 652566)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 705877)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 701803)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 666020)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 657194)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 746809)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 737784)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 635032)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 648879)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 627045)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 645065)),
        (ScalerAndLayers(1, 1), IndexAndSeed(0, 188157)),
        (ScalerAndLayers(1, 3), IndexAndSeed(0, 80986)),
        (ScalerAndLayers(1, 3), IndexAndSeed(0, 415370)),
        (ScalerAndLayers(1, 3), IndexAndSeed(0, 586926)),
        (ScalerAndLayers(1, 5), IndexAndSeed(0, 80986)),
        (ScalerAndLayers(1, 5), IndexAndSeed(0, 586926)),
        (ScalerAndLayers(1, 8), IndexAndSeed(0, 80986)),
        (ScalerAndLayers(1, 8), IndexAndSeed(0, 586926)),
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


def test_robustness_verification_solves_to_known_value_for_fixture(
    custom_linear_inclusion: LinearInclusion,
) -> None:
    optimization = RobustLU(custom_linear_inclusion)
    optimization.model.hideOutput()
    optimization.solve()
    assert_equal(optimization.model.getObjVal(), 1.0)


def test_robustness_verification_solves_to_known_value_for_custom_problem() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([1.5, 0.5]), np.array([0.5, 0.6]))),
        Identity,
        NNParams(),
    )
    optimization = RobustnessVerification(linear_inclusion)
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), -0.1
    )


def test_robustness_verification_solves_to_known_negative_value_with_sigmoid() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([-1.2, 0.2]), np.array([0.5, 1.0]))),
        Sigmoid,
        NNParams(),
    )
    optimization = RobustnessVerification(linear_inclusion)
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), -0.021786708959446344
    )


def test_robustness_verification_solves_to_known_value_with_sigmoid() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([1.0, 0.5]), np.array([0.2, 0.1]))),
        Sigmoid,
        NNParams(),
    )
    optimization = RobustnessVerification(linear_inclusion)
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), 0.04431817490181711
    )


def test_robustness_verification_solves_to_known_value_with_quadlu() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([2.0, 1.5]), np.array([0.2, 0.1]))),
        QuadLU,
        NNParams(),
    )
    optimization = RobustnessVerification(linear_inclusion)
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), 0.2
    )


def test_robustness_verification_solves_to_known_negative_value_with_quadlu() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([0.0, 0.25]), np.array([0.1, 0.25]))),
        QuadLU,
        NNParams(),
    )
    optimization = RobustnessVerification(linear_inclusion)
    optimization.solve()
    assert_almost_equal(  # type: ignore[no-untyped-call]
        optimization.model.getObjVal(), -0.06
    )


def test_robustness_verification_solve_solves_known_to_work_instances(
    custom_linear_inclusion: LinearInclusion,
    solvable_instances: tuple[tuple[ScalerAndLayers, IndexAndSeed], ...],
    compute_linear_inclusion_for_instance: Callable[
        [int, int, int, int], tuple[UncertainInputs, LinearInclusion]
    ],
    capfd: CaptureFixture[str],
) -> None:
    for (size_scaler, depth), (sample_idx, seed) in solvable_instances:
        uncertain_inputs, linear_inclusion = compute_linear_inclusion_for_instance(
            size_scaler, depth, sample_idx, seed
        )
        RobustnessVerification(linear_inclusion).solve()
        assert (
            "SCIP Status        : problem is solved [optimal solution found]"
            in capfd.readouterr().out
        )


def test_robust_lu_solve_solves_known_to_work_instances(
    custom_linear_inclusion: LinearInclusion,
    solvable_instances: tuple[tuple[ScalerAndLayers, IndexAndSeed], ...],
    compute_linear_inclusion_for_instance: Callable[
        [int, int, int, int], tuple[UncertainInputs, LinearInclusion]
    ],
    capfd: CaptureFixture[str],
) -> None:
    for (size_scaler, depth), (sample_idx, seed) in solvable_instances:
        uncertain_inputs, linear_inclusion = compute_linear_inclusion_for_instance(
            size_scaler, depth, sample_idx, seed
        )
        RobustLU(linear_inclusion).solve()
        assert (
            "SCIP Status        : problem is solved [optimal solution found]"
            in capfd.readouterr().out
        )


def test_robustness_verification_original_differs_from_adapted(
    custom_linear_inclusion: LinearInclusion,
    solvable_instances: tuple[tuple[ScalerAndLayers, IndexAndSeed], ...],
    compute_linear_inclusion_for_instance: Callable[
        [int, int, int, int], tuple[UncertainInputs, LinearInclusion]
    ],
    capfd: CaptureFixture[str],
) -> None:
    gap = np.empty(len(solvable_instances))
    for idx, ((size_scaler, depth), (sample_idx, seed)) in enumerate(
        solvable_instances
    ):
        uncertain_inputs, linear_inclusion = compute_linear_inclusion_for_instance(
            size_scaler, depth, sample_idx, seed
        )
        adapted = RobustnessVerification(linear_inclusion)
        adapted.solve()
        original = RobustLU(linear_inclusion)
        original.solve()
        original_objective_value = original.model.getObjVal()
        adapted_objective_value = adapted.model.getObjVal()
        # assert original_objective_value > 0
        # assert adapted_objective_value > 0
        gap[idx] = original_objective_value - adapted_objective_value
        assert gap[-1] >= 0
    print(gap)
