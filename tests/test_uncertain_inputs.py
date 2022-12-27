import pytest

from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import UncertainArray


@pytest.fixture(scope="session")
def uncertain_inputs() -> UncertainInputs:
    return UncertainInputs()


def test_default_init_uncertain_inputs(uncertain_inputs: UncertainInputs) -> None:
    assert uncertain_inputs


def test_default_init_uncertain_inputs_has_values(
    uncertain_inputs: UncertainInputs,
) -> None:
    assert hasattr(uncertain_inputs, "uncertain_values")


def test_default_init_uncertain_inputs_values_are_uncertain_array(
    uncertain_inputs: UncertainInputs,
) -> None:
    assert isinstance(uncertain_inputs.uncertain_values, UncertainArray)


def test_default_init_uncertain_inputs_has_intervals(
    uncertain_inputs: UncertainInputs,
) -> None:
    assert hasattr(uncertain_inputs, "intervals")


def test_default_init_intervals_are_tuple(uncertain_inputs: UncertainInputs) -> None:
    assert isinstance(uncertain_inputs.intervals, tuple)
