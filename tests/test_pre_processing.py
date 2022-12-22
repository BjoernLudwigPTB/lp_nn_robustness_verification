import numpy as np
import pytest
from interval import interval
from numpy.testing import assert_equal

from ilp_nn_robustness_verification.data_acquisition.activation_functions import Sigmoid
from ilp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from ilp_nn_robustness_verification.pre_processing import LinearInclusion
from ilp_nn_robustness_verification.data_types import NNParams, UncertainArray


@pytest.fixture(scope="session")
def linear_inclusion_instance() -> LinearInclusion:
    return LinearInclusion(activation=Sigmoid)


def test_default_linear_inclusion_has_z_is_tuple(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert isinstance(linear_inclusion_instance.z_is, tuple)


def test_default_linear_inclusion_has_z_is_tuple_of_tuple(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    for intervall_collection in linear_inclusion_instance.z_is:
        assert isinstance(intervall_collection, tuple)


def test_default_linear_inclusion_has_z_is_tuple_of_tuple_of_intervals(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    for intervall_collection in linear_inclusion_instance.z_is:
        for interv in intervall_collection:
            assert isinstance(interv, interval)


def test_default_linear_inclusion_z_is_are_correct(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert linear_inclusion_instance.z_is == (
        (
            interval([0.0, 1.0]),
            interval([0.0, 1.0]),
        ),
    )


def test_custom_linear_inclusion_z_is_are_correct() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([1.0, 2.0]), np.array([0.5, 1.0]))),
        Sigmoid,
        NNParams(
            biases=np.array([[1.5, 2.5]]), weights=np.array([[[2.0, 2.0], [3.0, 3.0]]])
        ),
    )
    assert linear_inclusion.z_is == ((interval([4.5, 10.5]), interval([7.0, 16.0])),)


def test_default_linear_inclusion_has_theta_tuple(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert isinstance(linear_inclusion_instance.theta, tuple)


def test_default_linear_inclusion_has_theta_tuple_of_tuple(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    for intervall_collection in linear_inclusion_instance.theta:
        assert isinstance(intervall_collection, tuple)


def test_default_linear_inclusion_has_theta_tuple_of_tuple_of_intervals(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    for intervall_collection in linear_inclusion_instance.theta:
        for interv in intervall_collection:
            assert isinstance(interv, interval)


def test_default_linear_inclusion_theta_are_correct(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert linear_inclusion_instance.theta == (
        (
            interval([0.0, 1.0]),
            interval([0.0, 1.0]),
        ),
        (
            interval([0.5, 0.7310585786300049]),
            interval([0.5, 0.7310585786300049]),
        ),
    )


def test_default_linear_inclusion_has_xi_is_ndarray(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert isinstance(linear_inclusion_instance.xi_is, np.ndarray)


def test_default_linear_inclusion_has_xi_is_of_length_similar_to_input_dimen(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert_equal(
        linear_inclusion_instance.xi_is.shape[1],
        len(linear_inclusion_instance.uncertain_inputs.uncertain_values.values),
    )


def test_default_linear_inclusion_xi_is_are_correct(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert_equal(linear_inclusion_instance.xi_is, np.array([[0.5, 0.5]]))


def test_custom_linear_inclusion_xi_is_are_correct() -> None:
    linear_inclusion = LinearInclusion(
        UncertainInputs(UncertainArray(np.array([1.0, 2.0]), np.array([0.5, 1.0]))),
        Sigmoid,
        NNParams(
            biases=np.array([[1.5, 2.5]]), weights=np.array([[[2.0, 2.0], [3.0, 3.0]]])
        ),
    )
    assert_equal(linear_inclusion.xi_is, np.array([[7.5, 11.5]]))


def test_default_linear_inclusion_has_r_is_tuple(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert isinstance(linear_inclusion_instance.r_is, tuple)


def test_default_linear_inclusion_has_r_is_tuple_of_tuple(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    for intervall_collection in linear_inclusion_instance.r_is:
        assert isinstance(intervall_collection, tuple)


def test_default_linear_inclusion_has_r_is_tuple_of_tuple_of_intervals(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    for intervall_collection in linear_inclusion_instance.r_is:
        for interv in intervall_collection:
            assert isinstance(interv, interval)


def test_default_linear_inclusion_r_is_are_correct(
    linear_inclusion_instance: LinearInclusion,
) -> None:
    assert linear_inclusion_instance.r_is == (
        (
            interval([-0.23996118730265184, 0.22610110352894755]),
            interval([-0.23996118730265184, 0.22610110352894755]),
        ),
    )
