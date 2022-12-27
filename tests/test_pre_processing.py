import numpy as np
import pytest
from hypothesis import given, settings, strategies as hst
from hypothesis.extra import numpy as hnp
from interval import interval
from numpy.testing import assert_equal

from lp_nn_robustness_verification import pre_processing
from lp_nn_robustness_verification.data_acquisition.activation_functions import Sigmoid
from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import (
    NNParams,
    RealMatrix,
    RealVector,
    UncertainArray,
)
from lp_nn_robustness_verification.pre_processing import (
    compute_values_label,
    LinearInclusion,
)


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


def test_pre_processing_has_compute_values_label() -> None:
    assert hasattr(pre_processing, "compute_values_label")


def test_compute_values_label_is_correct() -> None:
    assert_equal(compute_values_label(), 0)


@given(
    hnp.arrays(
        np.float64,
        2,
        elements=hst.floats(min_value=-1e291, max_value=1e291),
    ),
    hnp.arrays(np.float64, 2),
)
def test_default_compute_values_label_for_random_input_is_correct(
    values: RealVector, uncertainties: RealVector
) -> None:
    assert_equal(
        compute_values_label(UncertainInputs(UncertainArray(values, uncertainties))),
        values.argmax(),
    )


@given(
    hnp.arrays(
        np.float64,
        2,
        elements=hst.floats(min_value=-1e145, max_value=1e145),
    ),
    hnp.arrays(np.float64, 2),
    hnp.arrays(np.float64, 2),
    hnp.arrays(
        np.float64,
        (2, 2),
        elements=hst.floats(min_value=-1e145, max_value=1e145),
    ),
)
def test_compute_values_label_for_random_input_and_random_params_is_correct(
    values: RealVector,
    uncertainties: RealVector,
    biases: RealVector,
    weights: RealMatrix,
) -> None:
    assert_equal(
        compute_values_label(
            UncertainInputs(UncertainArray(values, uncertainties)),
            nn_params=NNParams(biases[np.newaxis, ...], weights[np.newaxis, ...]),
        ),
        (weights @ values + biases).argmax(),
    )


@given(
    hnp.arrays(
        np.float64,
        11,
        elements=hst.floats(min_value=-1e145, max_value=1e145),
    ),
    hnp.arrays(np.float64, 11, elements=hst.floats(allow_nan=False)),
    hnp.arrays(np.float64, 3, elements=hst.floats(allow_nan=False)),
    hnp.arrays(
        np.float64,
        (3, 11),
        elements=hst.floats(min_value=-1e145, max_value=1e145),
    ),
)
@settings(deadline=None)
def test_compute_values_label_for_random_inputs_of_zema_shape_is_correct(
    values: RealVector,
    uncertainties: RealVector,
    biases: RealVector,
    weights: RealMatrix,
) -> None:
    assert_equal(
        compute_values_label(
            UncertainInputs(UncertainArray(values, uncertainties)),
            nn_params=NNParams(biases[np.newaxis, ...], weights[np.newaxis, ...]),
        ),
        (weights @ values + biases).argmax(),
    )
