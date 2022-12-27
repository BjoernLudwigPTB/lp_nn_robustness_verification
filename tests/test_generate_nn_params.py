from inspect import signature
from typing import cast

import numpy as np
from hypothesis import given, settings, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_allclose, assert_equal

from lp_nn_robustness_verification.data_acquisition import generate_nn_params
from lp_nn_robustness_verification.data_acquisition.generate_nn_params import (
    construct_partition,
    generate_bias_vector,
    generate_weight_matrix,
    generate_weights_and_biases,
)
from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import NNParams, UncertainArray


@composite
def uncertain_inputs(
    draw: DrawFn, dimen: int | None = None
) -> SearchStrategy[UncertainInputs]:
    dimen = draw(hst.integers(min_value=1, max_value=100)) if dimen is None else dimen
    values = draw(
        hnp.arrays(
            np.float64, dimen, elements=hst.floats(min_value=1e250, max_value=1e250)
        )
    )
    values_absolute_minimum = np.absolute(values).min()
    uncertainties = draw(
        hnp.arrays(
            np.float64,
            dimen,
            elements=hst.floats(
                min_value=values_absolute_minimum * 1e-3,
                max_value=values_absolute_minimum * 1e2,
            ),
        )
    )
    return cast(
        SearchStrategy[UncertainInputs],
        UncertainInputs(UncertainArray(values, uncertainties)),
    )


@composite
def in_out_features_and_uncertain_values(
    draw: DrawFn,
) -> SearchStrategy[tuple[int, list[int], UncertainInputs]]:
    in_features = draw(hst.integers(10, 100))
    out_features = construct_partition(in_features)
    uncertain_values = draw(uncertain_inputs(dimen=in_features))
    return cast(
        SearchStrategy[tuple[int, list[int], UncertainInputs]],
        (in_features, out_features, uncertain_values),
    )


def test_construct_partition_exists() -> None:
    assert hasattr(generate_nn_params, "construct_partition")


def test_construct_partition_has_docstring() -> None:
    assert construct_partition.__doc__ is not None


def test_construct_partition_has_parameter_in_features() -> None:
    assert "in_features" in signature(construct_partition).parameters


def test_construct_partition_parameter_in_features_is_of_type_int() -> None:
    assert signature(construct_partition).parameters["in_features"].annotation is int


def test_construct_partition_parameter_states_to_return_int_list() -> None:
    assert signature(construct_partition).return_annotation == list[int]


@given(hst.integers(min_value=1, max_value=10))
def test_construct_partition_actually_returns_int_list(in_features: int) -> None:
    assert isinstance(construct_partition(in_features), list)


@given(hst.integers(min_value=1, max_value=100))
def test_construct_partition_returns_non_empty_list(in_features: int) -> None:
    assert len(construct_partition(in_features))


def test_construct_partition_returns_correct_small_example() -> None:
    assert construct_partition(10) == [10, 7, 5, 3, 2]


@given(hst.integers(min_value=1, max_value=100))
def test_construct_partition_is_descending(in_features: int) -> None:
    partition = construct_partition(in_features)
    assert partition[:] > partition[1:]


def test_construct_partition_returns_correct_large_example() -> None:
    assert construct_partition(100) == [100, 75, 56, 42, 31]


def test_initialize_weight_matrix_exists() -> None:
    assert hasattr(generate_nn_params, "generate_weight_matrix")


@given(
    hst.integers(min_value=1, max_value=100), hst.integers(min_value=1, max_value=100)
)
def test_initialize_weight_matrix_provides_ndarray(
    in_features: int, out_features: int
) -> None:
    assert isinstance(generate_weight_matrix(in_features, out_features), np.ndarray)


@given(uncertain_inputs(), hst.integers(min_value=1, max_value=100))
def test_initialize_weight_matrix_allows_matmul_with_inputs(
    uncertain_input: UncertainInputs, out_features: int
) -> None:
    assert isinstance(
        generate_weight_matrix(
            len(uncertain_input.uncertain_values.values), out_features
        )
        @ uncertain_input.uncertain_values.values,
        np.ndarray,
    )


@given(
    hst.integers(min_value=1000, max_value=2000),
    hst.integers(min_value=1000, max_value=2000),
)
@settings(deadline=None)
def test_weight_matrix_mean_is_around_zero(in_features: int, out_features: int) -> None:
    assert_allclose(
        np.mean(generate_weight_matrix(in_features, out_features)),
        0,
        rtol=1e-4,
        atol=1e-4,
    )


def test_initialize_bias_vector_exists() -> None:
    assert hasattr(generate_nn_params, "generate_bias_vector")


@given(
    hst.integers(min_value=1, max_value=100), hst.integers(min_value=1, max_value=100)
)
def test_initialize_bias_vector_provides_ndarray(
    in_features: int, out_features: int
) -> None:
    assert isinstance(generate_bias_vector(in_features, out_features), np.ndarray)


@given(uncertain_inputs(), hst.integers(min_value=1, max_value=100))
def test_initialize_bias_vector_has_desied_shape(
    uncertain_input: UncertainInputs, out_features: int
) -> None:
    assert_equal(
        generate_bias_vector(
            len(uncertain_input.uncertain_values.values), out_features
        ).shape,
        (out_features,),
    )


@given(
    hst.integers(min_value=1, max_value=100),
    hst.integers(min_value=800000, max_value=1200000),
)
@settings(deadline=None)
def test_initialize_bias_vector_mean_is_around_zero(
    in_features: int, out_features: int
) -> None:
    assert_allclose(
        np.mean(generate_bias_vector(in_features, out_features)),
        0,
        rtol=2e-3,
        atol=2e-3,
    )


def test_generate_weights_and_biases_exists() -> None:
    assert hasattr(generate_nn_params, "generate_weights_and_biases")


@given(in_out_features_and_uncertain_values())
def test_generate_weights_and_biases_provides_nn_params(
    in_and_out_feature: tuple[int, list[int], UncertainInputs]
) -> None:
    assert isinstance(generate_weights_and_biases(*in_and_out_feature[:2]), NNParams)


@given(in_out_features_and_uncertain_values())
def test_generate_weights_and_biases_provides_compatible_nn_params(
    in_and_out_feature: tuple[int, list[int], UncertainInputs]
) -> None:
    nn_params: NNParams = generate_weights_and_biases(*in_and_out_feature[:2])
    forward_pass = in_and_out_feature[2].uncertain_values.values
    for biases, weights in nn_params:
        forward_pass = weights @ forward_pass + biases
    assert_equal(len(forward_pass), in_and_out_feature[1][-1])
