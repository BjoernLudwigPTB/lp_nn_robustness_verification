"""Generate weights and biases"""
import math

import numpy as np
from numpy._typing import NDArray

from lp_nn_robustness_verification.data_types import NNParams


def construct_out_features_counts(
    in_features: int, out_features: int = 2, depth: int = 1
) -> list[int]:
    """Construct network architecture with desired depth for parameter generation"""
    if depth == 1:
        return [out_features]
    assert in_features > out_features
    assert (in_features - out_features) / depth >= 1.0
    partition = {out_features}
    while len(partition) < depth:
        step = (in_features - out_features) / (depth - len(partition) + 1)
        partition.add(in_features := math.ceil(in_features - step))
    assert len(partition) == depth
    assert min(partition) == out_features
    return list(sorted(partition, reverse=True))


def generate_weight_matrix(
    in_features: int, out_features: int, seed: int | None = None
) -> NDArray[np.float64]:
    """Initialize a weight matrix in accordance with Kaiming initialization scheme"""
    bound = 1 / math.sqrt(in_features)
    return np.random.default_rng(seed).uniform(
        -bound, bound, (out_features, in_features)
    )


def generate_bias_vector(
    in_features: int, out_features: int, seed: int | None = None
) -> NDArray[np.float64]:
    """Initialize a bias vector in accordance with pytorch initialization scheme"""
    bound = 1 / math.sqrt(in_features)
    return np.random.default_rng(seed).uniform(-bound, bound, out_features)


def generate_weights_and_biases(
    in_features: int, out_features: list[int], seed: int | None = None
) -> NNParams:
    """Initialize weight matrices and bias vectors according to Kaiming scheme"""
    weight_matrices = [generate_weight_matrix(in_features, out_features[0], seed)]
    bias_vectors = [generate_bias_vector(in_features, out_features[0], seed)]
    if len(out_features) >= 2:
        for n_in_features, n_out_features in zip(out_features, out_features[1:]):
            weight_matrices.append(
                generate_weight_matrix(n_in_features, n_out_features, seed)
            )
            bias_vectors.append(
                generate_bias_vector(n_in_features, n_out_features, seed)
            )
    return NNParams(tuple(bias_vectors), tuple(weight_matrices))
