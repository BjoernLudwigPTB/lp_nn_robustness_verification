"""Generate weights and biases"""
import math

import numpy as np
from numpy._typing import NDArray

from lp_nn_robustness_verification.data_types import NNParams


def construct_partition(in_features: int) -> list[int]:
    """Construct partition of each 0.75 times smaller sections"""
    # TODO replace by pytorch_gum_uncertainty_propagation.examples.propagate's function
    partition = {in_features}
    while in_features > 2 and len(partition) < 5:
        partition.add(in_features := 3 * in_features // 4)
    return list(sorted(partition, reverse=True))


def generate_weight_matrix(in_features: int, out_features: int) -> NDArray[np.float64]:
    """Initialize a weight matrix in accordance with Kaiming initialization scheme"""
    bound = 1 / math.sqrt(in_features)
    return np.random.default_rng().uniform(-bound, bound, (out_features, in_features))


def generate_bias_vector(in_features: int, out_features: int) -> NDArray[np.float64]:
    """Initialize a bias vector in accordance with pytorch initialization scheme"""
    bound = 1 / math.sqrt(in_features)
    return np.random.default_rng().uniform(-bound, bound, out_features)


def generate_weights_and_biases(in_features: int, out_features: list[int]) -> NNParams:
    """Initialize weight matrices and bias vectors according to Kaiming scheme"""
    weight_matrices = [generate_weight_matrix(in_features, out_features[0])]
    bias_vectors = [generate_bias_vector(in_features, out_features[0])]
    if len(out_features) >= 2:
        for n_in_features, n_out_features in zip(out_features, out_features[1:]):
            weight_matrices.append(
                generate_weight_matrix(n_in_features, n_out_features)
            )
            bias_vectors.append(generate_bias_vector(n_in_features, n_out_features))
    return NNParams(tuple(bias_vectors), tuple(weight_matrices))
