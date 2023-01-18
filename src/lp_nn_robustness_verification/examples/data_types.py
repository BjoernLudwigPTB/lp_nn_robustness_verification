"""This module contains named tubles for more convenient coding around the examples"""

__all__ = [
    "IndexAndSeed",
    "Instances",
    "ScalerAndLayers",
]

from typing import NamedTuple, TypeAlias


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


Instances: TypeAlias = dict[ScalerAndLayers, IndexAndSeed]
"""A tuple of an input sizer, sample index, number of layers and random seed"""
