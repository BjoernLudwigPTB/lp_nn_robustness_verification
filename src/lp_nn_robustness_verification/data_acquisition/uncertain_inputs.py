"""This module contains the class providing a unified interface for character sets"""

__all__ = ["UncertainInputs"]

from dataclasses import dataclass

import numpy as np
from interval import interval

from ..data_types import Intervals, UncertainArray


@dataclass
class UncertainInputs:
    r"""A unified interface to a collection of uncertain inputs

    Parameters
    ----------
    uncertain_values : UncertainArray, optional
        Values with associated uncertainties, defaults to the 2-d point :math:`x = (
        \frac{1}{2}, \frac{1}{2})` with uncertainties :math:`u(x) = (\frac{1}{2},
        \frac{1}{2})`
    """

    uncertain_values: UncertainArray
    intervals: Intervals

    def __init__(self, uncertain_values: UncertainArray | None = None) -> None:
        """Uncertain inputs i.e. an array of values and an array of uncertainties"""
        if uncertain_values is None:
            self.uncertain_values = UncertainArray(
                np.array([0.5, 0.5]), np.array([0.5, 0.5])
            )
        else:
            self.uncertain_values = uncertain_values
        assert (
            len(self.uncertain_values.values.shape)
            == len(self.uncertain_values.uncertainties.shape)
            == 1
        ), (
            f"Either one or both of the values and the uncertainties are not given "
            f"as a vector but the values are of shape "
            f"{self.uncertain_values.values.shape} and the uncertainties of "
            f"shape {self.uncertain_values.uncertainties.shape}"
        )
        assert len(self.uncertain_values.values) == len(
            self.uncertain_values.uncertainties
        ), (
            f"Somehow values and associated uncertainties are not of the same size "
            f"but the values are of length {len(self.uncertain_values.values)} and "
            f"the uncertainties of length {len(self.uncertain_values.uncertainties)}"
        )
        self.intervals = self.build_intervals(self.uncertain_values)
        assert len(self.intervals) == len(self.uncertain_values.values), (
            f"Somehow there were not as many intervals calculated as there are values, "
            f"but there are {len(self.intervals)} intervals and each "
            f"{len(self.uncertain_values.values)} values and uncertainties"
        )

    @staticmethod
    def build_intervals(uncertain_values: UncertainArray) -> Intervals:
        """Construct the interval arithmetically enabled datastructure"""
        return Intervals(
            interval[value - uncertainty, value + uncertainty]
            for value, uncertainty in zip(*uncertain_values)
        )
