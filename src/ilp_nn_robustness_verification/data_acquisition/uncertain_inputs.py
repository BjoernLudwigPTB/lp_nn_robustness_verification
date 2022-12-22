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
        self.intervals = self.build_intervals(self.uncertain_values)

    @staticmethod
    def build_intervals(uncertain_values: UncertainArray) -> Intervals:
        """Construct the interval arithmetically enabled datastructure"""
        return Intervals(
            interval[value - uncertainty, value + uncertainty]
            for value, uncertainty in zip(*uncertain_values)
        )
