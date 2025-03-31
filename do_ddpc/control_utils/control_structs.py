"""
Data structures for control theory.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HankelMatrices:
    """
    Stores Hankel matrices used in Data-Driven Predictive Control (DPC).

    Attributes:
        Z (np.ndarray): Full Hankel matrix containing past and future input-output data.
        Z_p (np.ndarray): Hankel matrix containing past input-output data.
        U_f (np.ndarray): Hankel matrix containing future control input data.
        Y_f (np.ndarray): Hankel matrix containing future system output data.
        n_col (int): Number of columns in the Hankel matrices.
        n_samples (int): Total number of collected samples.
    """

    Z: np.ndarray
    Z_p: np.ndarray
    U_f: np.ndarray
    Y_f: np.ndarray
    n_col: int
    n_samples: int

    def __post_init__(self):
        """
        Ensures that `n_col` and `n_samples` are positive integers.

        Raises:
            ValueError: If `n_col` or `n_samples` are not positive integers.
        """
        if not isinstance(self.n_col, int) or self.n_col <= 0:
            raise ValueError(f"n_col must be a positive integer, but got {self.n_col}")

        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, but got {self.n_samples}")


@dataclass
class InputOutputTrajectory:
    """
    Stores system input/output trajectory.

    Attributes:
        y (np.ndarray): System outputs of shape `(p, N)`.
        u (np.ndarray): Control inputs of shape `(m, N)`.
    """

    y: np.ndarray
    u: np.ndarray


@dataclass
class Bounds:
    """
    Represents upper and lower bounds for optimization variables.

    Attributes:
        max_values (np.ndarray): The upper bound values.
        min_values (np.ndarray): The lower bound values.

    Raises:
        ValueError: If `max_values` and `min_values` have different shapes.
        ValueError: If any `min_values[i] > max_values[i]`, ensuring valid bounds.
    """

    max_values: np.ndarray
    min_values: np.ndarray

    def __post_init__(self):
        """
        Validates that `max_values` and `min_values`:
        - Have the same shape.
        - Satisfy `min_values <= max_values` for all elements.

        Raises:
            ValueError: If shapes mismatch or if min_values exceed max_values.
        """
        if self.max_values.shape != self.min_values.shape:
            raise ValueError(
                f"Shape mismatch: max_values has shape {self.max_values.shape}, "
                f"but min_values has shape {self.min_values.shape}."
            )

        if np.any(self.min_values > self.max_values):
            raise ValueError(
                "Invalid bounds: All elements in min_values must be ≤ corresponding elements in max_values."
            )
