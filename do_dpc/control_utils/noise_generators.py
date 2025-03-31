"""
Generates white noise for control applications.

This module provides a `WhiteNoiseGenerator` class for creating Gaussian
white noise with configurable mean and standard deviation.
"""

from typing import Optional

import numpy as np

from do_dpc.utils.logging_config import get_logger

logger = get_logger(__name__)


class WhiteNoiseGenerator:  # pylint: disable=too-few-public-methods
    """
    This class generates Gaussian white noise with a specified mean and standard deviation.
    The standard deviation can vary for each state dimension, providing flexibility in noise generation.

    Attributes:
        mean (float): The mean of the Gaussian noise.
        std (np.ndarray): Standard deviations for each state dimension.
        rng (np.random.Generator): Random number generator used for generating noise.

    Args:
        mean (Optional[np.ndarray]): Mean of the Gaussian noise.
        std (Optional[np.ndarray], optional): Standard deviation for each state dimension.
            If not provided, defaults to an array of 1.0.
        seed (Optional[int], optional): Random seed for reproducibility.

    Raises:
        ValueError: If the standard deviations are not provided in an appropriate format (must be a 1D array or scalar).
    """

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, seed: Optional[int] = None):
        """
        Initializes the white noise generator.

        Args:
            mean (Optional[np.ndarray]): Mean of the Gaussian noise.
            std (Optional[np.ndarray]): Standard deviation for each state dimension.
            seed (Optional[int]): Random seed for reproducibility.

        Raises:
            ValueError: If `mean` and `std` have different shapes after ensuring they are 1D arrays.
        """

        if mean is not None:
            self.mean = np.atleast_1d(mean)
        elif std is not None:
            self.mean = np.zeros_like(std)
        else:
            self.mean = np.zeros(1)

        if std is not None:
            self.std = np.atleast_1d(std)
        else:
            self.std = np.ones_like(self.mean)

        if self.mean.shape != self.std.shape:
            raise ValueError(f"Mean and std must have the same shape, got {self.mean.shape} and {self.std.shape}.")

        self.rng = np.random.default_rng(seed)
        logger.info("WhiteNoiseGenerator initialized with mean=%s, std=%s, seed=%s", self.mean, self.std, seed)

    def generate(self) -> np.ndarray:
        """
        Generates white noise of the given size.

        Returns:
            np.ndarray: Generated white noise samples.
        """
        noise = self.rng.normal(self.mean, self.std)
        logger.debug("Generated white noise: %s", noise)
        return noise
