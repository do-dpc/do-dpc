"""
Rocket Environment Configuration Facade

This module provides a structured access to the `EnvConfig` parameters without requiring instantiation.
It includes methods for retrieving limits on state values, actuator values, and normalized actuator values
for the rocket environment.

The facade allows for easy configuration retrieval while keeping the main `EnvConfig` class clean.
"""

import math
import numpy as np

from do_dpc.environments.legacy_coco_rocket.env_cfg import EnvConfig


class RocketEnvCfgFacade:
    """
    Facade providing structured access to `EnvConfig` attributes without instantiation.

    This class exposes various environment parameters related to state limits, actuator limits,
    and transformations to normalized or virtual actuators.

    Note:
        The state variables are ordered as follows:
        - X position (m)
        - Y position (m)
        - X velocity (m/s)
        - Y velocity (m/s)
        - Theta (radians)
        - Theta velocity (rad/s)
    """

    @classmethod
    def max_state_values(cls) -> np.ndarray:
        """
        Returns the maximum permissible values for the state variables.

        Returns:
            np.ndarray: Array containing the maximum values for each state variable.

        Note:
            - Theta is represented in radians.
            - Some values are set to `np.inf` as they are unbounded.
        """
        return np.array([EnvConfig.width, EnvConfig.height, np.inf, np.inf, EnvConfig.theta_limit, np.inf])

    @classmethod
    def min_state_values(cls) -> np.ndarray:
        """
        Returns the minimum permissible values for the state variables.

        Returns:
            np.ndarray: Array containing the minimum values for each state variable.

        Note:
            - Theta is represented in radians.
            - Some values are set to `-np.inf` as they are unbounded.
        """
        return np.array([0, 0, -np.inf, -np.inf, -EnvConfig.theta_limit, -np.inf])

    @classmethod
    def max_actuator_values(cls) -> np.ndarray:
        """
        Returns the maximum possible actuator values.

        The actuators correspond to:
        - Main engine thrust (N)
        - Side engine thrust (N)
        - Nozzle angle (radians)

        Returns:
            np.ndarray: Array containing the maximum values for each actuator.
        """
        return np.array([EnvConfig.main_engine_thrust, EnvConfig.side_engine_thrust, EnvConfig.max_nozzle_angle])

    @classmethod
    def min_actuator_values(cls) -> np.ndarray:
        """
        Returns the minimum possible actuator values.

        Returns:
            np.ndarray: Array containing the minimum values for each actuator.
        """
        return np.array([0, -EnvConfig.side_engine_thrust, -EnvConfig.max_nozzle_angle])

    @classmethod
    def max_actuator_values_normalized(cls) -> np.ndarray:
        """
        Returns the maximum normalized actuator values.

        The normalized actuator values scale between -1 and 1, where:
        - 1 represents maximum thrust or angle.
        - 0 represents zero thrust or neutral angle.

        Returns:
            np.ndarray: Array containing normalized maximum actuator values.
        """
        return np.array([1, 1, 1])

    @classmethod
    def min_actuator_values_normalized(cls) -> np.ndarray:
        """
        Returns the minimum normalized actuator values.

        The normalized actuator values scale between -1 and 1, where:
        - 1 represents maximum thrust or angle.
        - 0 represents zero thrust or neutral angle.

        Returns:
            np.ndarray: Array containing normalized minimum actuator values.
        """
        return np.array([0, -1, -1])

    @classmethod
    def max_virtual_actuator_values(cls) -> np.ndarray:
        """
        Returns the maximum values for virtual actuators.

        Virtual actuators are an alternative representation of the real actuators.

        Returns:
            np.ndarray: Array containing the maximum values for virtual actuators.
        """
        return np.array([1, math.sin(cls.max_nozzle_angle()), 1])

    @classmethod
    def min_virtual_actuator_values(cls) -> np.ndarray:
        """
        Returns the minimum values for virtual actuators.

        Returns:
            np.ndarray: Array containing the minimum values for virtual actuators.
        """
        return np.array([0, -math.sin(cls.max_nozzle_angle()), -1])

    @classmethod
    def max_nozzle_angle(cls) -> float:
        """
        Returns the maximum allowable nozzle angle.

        Returns:
            float: Maximum nozzle angle in radians.
        """
        return EnvConfig.max_nozzle_angle

    @classmethod
    def normalized_thrust_to_hover(cls) -> float:
        """
        Returns the normalized thrust required to counteract gravity and maintain hover.

        Note:
            Assumes the rocket is upright with a neutral nozzle angle.

        Returns:
            float: Normalized thrust value needed for hovering.
        """
        return 0.322
