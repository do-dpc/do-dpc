"""
This module contains utility functions used for transforming rocket control inputs between
virtual space (used by the control algorithms) and rocket-specific space (used by the environment).
"""

import math

import numpy as np

from do_dpc.environments.rocket_env.rocket_env_cfg_facade import RocketEnvCfgFacade


def convert_virtual_u_to_rocket_u(u_virtual: np.ndarray) -> np.ndarray:
    """
    Converts virtual control inputs into rocket control inputs.

    The transformation follows:
        Fe = sqrt(Fx^2 + Fy^2)
        phi = arctan(Fy/Fx)

    Args:
        u_virtual (np.ndarray): Virtual actuator inputs, a 3-element array [Fx, Fy, Fs].

    Returns:
        np.ndarray: Rocket control inputs, a 3-element array [Fe, Fs, phi_normalized], where:
            - Fe is the magnitude of the thrust vector.
            - Fs is the side thrust value.
            - phi_normalized is the normalized nozzle angle.

    Note:
        - If Fx is too small (near zero), `phi` is safely calculated using arctan2.
        - phi is normalized using the maximum nozzle angle from `RocketEnvCfgFacade`.
    """
    Fx, Fy, Fs = u_virtual

    Fe = math.sqrt(Fx**2 + Fy**2)
    phi = math.atan2(Fy, Fx)

    return np.array([Fe, Fs, phi / RocketEnvCfgFacade.max_nozzle_angle()])


def convert_rocket_u_to_virtual_u(u_rocket: np.ndarray) -> np.ndarray:
    """
    Converts rocket control inputs into virtual control inputs.

    The transformation follows:
        u1 = Fe * cos(phi)
        u2 = Fe * sin(phi)

    Args:
        u_rocket (np.ndarray): Rocket control inputs, a 3-element array [Fe, Fs, phi_normalized].

    Returns:
        np.ndarray: Virtual actuator inputs, a 3-element array [Fx, Fy, Fs], where:
            - Fx and Fy are the thrust components in the x and y directions.
            - Fs is the side thrust value.
    """
    Fe, Fs, phi_norm = u_rocket

    phi = phi_norm * RocketEnvCfgFacade.max_nozzle_angle()

    Fx = Fe * math.cos(phi)
    Fy = Fe * math.sin(phi)

    return np.array([Fx, Fy, Fs])


def calculate_normalized_thrust_to_hover() -> float:
    """
    Calculates the normalized thrust value required to counteract gravity and keep the rocket hovering.

    This is based on the specific conditions under which the rocket is assumed to be straight positioned,
    with the nozzle pointed directly downwards.

    Note:
       This is a constant value in the current model; it could depend on other factors in a more complex system

    Returns:
        float: Normalized thrust required for hover.
    """
    return 0.32


def is_rocket_outside_reasonable_bounds(output: np.ndarray) -> bool:
    """
    Checks if the rocket is outside reasonable safety bounds.

    Args:
        output (np.ndarray): A state vector containing position, angle, and velocity components.

    Returns:
        bool: True if the rocket is outside reasonable bounds, False otherwise.
    """
    _, y, _, _, angle, angle_vel = output

    return y > 5 or abs(angle) > 0.1 or abs(angle_vel) > 0.4
