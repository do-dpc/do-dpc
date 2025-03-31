"""
Helper functions for converting double integrator system outputs to error and derivative of error.
"""

from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import numpy as np

from do_dpc.control_utils.pid import MIMOPIDController


@dataclass
class PIDCombo:
    """
    A dataclass to bundle a MIMOPIDController and a converter function.

    Attributes:
        MIMO_PID (MIMOPIDController): An instance of MIMO_PIDController.
        converter_function (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]):
        A function to convert outputs to error and derivative of error.
    """

    MIMO_PID: MIMOPIDController
    converter_function: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]]


def one_D_double_integrator_output_to_err_der_err(
    output: np.ndarray, output_target: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 1D double integrator output to error and derivative of error.

    Args:
        output (np.ndarray): The output array containing the state and its derivative.
        output_target (np.ndarray, optional): The target output array. Defaults to an array of zeros.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the error and the derivative of the error.
    """
    if output.shape[0] != 2:
        raise ValueError("Output must be a 1D array with exactly 2 elements.")

    if output_target is None:
        output_target = np.zeros_like(output)

    error = np.array([output[0] - output_target[0]])
    der_error = np.array([output[1] - output_target[1]])
    return error, der_error


def three_D_double_integrator_output_to_err_der_err(
    output: np.ndarray, output_target: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 3D double integrator output to error and derivative of error.

    Args:
        output (np.ndarray): The output array containing the states and their derivatives.
        output_target (np.ndarray, optional): The target output array. Defaults to an array of zeros.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the errors and the derivatives of the errors.
    """
    if output.shape[0] != 6:
        raise ValueError("Output must be a 1D array with exactly 6 elements.")

    if output_target is None:
        output_target = np.zeros_like(output)

    error = np.array([output[0] - output_target[0], output[2] - output_target[2], output[4] - output_target[4]])
    der_error = np.array([output[1] - output_target[1], output[3] - output_target[3], output[5] - output_target[5]])
    return error, der_error


# pylint: disable=too-many-locals
def rocket_output_to_err_der_err(
    output: np.ndarray, output_target: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a Rocket output to error and derivative of error.

    Args:
        output (np.ndarray): The output array containing the states and their derivatives.
        output_target (np.ndarray, optional): The target output array. Defaults to an array of zeros.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the errors and the derivatives of the errors.
    """
    if output.shape[0] != 6 and output.shape[0] != 8:
        raise ValueError("Output must be a 1D array with exactly 6 or 8 elements.")

    if output_target is None:
        output_target = np.zeros_like(output)
    if output.shape[0] == 6:
        x, y, vel_x, vel_y, theta, omega = output
    else:
        x, y, vel_x, vel_y, theta, omega, _, _ = output

    dx = x - output_target[0]
    dy = y - output_target[1]

    # Fe
    y_adj = -0.1  # Adjust speed
    err_1 = y_adj - dy + 0.1 * dx
    derr_1 = -vel_y + 0.1 * vel_x

    # Fs
    err_2 = -theta + 0.2 * dx
    derr_2 = -omega + 0.2 * vel_x

    # Psi
    err_3 = theta
    derr_3 = omega
    if abs(dx) > 0.01 and dy < 0.5:
        err_3 = err_3 - 0.06 * dx  # theta is negative when slanted to the right
        derr_3 = derr_3 - 0.06 * vel_x

    err = np.array([-err_1, err_2, -err_3])
    derr = np.array([-derr_1, derr_2, -derr_3])

    return err, derr
