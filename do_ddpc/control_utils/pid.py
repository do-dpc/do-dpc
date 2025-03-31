"""
Implements a discrete-time PID controller.

This module provides a `PIDController` class for computing PID control actions
and a `PIDGains` dataclass for storing PID gain parameters.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PIDGains:
    """
    A dataclass to store PID gains.
    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
    """

    Kp: float
    Ki: float
    Kd: float


class PIDController:
    r"""
    A discrete-time PID controller.
    This class implements a discrete-time PID (Proportional-Integral-Derivative) controller.
    It computes the control action based on the error between the desired setpoint and the current process variable,
    using the PID control law:

    .. math::
        u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{d}{dt} e(t)

    where:
        - :math:`u(t)` is the control output
        - :math:`e(t)` is the error between the setpoint and the process variable
        - :math:`K_p`, :math:`K_i`, and :math:`K_d` are the proportional, integral, and derivative gains, respectively.

    Attributes:
        gains (PIDGains): An instance of PIDGains containing the PID constants: `Kp`, `Ki`, and `Kd`.
        dt (float): Sampling time (time step between updates).
        alpha (float): Smoothing factor for the derivative term (low-pass filter for noise reduction).
        integral (float): The accumulated integral term.
        prev_error (float): The previous error used for the derivative calculation.
        filtered_derivative (float): The filtered derivative term.

    Args:
        gains (PIDGains): An instance of PIDGains containing `Kp`, `Ki`, and `Kd`.
        dt (float): Sampling time.
        alpha (float, optional): Smoothing factor for the derivative term. Defaults to 0.1.

    Raises:
        ValueError: If `dt` is not positive.
    """

    def __init__(self, gains: PIDGains, dt: float, alpha: float = 0.1):
        """
        Initializes the discrete-time PID controller.

        Args:
            gains (PIDGains): An instance of PIDGains containing Kp, Ki, and Kd.
            dt (float): Sampling time.
            alpha (float, optional): Smoothing factor for the derivative term. Defaults to 0.1.

        Raises:
            ValueError: If dt is not positive.
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        self.gains = gains
        self.dt = dt
        self.alpha = alpha
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def compute(self, error: float) -> float:
        """
        Computes the discrete PID control output.

        Args:
            error (float): The error signal (difference between setpoint and measurement).

        Returns:
            float: The computed control signal.
        """
        P = self.gains.Kp * error
        self.integral += error * self.dt
        I = self.gains.Ki * self.integral
        raw_derivative = (error - self.prev_error) / self.dt
        self.filtered_derivative = self.alpha * raw_derivative + (1 - self.alpha) * self.filtered_derivative
        D = self.gains.Kd * self.filtered_derivative
        self.prev_error = error
        return P + I + D

    def compute_with_derivative(self, error: float, error_derivative: float) -> float:
        """
        Computes the discrete PID control output with an externally provided error derivative.

        Args:
            error (float): The error signal (difference between setpoint and measurement).
            error_derivative (float): The derivative of the error signal.

        Returns:
            float: The computed control signal.
        """
        P = self.gains.Kp * error
        self.integral += error * self.dt
        I = self.gains.Ki * self.integral
        D = self.gains.Kd * error_derivative
        self.prev_error = error
        return P + I + D


class MIMOPIDController:
    """
    A class to manage multiple PID controllers for a decoupled MIMO system.

    This class holds multiple PIDController instances and computes the control actions for each one.

    Attributes:
        pid_controllers (List[PIDController]): A list of PIDController instances.
    """

    def __init__(self, pid_controllers: List[PIDController]):
        """
        Initializes the MIMO_PIDController with a list of PIDController instances.

        Args:
            pid_controllers (List[PIDController]): A list of PIDController instances.
        """
        self.pid_controllers = pid_controllers

    def compute(self, errors: np.ndarray) -> np.ndarray:
        """
        Computes the control actions for all PID controllers.

        Args:
            errors (np.ndarray): An array of error signals (differences between setpoints and measurements).

        Returns:
            np.ndarray: An array of computed control signals.
        """
        return np.array([pid.compute(error) for pid, error in zip(self.pid_controllers, errors)])

    def compute_with_derivative(self, errors: np.ndarray, error_derivatives: np.ndarray) -> np.ndarray:
        """
        Computes the control actions for all PID controllers using provided error derivatives.

        Args:
            errors (np.ndarray): An array of error signals (differences between setpoints and measurements).
            error_derivatives (np.ndarray): An array of error derivatives.

        Returns:
            np.ndarray: An array of computed control signals.
        """
        return np.array(
            [
                pid.compute_with_derivative(error, error_derivative)
                for pid, error, error_derivative in zip(self.pid_controllers, errors, error_derivatives)
            ]
        )
