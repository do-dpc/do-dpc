# mypy: ignore-errors
# pylint: skip-file
"""
System Model of the RocketLander environment problem
Created by dyvogel on 2023-03-29

"""

from typing import Tuple

import numpy as np
import scipy

from do_dpc.environments.legacy_coco_rocket import RocketLander


class SystemModel:
    """Class which wraps a RocketLander environment and calculates the system model"""

    def __init__(self, env: RocketLander):
        """Constructor method

        Args:
            env (RocketLander): RocketLander environment instance
        """

        self.env = env

        mass, inertia = env.get_mass_properties()
        l1, l2 = env.get_dimensional_properties()
        gravity = -env.world.gravity[1]

        self.mass = mass
        self.inertia = inertia
        self.l1 = l1
        self.l2 = l2
        self.gravity = gravity

        self.A = None
        self.B = None
        self.Ad = None
        self.Bd = None

        self.state_shape = 6
        self.action_shape = 3

    def get_discrete_linear_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the discrete-time linearized system matrices

        Raises:
            AttributeError: if discretize_system_matrices has not been called first

        Returns:
            Tuple[np.ndarray, np.ndarray]: A, B
        """

        if self.Ad is None or self.Bd is None:
            raise AttributeError(
                "Ad and Bd matrices are not initialized. Please call `discretize_system_matrices` first"
            )
        return self.Ad, self.Bd

    def get_continuous_linear_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the continuous-time linearized system matrices

        Raises:
            AttributeError: if calculate_linear_system_matrices has not been called first

        Returns:
            Tuple[np.ndarray, np.ndarray]: A, B
        """

        if self.A is None or self.B is None:
            raise AttributeError(
                "A and B matrices are not initialized. Please call `calculate_linear_system_matrices` first"
            )
        return self.A, self.B

    def calculate_linear_system_matrices(self, x_eq: np.ndarray = None, u_eq: np.ndarray = None) -> None:
        """Calculate the linearized system matrices

        If x_eq and u_eq are not provided, we default to:
            - x_eq = [0, 0, 0, 0, 0, 0] (x = [x, y, x', y', theta, theta'])
            - u_eq = [mass*gravity, 0, 0] (u = [Fe, Fs, phi])

        Args:
            x_eq (np.ndarray, optional): state equilibrium point. Defaults to None.
            u_eq (np.ndarray, optional): action equilibrium point. Defaults to None.
        """

        if x_eq is None:
            x_eq = np.zeros(self.state_shape)
        if u_eq is None:
            u_eq = np.array([self.mass * self.gravity, 0, 0])

        # linearized state dynamics
        a24 = (-u_eq[1] * np.sin(x_eq[4]) - u_eq[0] * np.cos(x_eq[4] + u_eq[2])) / self.mass
        a34 = (+u_eq[1] * np.cos(x_eq[4]) - u_eq[0] * np.sin(x_eq[4] + u_eq[2])) / self.mass

        # linearized input dynamics
        b20 = -np.sin(x_eq[4] + u_eq[2]) / self.mass
        b21 = np.cos(x_eq[4]) / self.mass
        b22 = -u_eq[0] * np.cos(x_eq[4] + u_eq[2]) / self.mass

        b30 = np.cos(x_eq[4] + u_eq[2]) / self.mass
        b31 = np.sin(x_eq[4]) / self.mass
        b32 = -u_eq[0] * np.sin(x_eq[4] + u_eq[2]) / self.mass

        b50 = -self.l1 * np.sin(u_eq[2]) / self.inertia
        b51 = -self.l2 / self.inertia
        b52 = -self.l1 * u_eq[0] * np.cos(u_eq[2]) / self.inertia

        # A matrix
        self.A = np.zeros((self.state_shape, self.state_shape))
        self.A[0, 2] = 1
        self.A[1, 3] = 1
        self.A[2, 4] = a24
        self.A[3, 4] = a34
        self.A[4, 5] = 1

        # B matrix
        self.B = np.zeros((self.state_shape, self.action_shape))
        self.B[2, 0] = b20
        self.B[2, 1] = b21
        self.B[2, 2] = b22

        self.B[3, 0] = b30
        self.B[3, 1] = b31
        self.B[3, 2] = b32

        self.B[5, 0] = b50
        self.B[5, 1] = b51
        self.B[5, 2] = b52

        # we normalize the applied actions within the allowable input range
        normalization_u = np.diag(
            [self.env.cfg.main_engine_thrust, self.env.cfg.side_engine_thrust, self.env.cfg.max_nozzle_angle]
        )

        self.B = self.B @ normalization_u

    def discretize_system_matrices(self, sample_time: float) -> None:
        """Exact discretization of the linearized system matrices using the matrix exponential

        Args:
            sample_time (float): discrete sampling time in seconds

        Raises:
            AttributeError: if calculate_linear_system_matrices has not been called first
        """

        if self.A is None or self.B is None:
            print("Note: A and B matrices have not been initialized. Using the (default) upright equilibrium")
            self.calculate_linear_system_matrices()

        # exact discretization using matrix exponential
        self.Ad = scipy.linalg.expm(self.A * sample_time)

        # integrate matrix exponential, multiply with B
        Ad_int, _ = scipy.integrate.quad_vec(lambda tau: scipy.linalg.expm(self.A * tau), 0, sample_time)
        self.Bd = Ad_int @ self.B
