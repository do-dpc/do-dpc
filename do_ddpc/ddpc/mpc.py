"""
Abstract base class for Linear Quadratic Gaussian (MPC) Control based on Data-Driven Predictive Control (DPC).

This module provides a structured interface for all MPC-based controllers.

Subclasses must implement:
- `calculate_system_data()`
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.linalg import pinv

from do_ddpc.control_utils.lti_systems import StateSpaceModel
from do_ddpc.ddpc.ddpc import DPC
from do_ddpc.ddpc.ddpc_structs import (
    DPCParameters,
    DPCClosedFormSolutionMatrices,
    DPCPredictorMatrices,
    DPCRegularizationMatrices,
)
from do_ddpc.control_utils.control_structs import InputOutputTrajectory

logger = logging.getLogger(__name__)


@dataclass
class MPCRegularizationMatrices(DPCRegularizationMatrices):
    """Currently there is no regularization cost function for the MPC algorithm."""


@dataclass
class MPCSystemMatrices:
    """
    Stores system matrices for the MPC controller.

    Attributes:
        K (np.ndarray): Infinite horizon Kalman gain.
        sys (StateSpaceModel): System data containing state-space matrices A, B, C, D.
    """

    K: np.ndarray
    sys: StateSpaceModel


@dataclass
class MPCPredictorMatrices(DPCPredictorMatrices):
    """
    Stores offline data for MPC  controller.

    The relationship is given by:
    y_f = Gamma x + H_u u_f

    Here, Gamma is the extended observability matrix, i.e., Gamma * x represents the unforced response.
    H_u maps the future inputs to the future outputs.

    Definitions:
    Gamma = [C; CA; CA^2; ... CA^(tau_f-1)]
    H_u = [D 0 ... 0; CB D ... 0; CAB CB D ... 0; ...; CA^(tau_f-2)B ... D]

    Attributes:
        Gamma (np.ndarray): Extended observability matrix.
        H_u (np.ndarray): Matrix mapping future inputs to future outputs.
    """

    Gamma: np.ndarray
    H_u: np.ndarray


@dataclass
class MPCClosedFormSolutionMatrices:
    """
    Stores closed-form gains for the MPC (Linear Quadratic Gaussian) controller.

    Attributes:
        K_x (np.ndarray): State feedback gain matrix.
        K_y_r (np.ndarray): Output reference gain matrix.
        K_u_r (np.ndarray): Control input reference gain matrix.
    """

    K_x: np.ndarray
    K_y_r: np.ndarray
    K_u_r: np.ndarray


# pylint: disable=too-many-instance-attributes
class MPC(DPC, ABC):
    """
    Abstract class for implementing MPC based on DPC.
    The matrices A, B, C, D and the Kalman filter gain K can be calculated differently for different methods.

    Args:
        n_state (int): Number of states in the system. Must be greater than or equal to 1.
        **Kwargs: Additional keyword arguments.

    Raises:
        ValueError: If `n_state` is less than 1.

    Attributes:
        n_state (int): Number of states in the system.
        x_cp (cp.Parameter): Control parameter representing the state vector.
        mpc_cf_gains (MPCClosedFormSolutionMatrices): Calculated closed-form gains for the MPC controller.
    """

    def __init__(self, ddpc_params: DPCParameters, training_data: InputOutputTrajectory, n_state: int, **Kwargs):
        """
        Initializes the MPC controller.

        Args:
            n_state (int): Number of states in the system. Must be greater than or equal to 1.
            **Kwargs: Additional keyword arguments.

        Raises:
            ValueError: If `n_state` is less than 1.

        Attributes:
            n_state (int): Number of states in the system.
            x_cp (cp.Parameter): Control parameter representing the state vector.
            mpc_cf_gains (MPCClosedFormSolutionMatrices): Calculated closed-form gains for the MPC controller.
        """
        if n_state < 1:
            raise ValueError("N_state can not be smaller than 1.")

        self.sys_matrices: Optional[MPCSystemMatrices] = None
        self.n_state = n_state
        self.x_cp = cp.Parameter(shape=self.n_state, value=np.zeros(self.n_state))
        super().__init__(ddpc_params, training_data)
        self.mpc_cf_gains = self._calculate_mpc_closed_form_solution_matrices()

    def calculate_predictor_matrices(self) -> MPCPredictorMatrices:
        """
        Calculates the offline data required for the MPC  controller.

        This method computes the extended observability matrix (Gamma) and the matrix mapping
        future inputs to future outputs (H_u) based on the system's state-space representation
        matrices (A, B, C, D) and control parameters (tau_f, p, m).

        Returns:
            MPCPredictorMatrices: An instance containing the computed Gamma, H_u, system data (sys),
                            and the infinite horizon Kalman gain (K).
        """

        self.sys_matrices = self.calculate_system_data()
        A = self.sys_matrices.sys.A
        C = self.sys_matrices.sys.C
        B = self.sys_matrices.sys.B
        D = self.sys_matrices.sys.D

        tau_f = self.ddpc_params.tau_f
        p = self.dims.p
        m = self.dims.m

        Gamma = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(tau_f)])

        H_u = np.zeros((self.dims.n_y_f, self.dims.n_u_f))
        for i in range(tau_f):
            for k in range(tau_f):
                if i > k:
                    H_u[i * p : (i + 1) * p, k * m : (k + 1) * m] = C @ np.linalg.matrix_power(A, i - k - 1) @ B
                if i == k:
                    H_u[i * p : (i + 1) * p, k * m : (k + 1) * m] = D

        return MPCPredictorMatrices(Gamma=Gamma, H_u=H_u)

    def calculate_regularization_matrices(self) -> MPCRegularizationMatrices:
        """Currently there is no regularization cost for the MPC algorithm."""
        return MPCRegularizationMatrices()

    def get_regularization_cost_expression(self) -> cp.Expression:
        """
        Currently there is no regularization cost implemented for the MPC algorithm.
        """
        return cp.Constant(0.0)

    def get_predictor_constraint_expression(self) -> cp.constraints.Constraint:
        r"""
        Calculates and returns the CVXPY constraint for the predictor constraint.

        The predictor constraint is calculated as follows:

        .. math::
            y_f = \Gamma x + H_u u_f

        Returns:
            cp.constraints.Constraint: The CVXPY constraint for the predictor constraint.
        """

        return (  # type: ignore
            self.y_f_cp == self.pred_matrices.Gamma @ self.x_cp + self.pred_matrices.H_u @ self.u_f_cp  # type: ignore
        )  # type: ignore

    @abstractmethod
    def calculate_system_data(self) -> MPCSystemMatrices:
        """
        Abstract method to calculate the system data, including the state-space matrices and Kalman filter gain.

        Returns:
            MPCSystemMatrices: An instance containing the system matrices and Kalman filter gain.
        """

    def set_state_x(self, x_new: np.ndarray):
        """
        Sets the state for the MPC controller.

        Args:
            x_new (np.ndarray): New state vector. Must be of shape (self.n_state,).


        Raises:
            ValueError: If `x_new` does not match the required shape.
        """
        if x_new.shape != (self.n_state,):
            raise ValueError(f"x_new must be of shape ({self.n_state},)")

        self.x_cp.value = x_new

    def solve(self, verbose: bool = False, solver: str = cp.SCS, **kwargs):
        """
        Solves the optimization problem for the MPC controller.

        This method performs the following steps:

            1. Updates the state using the Kalman Filter.
            2. If the problem is unconstrained, it uses the MPC closed-form gains.
               Otherwise, it solves the optimization problem as implemented in the DPC class.

        """
        u = self.z_p_cp.value[-self.dims.m :] # type: ignore
        y = self.z_p_cp.value[-self.dims.mp : -self.dims.m] # type: ignore

        # Update state with Kalman Filter
        self.x_cp.value = self.sys_matrices.sys.A @ self.x_cp.value + self.sys_matrices.sys.B @ u  # type: ignore
        self.x_cp.value = self.x_cp.value + self.sys_matrices.K @ (  # type: ignore
            y - (self.sys_matrices.sys.C @ self.x_cp.value + self.sys_matrices.sys.D @ u)  # type: ignore
        )

        if self.is_unconstrained:
            # If unconstrained, use the state feedback gain matrix K_x
            self.u_f = (
                self.mpc_cf_gains.K_x @ self.x_cp.value  # type: ignore
                + self.mpc_cf_gains.K_y_r @ self.y_r_cp.value  # type: ignore
                + self.mpc_cf_gains.K_u_r @ self.u_r_cp.value  # type: ignore
            )

            self.use_mpc_cf = True
            self.control_step = 0
        else:
            # For constrained problem, solve the built optimization problem
            super().solve(verbose=False, solver=cp.SCS, **kwargs)

    def calculate_closed_form_solution_matrices(self) -> Optional[DPCClosedFormSolutionMatrices]:
        """
        Cannot be implemented as the closed form is not calculated as:
        u_f^star = K_z_p z_p + ...
        """
        return None

    def _calculate_mpc_closed_form_solution_matrices(self) -> MPCClosedFormSolutionMatrices:
        """
        Calculates the closed-form gains for the MPC controller.

        This specific function calculates the gains s.t.:
        u_f^star = K_x*x + K_y_r*y_r + K_u_r*u_r

        Returns:
            MPCClosedFormSolutionMatrices: An instance containing the calculated gains.
        """
        H_u = self.pred_matrices.H_u  # type: ignore
        Gamma = self.pred_matrices.Gamma  # type: ignore
        Q_h = self.ddpc_params.Q_horizon
        R_h = self.ddpc_params.R_horizon

        F_1 = pinv(H_u.T @ Q_h @ H_u + R_h)
        F_2 = -H_u.T @ Q_h @ Gamma
        F_3 = H_u.T @ Q_h

        return MPCClosedFormSolutionMatrices(K_x=F_1 @ F_2, K_y_r=F_1 @ F_3, K_u_r=F_1 @ R_h)
