"""
Subspace Predictive Control (SPC) implementation.

This module implements SPC based on the methodology presented in:
- `SPC: Subspace Predictive Control <https://www.sciencedirect.com/science/article/pii/S1474667017566835>`

The SPC controller is a subclass of Data-Driven Predictive Control (DPC).

Classes:
    SPCOfflineData: Stores the Subspace Multistep Predictor matrix computed offline.
    SPC: Implements the SPC controller, handling optimization problem formulation and execution.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.linalg import pinv

from do_dpc.dpc.dpc import DPC
from do_dpc.dpc.dpc_structs import DPCClosedFormSolutionMatrices, DPCPredictorMatrices, DPCRegularizationMatrices

logger = logging.getLogger(__name__)


@dataclass
class SPCPredictorMatrices(DPCPredictorMatrices):
    """
    Stores precomputed offline data for Subspace Predictive Control (SPC).

    This data is used to predict future outputs based on past measurements and control inputs.

        y_f = S [z_p   u_f]^T  or equivalently,
        y_f = S_p z_p + S_u u_f

    Attributes:
        S (np.ndarray): The full subspace predictor matrix.
        S_p (np.ndarray): The predictor matrix associated with past measurements (z_p).
        S_u (np.ndarray): The predictor matrix associated with control inputs (u_f).
    """

    S: np.ndarray
    S_u: np.ndarray
    S_p: np.ndarray


@dataclass
class SPCRegularizationMatrices(DPCRegularizationMatrices):
    """Currently there is no regularization cost function for the SPC algorithm."""


@DPC.register("SPC")
class SPC(DPC):
    """
    Implements Subspace Predictive Control (SPC) based on DPC.
    """

    def calculate_predictor_matrices(self) -> SPCPredictorMatrices:
        """
        Computes the Subspace Multistep Predictor matrix `S` using Hankel matrices.

        Returns:
            SPCPredictorMatrices: An instance containing the computed predictor matrix `S`.
        """
        H_stacked = np.vstack((self.hankel_matrices.Z_p, self.hankel_matrices.U_f))
        S = self.hankel_matrices.Y_f @ H_stacked.T @ pinv(H_stacked @ H_stacked.T)
        S = np.real_if_close(S, tol=1e-3)

        S_u = S[:, self.dims.n_z_p :]
        S_p = S[:, : self.dims.n_z_p]

        return SPCPredictorMatrices(S=S, S_u=S_u, S_p=S_p)

    def calculate_regularization_matrices(self) -> SPCRegularizationMatrices:
        """Currently there is no regularization cost for the SPC algorithm."""
        return SPCRegularizationMatrices()

    def get_regularization_cost_expression(self) -> cp.Expression:
        """
        Currently there is no regularization cost implemented for the SPC algorithm
        """
        return cp.Constant(0)

    def get_predictor_constraint_expression(self) -> cp.Constraint:
        r"""
        Calculates and returns the CVXPY expression for the predictor constraint f.

        The predictor constraint is calculated as follows:

        .. math::
            y_f = S  \begin{bmatrix} z_p \\ u_f \end{bmatrix}

        Returns:
            cp.Constraint: The CVXPY constraint for the predictor constraint.

        Raises:
            ValueError: If any matrix in the constraint is not correctly defined.
        """

        try:
            return self.y_f_cp == self.pred_matrices.S @ cp.hstack((self.z_p_cp, self.u_f_cp))  # type: ignore
        except AttributeError as e:
            raise ValueError(f"A matrix in the predictor constraint is not correctly defined: {e}") from e

    def calculate_closed_form_solution_matrices(self) -> Optional[DPCClosedFormSolutionMatrices]:
        """
        Calculates and returns the closed-form gains for the DPC controller.

        The gains are calculated using the pseudo-inverse of the transformed matrices.

        Returns:
            Optional[DPCClosedFormSolutionMatrices]: The closed-form gains for the DPC controller,
        """
        S_u = self.pred_matrices.S_u  # type: ignore
        S_p = self.pred_matrices.S_p  # type: ignore
        Q_h = self.dpc_params.Q_horizon
        R_h = self.dpc_params.R_horizon

        F_1 = pinv(S_u.T @ Q_h @ S_u + R_h)
        F_2 = -S_u.T @ Q_h @ S_p
        F_3 = S_u.T @ Q_h

        return DPCClosedFormSolutionMatrices(K_z_p=F_1 @ F_2, K_y_r=F_1 @ F_3, K_u_r=F_1 @ R_h)
