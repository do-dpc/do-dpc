# pylint: disable=line-too-long
"""
gamma-DPC implementation.

This module implements gamma-DPC based on the methodology presented in:

- `Uncertainty-aware data-driven predictive control in a stochastic setting <https://arxiv.org/pdf/2211.10321>`_
- `Data-driven predictive control in a stochastic setting: a unified framework <https://www.sciencedirect.com/science/article/pii/S0005109823001139>`_

The gamma-DPC controller is a subclass of Data-Driven Predictive Control (DPC).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.linalg import pinv

from do_dpc.dpc.dpc import DPC
from do_dpc.dpc.dpc_structs import (
    DPCParameters,
    DPCClosedFormSolutionMatrices,
    DPCPredictorMatrices,
    DPCRegularizationMatrices,
)
from do_dpc.control_utils.control_structs import InputOutputTrajectory
from do_dpc.dpc.dpc_utils import DPCUtils

logger = logging.getLogger(__name__)


@dataclass
class GammaDDPCPredictorMatrices(DPCPredictorMatrices):
    r""""
    Stores predictor matrices for the gamma-DPC controller.

    The LQ decomposition of the Hankel Matrix is represented as:

    .. math::
        L =
        \begin{pmatrix}
        L_{11} & 0 & 0 \\
        L_{21} & L_{22} & 0 \\
        L_{31} & L_{32} & L_{33}
        \end{pmatrix}

    and the concatenated matrix :math:`L_{2,3}` is:

    .. math::
        L_{2,3} =
        \begin{pmatrix}
        L_{21} & L_{22} \\
        L_{31} & L_{32}
        \end{pmatrix}

    Attributes:
        pinv_L_11 (np.ndarray): Pseudo-inverse of :math:`L_{11}`.
        L_2_3 (np.ndarray): Concatenated matrix from :math:`L_{21}, L_{22}, L_{31}, L_{32}`.
        L_21 (np.ndarray): Part of :math:`L` representing :math:`\gamma_1` and :math:`U_f`.
        L_22 (np.ndarray): Part of :math:`L` representing :math:`\gamma_2` and :math:`U_f`.
        L_31 (np.ndarray): Part of :math:`L` representing :math:`\gamma_1` and :math:`Y_f`.
        L_32 (np.ndarray): Part of :math:`L` representing :math:`\gamma_2` and :math:`Y_f`.
    """

    pinv_L_11: np.ndarray
    L_2_3: np.ndarray
    L_21: np.ndarray
    L_22: np.ndarray
    L_31: np.ndarray
    L_32: np.ndarray


@dataclass
class GammaDDPCRegularizationMatrices(DPCRegularizationMatrices):
    """Currently there is no regularization cost function for the Gamma DPC algorithm."""


@DPC.register("GammaDDPC")
class GammaDDPC(DPC):
    """
    Implements gamma-DPC based on DPC.

    Attributes:
        gamma_2_cp (cp.Variable): Slack decision variable for the gamma parameter in the optimization.
        lambda_gamma_2 (float): Tunable parameter associated with the gamma-2 term.
    """

    def __init__(self, dpc_params: DPCParameters, training_data: InputOutputTrajectory):
        """
        Initializes the gamma DPC controller.
        """
        super().__init__(dpc_params, training_data)

        # Additional slack variables
        self.gamma_2_cp = cp.Variable(self.dims.n_u_f)

        # Tunable parameters
        self.lambda_gamma_2 = 0.0

    def calculate_predictor_matrices(self) -> GammaDDPCPredictorMatrices:
        """
        Computes the L matrices.

        Returns:
            GammaDDPCOfflineData
        """
        H_stacked = np.vstack((self.hankel_matrices.Z_p, self.hankel_matrices.U_f, self.hankel_matrices.Y_f))

        L = DPCUtils.save_lq_decomposition(H_stacked @ H_stacked.T)

        mp_taup = self.dims.n_z_p
        mp_taup_m_tauf = self.dims.n_z_p + self.dims.n_u_f
        mp_taup_mp_tauf = self.dims.n_z_p + self.dims.n_u_f + self.dims.n_y_f

        L_11 = L[:mp_taup, :mp_taup]

        L_21 = L[mp_taup:mp_taup_m_tauf, :mp_taup]
        L_22 = L[mp_taup:mp_taup_m_tauf, mp_taup:mp_taup_m_tauf]

        L_31 = L[mp_taup_m_tauf:mp_taup_mp_tauf, :mp_taup]
        L_32 = L[mp_taup_m_tauf:mp_taup_mp_tauf, mp_taup:mp_taup_m_tauf]

        # L_33 = L[mp_taup_m_tauf:mp_taup_mp_tauf, mp_taup_m_tauf:mp_taup_mp_tauf]

        L_2_3 = np.vstack((np.hstack((L_21, L_22)), np.hstack((L_31, L_32))))
        return GammaDDPCPredictorMatrices(pinv_L_11=pinv(L_11), L_2_3=L_2_3, L_21=L_21, L_22=L_22, L_31=L_31, L_32=L_32)

    def calculate_regularization_matrices(self) -> GammaDDPCRegularizationMatrices:
        return GammaDDPCRegularizationMatrices()

    def get_regularization_cost_expression(self) -> cp.Expression:
        r"""
        Calculates and returns the CVXPY expression for the regularization cost.

        The regularization cost is calculated as follows:

        .. math::
            \text{cost} = \lambda_{\gamma_2} \|\gamma_2\|_2^2

        Returns:
            cp.Expression: The CVXPY expression for the regularization cost.
        """
        cost = cp.Constant(0.0)
        cost += self.lambda_gamma_2 * cp.norm(self.gamma_2_cp, 2) ** 2
        return cost

    def get_predictor_constraint_expression(self) -> cp.Constraint:
        r"""
        Calculates and returns the CVXPY constraint for the predictor constraint.

        The predictor constraint is calculated as follows:

        .. math::
            \gamma_1 = L_{11}^{-1} z_p \\
            \begin{bmatrix} u_f \\ y_f \end{bmatrix} = L_{23} \begin{bmatrix} \gamma_1 \\ \gamma_2 \end{bmatrix}

        Returns:
            cp.Constraint: The CVXPY constraint for the predictor constraint.
        """

        # Setting gamma_1 = L_11^-1 z_p
        gamma_1 = self.pred_matrices.pinv_L_11 @ self.z_p_cp  # type: ignore

        gamma = cp.hstack((gamma_1, self.gamma_2_cp))  # type: ignore

        # Equality constraints for u_f, y_f
        return cp.hstack((self.u_f_cp, self.y_f_cp)) == self.pred_matrices.L_2_3 @ gamma  # type: ignore

    def calculate_closed_form_solution_matrices(self) -> Optional[DPCClosedFormSolutionMatrices]:
        """
        Calculates and returns the closed-form gains for the DPC controller.

        The gains are calculated using the pseudo-inverse of the transformed matrices.

        Returns:
            Optional[DPCClosedFormSolutionMatrices]: The closed-form gains for the DPC controller,
        """
        # pylint: disable=too-many-locals
        pinv_L_11 = self.pred_matrices.pinv_L_11  # type: ignore
        L_21 = self.pred_matrices.L_21  # type: ignore
        L_22 = self.pred_matrices.L_22  # type: ignore
        L_31 = self.pred_matrices.L_31  # type: ignore
        L_32 = self.pred_matrices.L_32  # type: ignore

        Q_h = self.dpc_params.Q_horizon
        R_h = self.dpc_params.R_horizon

        lambda_gamma_2 = getattr(self, "lambda_gamma_2", 0)

        T_1 = pinv(L_22) @ L_21 @ pinv_L_11
        T_2 = pinv(L_22)

        T_3 = L_31 @ pinv_L_11 - L_32 @ T_1
        T_4 = L_32 @ T_2

        F_1 = pinv(T_4.T @ Q_h @ T_4 + R_h + lambda_gamma_2 * T_2.T @ T_2)
        F_2 = -T_4.T @ Q_h @ T_3 + lambda_gamma_2 * T_2.T @ T_1
        F_3 = T_4.T @ Q_h

        return DPCClosedFormSolutionMatrices(K_z_p=F_1 @ F_2, K_y_r=F_1 @ F_3, K_u_r=F_1 @ R_h)

    def set_lambda_gamma_2(self, new_lambda_gamma_2: float):
        """Set the value of lambda_gamma_2, ensuring it is positive.

        Args:
            new_lambda_gamma_2 (float): The new value for lambda_gamma_2. Must be a positive number.

        Note:
            The closed-form data will be recalculated as the matrices depend on lambda_gamma_2.

        Raises:
            TypeError: If new_lambda_gamma_2 is not a number.
            ValueError: If new_lambda_gamma_2 is not positive.
        """
        if not isinstance(new_lambda_gamma_2, (int, float)):
            raise TypeError("new_lambda_gamma_2 must be a number.")
        if new_lambda_gamma_2 <= 0:
            raise ValueError("new_lambda_gamma_2 must be a positive number.")

        self.lambda_gamma_2 = new_lambda_gamma_2

        self.cf_matrices = self.calculate_closed_form_solution_matrices()
