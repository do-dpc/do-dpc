"""
Transient Predictive Control (TPC) implementation.

This module implements TPC based on the methodology presented in:
- `The Transient Predictor <https://www.research-collection.ethz.ch/handle/20.500.11850/716622>`
- `On the impact of regularization in data-driven predictive control <https://arxiv.org/abs/2304.00263>`


The TPC controller is a subclass of Data-Driven Predictive Control (DPC).

Classes:
    TPC: Implements the TPC controller, handling optimization problem formulation and execution.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import cvxpy as cp
import numpy as np
from numpy.linalg import pinv

from do_dpc.dpc.dpc import DPC
from do_dpc.dpc.dpc_structs import (
    DPCClosedFormSolutionMatrices,
    DPCParameters,
    DPCRegularizationMatrices,
    DPCPredictorMatrices,
)
from do_dpc.control_utils.control_structs import InputOutputTrajectory
from do_dpc.dpc.dpc_utils import DPCUtils

logger = logging.getLogger(__name__)


@dataclass
class TPCPredictorMatrices(DPCPredictorMatrices):
    """
    Stores predictor matrices used for future control inputs and past system states.

    This class contains the matrices that are used to predict the future control inputs
    (`H_u`) and the past system states (`H_p`) for trajectory predictive control.

    Attributes:
        H_u (np.ndarray): Multistep predictor matrix for future control inputs.
        H_p (np.ndarray): Multistep predictor matrix for past system states.
    """

    H_u: np.ndarray
    H_p: np.ndarray


@dataclass
class TPCRegularizationMatrices(DPCRegularizationMatrices):
    """
    Stores regularization matrices for control inputs, states, and reference outputs.

    This class contains the regularization matrices used in the trajectory predictive control
    to penalize the control inputs (`Lambda_uu`), the coupling between control inputs and past
    states (`Lambda_uz`), and the coupling between control inputs and reference outputs (`Lambda_uy`).

    Attributes:
        Lambda_uu (np.ndarray): Regularization matrix for control inputs.
        Lambda_uz (np.ndarray): Regularization matrix coupling control inputs and past states.
        Lambda_uy (np.ndarray): Regularization matrix coupling control inputs and reference outputs.
    """

    Lambda_uu: np.ndarray
    Lambda_uz: np.ndarray
    Lambda_uy: np.ndarray


@dataclass
class TPCHelperMatrices:
    """
    Stores helper matrices used in the trajectory predictive control (TPC) calculations.

    This class holds matrices such as `W` (used for observability properties), `L` (the LQ decomposition
    matrix), and the matrices related to system dynamics (`Phi_P`, `Phi_U`).

    Attributes:
        W (np.ndarray): Matrix used for system observability properties.
        L (np.ndarray): Matrix from the LQ decomposition.
        Phi_P (np.ndarray): Portion of the `Phi` matrix corresponding to past inputs & outputs.
        Phi_U (np.ndarray): Portion of the `Phi` matrix corresponding to future inputs.
    """

    W: np.ndarray
    L: np.ndarray
    Phi_P: np.ndarray
    Phi_U: np.ndarray


@DPC.register("TPC")
class TPC(DPC):
    """
    Implements Transient Predictive Control based on DPC.

    Attributes:
        tpc_helper_matrices(TPCHelperMatrices): Holds helper matrices.
    """

    def __init__(self, dpc_params: DPCParameters, training_data: InputOutputTrajectory):
        """
        Initializes the DeePC controller.
        """
        self.tpc_helper_matrices: Optional[TPCHelperMatrices] = None

        super().__init__(dpc_params, training_data)

    def get_regularization_cost_expression(self) -> cp.Expression:
        r"""
        Calculates and returns the CVXPY expression for the regularization cost r.

        The regularization cost is calculated as follows:

        .. math::
            r(u_f, z_p, y_r) = u_f^T \Lambda_{uu} u_f + 2 u_f^T \Lambda_{uz} z_p + 2 u_f^T \Lambda_{uy} y_r

        Returns:
            cp.Expression: The CVXPY expression for the regularization cost.

        Raises:
            Warning: If Lambda_uu is not positive semidefinite.
        """

        cost = cp.Constant(0.0)

        if DPCUtils.is_positive_semidefinite(self.reg_matrices.Lambda_uu, tolerance=1e-10):  # type: ignore
            cost += cp.quad_form(self.u_f_cp, self.reg_matrices.Lambda_uu)  # type: ignore
            cost += 2 * self.u_f_cp.T @ self.reg_matrices.Lambda_uz @ self.z_p_cp  # type: ignore
            cost += 2 * self.u_f_cp.T @ self.reg_matrices.Lambda_uy @ self.y_r_cp  # type: ignore
        else:
            logger.warning(
                "Lambda_uu is not positive semidefinite. Regularization cost is skipped to avoid non-convex cost."
            )

        return cost

    def get_predictor_constraint_expression(self) -> cp.constraints.Constraint:
        r"""
        Calculates and returns the CVXPY expression for the predictor constraint f.

        The predictor constraint is calculated as follows:

        .. math::
            y_f = H_u u_f + H_p z_p

        Returns:
            cp.constraints.Constraint: The CVXPY constraint for the predictor constraint.

        Raises:
            ValueError: If any matrix in the constraint is not correctly defined.
        """

        try:
            return (  # type: ignore
                self.y_f_cp  # type: ignore
                == self.pred_matrices.H_u @ self.u_f_cp + self.pred_matrices.H_p @ self.z_p_cp  # type: ignore
            )  # type: ignore
        except AttributeError as e:
            raise ValueError(f"A matrix in the predictor constraint is not correctly defined: {e}") from e

    def calculate_predictor_matrices(self) -> TPCPredictorMatrices:
        """
        Calculates and returns the predictor matrices for future control inputs and past system states.

        Uses the helper matrices `W`, `Phi_U`, and `Phi_P` to compute `H_u` and `H_p`.

        Returns:
            TPCPredictorMatrices: Containing the matrices `H_u` and `H_p`.
        """
        if not self.tpc_helper_matrices:
            self.tpc_helper_matrices = self._calculate_helper_matrices()

        H_u = self.tpc_helper_matrices.W @ self.tpc_helper_matrices.Phi_U  # type: ignore
        H_p = self.tpc_helper_matrices.W @ self.tpc_helper_matrices.Phi_P  # type: ignore

        return TPCPredictorMatrices(H_u=H_u, H_p=H_p)

    def calculate_regularization_matrices(self) -> TPCRegularizationMatrices:
        """
        Calculates and returns the regularization matrices for control inputs, states, and outputs.

        Uses the helper matrices `L` and `W` to compute `Lambda_uu`, `Lambda_uy`, and `Lambda_uz`
        with regularization terms and system dynamics.

        Returns:
            TPCRegularizationMatrices: Containing the regularization matrices `Lambda_uu`, `Lambda_uy`, and `Lambda_uz`.
        """
        L = self.tpc_helper_matrices.L  # type: ignore
        W = self.tpc_helper_matrices.W  # type: ignore

        Qw = pinv(W).T @ self.dpc_params.Q_horizon @ pinv(W)

        Sigma_Phi = self._calculate_sigma_phi(L)

        Lambda_uu, Lambda_uy, Lambda_uz = self._calculate_lambda_matrices(Qw, Sigma_Phi)

        return TPCRegularizationMatrices(Lambda_uu=Lambda_uu, Lambda_uy=Lambda_uy, Lambda_uz=Lambda_uz)

    def calculate_closed_form_solution_matrices(self) -> DPCClosedFormSolutionMatrices:
        """
        Calculates and returns the closed-form gains for the DPC controller.

        The gains are calculated using the pseudo-inverse of the transformed matrices.

        Returns:
            Optional[DPCClosedFormSolutionMatrices]: The closed-form gains for the DPC controller,
        """
        H_u = self.pred_matrices.H_u  # type: ignore
        H_p = self.pred_matrices.H_p  # type: ignore

        Q_horizon = self.dpc_params.Q_horizon
        R_horizon = self.dpc_params.R_horizon

        Lambda_uu = self.reg_matrices.Lambda_uu  # type: ignore
        Lambda_uy = self.reg_matrices.Lambda_uy  # type: ignore
        Lambda_uz = self.reg_matrices.Lambda_uz  # type: ignore

        F_1 = np.linalg.pinv(H_u.T @ Q_horizon @ H_u + R_horizon + Lambda_uu)
        F_2 = -H_u.T @ Q_horizon @ H_p - Lambda_uz
        F_3 = H_u.T @ Q_horizon - Lambda_uy

        return DPCClosedFormSolutionMatrices(K_z_p=F_1 @ F_2, K_y_r=F_1 @ F_3, K_u_r=F_1 @ R_horizon)

    def _calculate_helper_matrices(self) -> TPCHelperMatrices:
        """
        Computes helper matrices required for control computations.

        This function performs the following steps:
        1. Computes the LQ decomposition of the Hankel matrix `Z`.
        2. Constructs structured index arrays (`indy` and `indu`) to select specific matrix rows.
        3. Computes `Phi`, which transforms data using the pseudo-inverse of `L`.
        4. Extracts submatrices `Phi_P`, `Phi_U`, and `Phi_Y` from `Phi`.
        5. Computes matrix `W`, which accounts for system observability properties.

        Returns:
            TPCHelperMatrices: A dataclass containing matrices `W`, `L`, `Phi_P`, and `Phi_U`.
        """

        L = DPCUtils.save_lq_decomposition(self.hankel_matrices.Z)

        Phi = self._extract_L0y_matrix(L) @ np.linalg.pinv(L)

        Phi_P, Phi_U, Phi_Y = self._split_Phi_into_PhiP_PhiUF_PhiYF(Phi)

        W = pinv(np.eye(Phi_Y.shape[0]) - Phi_Y)

        return TPCHelperMatrices(W=W, L=L, Phi_P=Phi_P, Phi_U=Phi_U)

    def _extract_L0y_matrix(self, L: np.ndarray) -> np.ndarray:
        """
        Extracts the L0y matrix from the L matrix.

        L0 is the L matrix with the diagonal block matrices set to zero.
        L0y is the L0 matrix rows corresponding to y_f.

        Args:
            L (np.ndarray): The lower triangular matrix obtained from LQ decomposition.

        Returns:
            np.ndarray: The modified matrix `L_0y`.
        """

        L_0 = L.copy()

        # Zeroing out specific blocks more efficiently
        for i in range(self.dpc_params.tau_f):
            start = self.dims.n_z_p + i * self.dims.mp
            end = start + self.dims.p
            L_0[start:end, start:end] = 0

        # Extracting relevant rows from L_0
        indy0 = np.add.outer(
            np.arange(0, self.dims.n_y_f + self.dims.n_u_f, self.dims.mp), np.arange(self.dims.p)
        ).ravel()
        L_0y = L_0[self.dims.n_z_p + indy0, :]

        return L_0y

    def _extract_L22y_matrix(self, L: np.ndarray) -> np.ndarray:
        """
        Extracts the L22 matrix part from the L matrix corresponding to the y_f.

        Args:
            L (np.ndarray): The lower triangular matrix obtained from LQ decomposition.

        Returns:
            np.ndarray: The modified matrix `L22`.
        """

        indy = (
            np.array([np.arange(1, self.dims.p + 1) + self.dims.mp * i for i in range(self.dpc_params.tau_f)]).flatten()
            - 1
        )
        return L[np.ix_(self.dims.n_z_p + indy, self.dims.n_z_p + indy)]

    def _split_Phi_into_PhiP_PhiUF_PhiYF(self, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the matrix `Phi` into three submatrices: `Phi_P`, `Phi_UF`, and `Phi_YF`.

        Args:
            Phi (np.ndarray): The matrix to be split.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - `Phi_P`: The portion of `Phi` corresponding to past inputs & outputs.
                - `Phi_UF`: The portion of `Phi` corresponding to future inputs.
                - `Phi_YF`: The portion of `Phi` corresponding to system outputs.
        """

        indy = np.add.outer(
            np.arange(0, self.dims.n_y_f + self.dims.n_u_f, self.dims.mp), np.arange(self.dims.p)
        ).ravel()
        indu = np.add.outer(
            np.arange(self.dims.p, self.dims.n_y_f + self.dims.n_u_f, self.dims.mp), np.arange(self.dims.m)
        ).ravel()

        return Phi[:, : self.dims.n_z_p], Phi[:, self.dims.n_z_p + indu], Phi[:, self.dims.n_z_p + indy]

    def _calculate_sigma_phi(self, L: np.ndarray) -> np.ndarray:
        """
        Calculates the matrix `Sigma_Phi` based on the permutation matrix and `Sigma_Theta`.

        Uses the helper matrix `L` and the computed `Sigma_Theta` to calculate `Sigma_Phi`.

        Args:
            L (np.ndarray): Matrix used to calculate `Sigma_Phi`.

        Returns:
            np.ndarray: The computed `Sigma_Phi` matrix.
        """
        Sigma_Theta = self._calculate_sigma_theta(L)

        # Build the permutation matrix P
        k = self.dims.mp * (self.dpc_params.tau_p + self.dpc_params.tau_f)
        P = np.kron(np.eye(self.dpc_params.tau_f), np.eye(k))
        P = np.kron(P, np.eye(self.dims.p))

        return P @ Sigma_Theta @ P.T

    def _calculate_sigma_theta(self, L: np.ndarray) -> np.ndarray:
        # pylint: disable=too-many-locals
        """
        Calculates the matrix `Sigma_Theta` used for regularization.

        Uses the matrix `L` to compute the blocks of `Sigma_Theta`, which is essential for
        regularization in trajectory predictive control. It accounts for errors, states, and control inputs.

        Args:
            L (np.ndarray): Matrix used to calculate `Sigma_Theta`.

        Returns:
            np.ndarray: The computed `Sigma_Theta` matrix.
        """
        tau_f = self.dpc_params.tau_f
        tau_p = self.dpc_params.tau_p
        n_col = self.hankel_matrices.n_col
        mp = self.dims.mp
        p = self.dims.p

        L22y = self._extract_L22y_matrix(L)

        Sigma_epsilon = []
        for i in range(tau_f):
            block = L22y[p * i : p * (i + 1), p * i : p * (i + 1)]
            sigma = block @ block.T
            sigma *= n_col / (n_col - mp * (tau_p + i))
            Sigma_epsilon.append(sigma)

        # Initialize Sigma_Theta
        k_block = p * mp * (tau_p + tau_f)
        Sigma_Theta = np.zeros((tau_f * k_block, tau_f * k_block))

        for i in range(tau_f):
            Sigmai = L[: mp * (tau_p + i), : mp * (tau_p + i)]
            Sigmai = Sigmai @ Sigmai.T

            for j in range(i, tau_f):
                Sigmaijbar = L[(j - i) * mp : mp * (tau_p + j), : mp * (tau_p + j)]
                Sigmaijbar = Sigmaijbar @ pinv(L[: mp * (tau_p + j), : mp * (tau_p + j)])
                Qij = pinv(Sigmai) @ Sigmaijbar

                Sigma_tildeTheta_block = np.zeros((k_block, k_block))
                Sigma_tildeTheta_block[: p * mp * (tau_p + i), : p * mp * (tau_p + j)] = (
                    (n_col - (j - i)) / n_col**2 * np.kron(Qij, Sigma_epsilon[i])
                )

                Sigma_Theta[i * k_block : (i + 1) * k_block, j * k_block : (j + 1) * k_block] = Sigma_tildeTheta_block
                if j > i:
                    row_start_j = j * k_block
                    row_end_j = (j + 1) * k_block
                    col_start_i = i * k_block
                    col_end_i = (i + 1) * k_block
                    Sigma_Theta[row_start_j:row_end_j, col_start_i:col_end_i] = Sigma_tildeTheta_block.T
        return Sigma_Theta

    def _calculate_lambda_matrices(
        self, Qw: np.ndarray, Sigma_Phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # pylint: disable=too-many-locals
        """
        Calculates the regularization matrices `Lambda_uu`, `Lambda_uy`, and `Lambda_uz`.

        This method computes the optimal regularization matrices for control inputs,
        states, and outputs based on the matrices `Qw` and `Sigma_Phi`.

        Args:
            Qw (np.ndarray): Weighting matrix used for regularization.
            Sigma_Phi (np.ndarray): Matrix derived from system dynamics used for regularization.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The regularization matrices.
        """
        tau_f = self.dpc_params.tau_f

        indu = (
            np.array(
                [
                    self.dims.n_z_p + np.arange(self.dims.p + 1, self.dims.mp + 1) + i * self.dims.mp
                    for i in range(tau_f)
                ]
            ).flatten()
            - 1
        )
        indy = (
            np.array(
                [self.dims.n_z_p + np.arange(1, self.dims.p + 1) + i * self.dims.mp for i in range(tau_f)]
            ).flatten()
            - 1
        )
        indz = np.arange(self.dims.n_z_p)

        Lambda_uu, Lambda_uz, Lambda_uy = (
            np.zeros((len(indu), len(indu))),
            np.zeros((len(indu), len(indz))),
            np.zeros((len(indu), len(indy))),
        )

        for i, indu_i in enumerate(indu):
            ii = np.arange(self.dims.n_y_f * indu_i, self.dims.n_y_f * (indu_i + 1))  # type: ignore

            for j, indz_j in enumerate(indz):
                jj = np.arange(self.dims.n_y_f * indz_j, self.dims.n_y_f * (indz_j + 1))  # type: ignore
                Lambda_uz[i, j] = np.trace(Qw @ Sigma_Phi[np.ix_(ii, jj)])

            for j, indu_j in enumerate(indu):
                jj = np.arange(self.dims.n_y_f * indu_j, self.dims.n_y_f * (indu_j + 1))  # type: ignore
                Lambda_uu[i, j] = np.trace(Qw @ Sigma_Phi[np.ix_(ii, jj)])

            for j, indy_j in enumerate(indy):
                jj = np.arange(self.dims.n_y_f * indy_j, self.dims.n_y_f * (indy_j + 1))  # type: ignore
                Lambda_uy[i, j] = np.trace(Qw @ Sigma_Phi[np.ix_(ii, jj)])

        Lambda_uu = (Lambda_uu + Lambda_uu.T) / 2  # Symmetrize Lambda_uu

        return Lambda_uu, Lambda_uy, Lambda_uz
