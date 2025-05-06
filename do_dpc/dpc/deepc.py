# pylint: disable=line-too-long
"""
DeePC Implementation.

This module implements Data-Enabled Predictive Control (DeePC) based on the methodology presented in:

- `Data-Enabled Predictive Control: In the Shallows of the DeePC <https://ieeexplore.ieee.org/document/8795639>`_
- `Regularized and Distributionally Robust Data-Enabled Predictive Control <https://ieeexplore.ieee.org/document/9028943>`_

The DeePC controller is a subclass of Data-Driven Predictive Control (DPC).
It differs from other DPC methods by optimizing over a slack decision variable `g` of size (n_col, 1).
As n_col ~ n_samples for large n_samples, the computational performance is directly impacted by n_samples.
Leading to a computational complexity of :math:`O(n_{\text{samples}}^2)`.
With a large number of samples, performance degrades further due to memory limitations.

Performance Benchmark (MacBook Pro, Double Integrator)
------------------------------------------------------

+-------------------+-----------------------+
| Number of Samples | Total Simulation Time |
+===================+=======================+
| 100               | ~1 second             |
+-------------------+-----------------------+
| 300               | ~10 seconds           |
+-------------------+-----------------------+
| 1000              | ~4 minutes            |
+-------------------+-----------------------+

Note:
   100 samples are generally insufficient for the double integrator when process and measurement noise are present.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.linalg import pinv

from do_dpc.dpc.dpc import DPC
from do_dpc.dpc.dpc_structs import (
    DPCParameters,
    DPCClosedFormSolutionMatrices,
    DPCRegularizationMatrices,
    DPCPredictorMatrices,
)
from do_dpc.control_utils.control_structs import InputOutputTrajectory

logger = logging.getLogger(__name__)


@dataclass
class DeePCRegularizationMatrices(DPCRegularizationMatrices):
    """
    Stores precomputed regularization matrix for the DeePC controller.

    Attributes:
        I_minus_Pi (np.ndarray): The difference between the identity matrix and the projection matrix Pi.
    """

    I_minus_Pi: np.ndarray


@dataclass
class DeePCPredictorMatrices(DPCPredictorMatrices):
    """DeePC does not have a predictor as it utilizes directly the Hankel matrices."""


@dataclass
class DeePCSpecificParameters:
    """
    Stores and validates specific parameters for the DeePC controller.

    Attributes:
        lambda_sigma (float): Regularization parameter for consistency. Must be positive and nonzero.
        lambda_g_1 (float): Regularization parameter for robustness. Default is 0.
        lambda_g_2 (float): Regularization parameter for robustness. Default is 0.
        lambda_p (float): Regularization parameter for input penalties. Default is 0.
        suppress_warnings (bool): If True, suppresses warnings during initialization.
    """

    lambda_sigma: float = field(default=np.inf)
    lambda_g_1: float = field(default=0)
    lambda_g_2: float = field(default=0)
    lambda_p: float = field(default=0)
    suppress_warnings: bool = field(default=False, repr=False)

    def __post_init__(self):
        """
        Validates the tunable parameters after initialization and raises warnings if needed.

        Raises:
            ValueError: If any lambda value is negative.
            ValueError: If lambda_sigma is zero, as it must be positive.

            Warning: If lambda_sigma is set to np.inf, indicating potential infeasibility with noisy data.
            Warning: If lambda_g is 0, warning about potential robustness issues.
        """

        if not all(
            isinstance(val, (float, int))
            for val in [self.lambda_sigma, self.lambda_g_1, self.lambda_g_2, self.lambda_p]
        ):
            raise ValueError("All lambda parameters must be of type float or int.")

        if self.lambda_sigma < 0 or self.lambda_g_1 < 0 or self.lambda_p < 0 or self.lambda_g_2 < 0:
            raise ValueError("All lambda parameters must be non-negative.")

        if self.lambda_sigma == 0:
            raise ValueError("lambda_sigma must be positive and cannot be zero.")

        if self.suppress_warnings:
            return

        if self.lambda_sigma == np.inf:
            logger.warning(
                "lambda_sigma is set to np.inf. If there is noise in the data, "
                "there is a high chance that the problem will be infeasible. "
                "Further details in: https://ieeexplore.ieee.org/document/8795639/",
            )

        if self.lambda_g_1 == 0 and self.lambda_g_2 == 0:
            logger.warning(
                "lambda_g_1 and lambda_g_2 are both set to 0. This may lead to a non-robust controller. "
                "Further details in: https://ieeexplore.ieee.org/document/9028943/",
            )


# pylint: disable=too-many-instance-attributes
@DPC.register("DeePC")
class DeePC(DPC):
    """
    Implements DeePC based on DPC.

    Attributes:
        g_cp (cp.Variable): Slack decision variable used in the optimization.
        sigma_cp (cp.Parameter): Slack parameter, needed for feasibility in the presence of noise.
        specific_params (DeePCSpecificParameters): Stores tunable parameters for the controller.
        specific_params_set (bool): Flag indicating whether tunable parameters have been set.
    """

    def __init__(self, dpc_params: DPCParameters, training_data: InputOutputTrajectory):
        """
        Initializes the DeePC controller.
        """

        # Specific parameters
        self.specific_params = DeePCSpecificParameters(suppress_warnings=True)
        self.specific_params_set = False

        self.reg_matrices: DeePCRegularizationMatrices

        super().__init__(dpc_params, training_data)

        # Additional slack variables
        self.g_cp = cp.Variable(self.hankel_matrices.n_col)
        self.sigma_cp = self._construct_sigma()

    def calculate_predictor_matrices(self) -> DeePCPredictorMatrices:
        """DeePC does not have any predictor matrices"""
        return DeePCPredictorMatrices()

    def calculate_regularization_matrices(self) -> DeePCRegularizationMatrices:
        """
        Computes the regularization matrix for the DeePC controller.

        Returns:
            DeePCRegularizationMatrices
        """
        H_stacked = np.vstack((self.hankel_matrices.Z_p, self.hankel_matrices.U_f))
        H_stacked_dagger = H_stacked.T @ pinv(H_stacked @ H_stacked.T)
        Pi = H_stacked_dagger @ H_stacked

        return DeePCRegularizationMatrices(I_minus_Pi=np.eye(Pi.shape[0]) - Pi)

    def get_regularization_cost_expression(self) -> cp.Expression:
        r"""
        Calculates and returns the CVXPY expression for the regularization cost.

        The regularization cost is calculated as follows:

        .. math::
            r = \lambda_{g_1} \|g\|_1
                + \lambda_{g_2} \|g\|_2^2
                + \lambda_p \|I - \Pi g\|_2^2
                + \lambda_{\sigma} \|\sigma\|_2^2

        Returns:
            cp.Expression: The CVXPY expression for the regularization cost.
        """
        cost = cp.Constant(0.0)

        cost += self.specific_params.lambda_g_1 * cp.norm(self.g_cp, 1)
        cost += self.specific_params.lambda_g_2 * cp.norm(self.g_cp, 2) ** 2
        cost += (
            self.specific_params.lambda_p * cp.norm(self.reg_matrices.I_minus_Pi @ self.g_cp, 2) ** 2  # type: ignore
        )
        if self.specific_params.lambda_sigma != np.inf:
            cost += self.specific_params.lambda_sigma * cp.norm(self.sigma_cp, 2) ** 2

        return cost

    def get_predictor_constraint_expression(self) -> cp.Constraint:
        """
        Returns:
            cp.Constraint: CVXPY constraints for u_f and y_f depending on the Hankel matrices and `g`.
        """

        # Equality constraints for past horizon
        if self.specific_params.lambda_sigma == np.inf:
            z_p_expr = self.z_p_cp
        else:
            z_p_expr = self.z_p_cp - self.sigma_cp

        combined_constr = cp.hstack((z_p_expr, self.u_f_cp, self.y_f_cp)) == cp.hstack(  # type: ignore
            (  # type: ignore
                self.hankel_matrices.Z_p @ self.g_cp,  # type: ignore
                self.hankel_matrices.U_f @ self.g_cp,  # type: ignore
                self.hankel_matrices.Y_f @ self.g_cp,  # type: ignore
            )
        )

        return combined_constr  # type: ignore

    def calculate_closed_form_solution_matrices(self) -> Optional[DPCClosedFormSolutionMatrices]:
        """
        Computes the closed-form gain matrices.

        Returns:
            Optional[DPCClosedFormSolutionMatrices]: The computed gain matrices if conditions are met; otherwise, None.

        Conditions:
            - If `lambda_g_1` is nonzero, return None (1-norm differentiation issue).
            - If `lambda_sigma` is infinite, return None.
            - If `lambda_p` is zero, return None.
        """
        if self.specific_params.lambda_g_1 != 0:
            return None

        if self.specific_params.lambda_sigma == np.inf:
            return None

        if self.specific_params.lambda_p == 0:
            return None

        I_minus_Pi = self.reg_matrices.I_minus_Pi  # type: ignore
        U_f = self.hankel_matrices.U_f
        Y_f = self.hankel_matrices.Y_f
        Z_p = self.hankel_matrices.Z_p

        Q_h = self.dpc_params.Q_horizon
        R_h = self.dpc_params.R_horizon

        T_1 = Y_f.T @ Q_h @ Y_f + U_f.T @ R_h @ U_f
        T_2 = self.specific_params.lambda_g_2 * np.eye(self.hankel_matrices.n_col) + self.specific_params.lambda_p * I_minus_Pi.T @ I_minus_Pi
        T_3 = self.specific_params.lambda_sigma * Z_p.T @ Z_p

        F_1 = U_f @ pinv(T_1 + T_2 + T_3)
        F_2 = self.specific_params.lambda_sigma * Z_p.T
        F_3 = Y_f.T @ Q_h
        F_4 = U_f.T @ R_h

        return DPCClosedFormSolutionMatrices(K_z_p=F_1 @ F_2, K_y_r=F_1 @ F_3, K_u_r=F_1 @ F_4)

    def set_specific_parameters(self, specific_params: DeePCSpecificParameters):
        """
        Sets the specific parameters for the DeePC controller.

        Note:
            The closed-form data will be recalculated as the matrices depend on lambda_.

        Args:
            specific_params: DeePCSpecificParameters
        """
        self.specific_params_set = True
        self.specific_params = specific_params

        self.cf_matrices = self.calculate_closed_form_solution_matrices()

    def build_optimization_problem(self):
        """
        Raises:
            Warning: If tunable parameters have not been explicitly set using set_specific_parameters().
        """
        if not self.specific_params_set:
            logger.warning(
                "Tunable parameters have not been explicitly set using set_specific_parameters(). "
                "Default values are being used, which may lead to unexpected behavior."
            )

        super().build_optimization_problem()

    def _construct_sigma(self) -> cp.Expression:
        """
        Constructs the sigma vector of the specified form:
        sigma_i in p, where each sigma_i is a cp.Variable(p),
        and sigma is structured as [sigma_1, 0_m, sigma_2, ..., sigma_tau_p, 0_m]^T.
        """
        sigma_vars = [cp.Variable(self.dims.p) for _ in range(self.dpc_params.tau_p)]

        sigma_list = []
        zero_vec = cp.Constant(np.zeros((self.dims.m,)))

        for sigma_i in sigma_vars:
            sigma_list.append(sigma_i)
            sigma_list.append(zero_vec)  # type: ignore[arg-type]

        sigma = cp.hstack(sigma_list)

        return sigma
