"""
Shared dataclasses for data-driven control.

Defines common data structures used across different modules.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import block_diag  # type: ignore


@dataclass
class DPCPredictorMatrices:
    """
    Base class for storing matrices used in Data-Driven Predictive Control (DPC) to predict `y_f`.

    Note:
        Subclasses should extend it with relevant optimization data.
    """


@dataclass
class DPCRegularizationMatrices:
    """
    Base class for storing matrices used in Data-Driven Predictive Control (DPC) for additional cost.

    Note:
        Subclasses should extend it with relevant optimization data.
    """


@dataclass
class DPCClosedFormSolutionMatrices:
    """
    Data class for storing closed-form gains.

    Attributes:
        K_z_p (np.ndarray): Gain matrix for past measurements of shape (n_u_f, n_z_p).
        K_y_r (np.ndarray): Gain matrix for output reference trajectory of shape (n_u_f, n_y_f).
        K_u_r (np.ndarray): Gain matrix for input reference trajectory of shape (n_u_f, n_u_f).
    """

    K_z_p: np.ndarray
    K_y_r: np.ndarray
    K_u_r: np.ndarray


@dataclass
class DPCParameters:
    # pylint: disable=too-many-instance-attributes
    """
    Stores parameters for a data-driven predictive controller.

    Attributes:
        Q (np.ndarray): State weighting matrix for the cost function (shape `(p, p)`).
        R (np.ndarray): Control input weighting matrix for the cost function (shape `(m, m)`).
        tau_p (int): Initial time horizon for past data.
        tau_f (int): Future time horizon for predictions.
        Q_horizon (np.ndarray): Block-diagonal matrix of Q repeated for `tau_f` steps.
        R_horizon (np.ndarray): Block-diagonal matrix of R repeated for `tau_f` steps.
        Q_final (Optional[np.ndarray]): Final state cost. If None, same as Q (shape `(p, p)`).
        R_final (Optional[np.ndarray]): Final input cost. If None, same as R (shape `(m, m)`).
        R_delta (Optional[np.ndarray]): Weight cost for the Delta u_f (Change in control input) (shape `(m, m)`).
        R_delta_first (Optional[np.ndarray]): Weight cost for u_p(0)-u_f(0) (Change in control input) (shape `(m, m)`).

    Note:
        If `R_delta_first` is None but `R_delta` is provided, `R_delta` will be used as the first value.

    Raises:
        ValueError: If `tau_p` or `tau_f` are not positive integers.
        ValueError: If `Q` or `R` are not square matrices of expected shape.
        ValueError: If `R_final`, `R_delta_first`, `R_delta` do not have the same shape as `R`.
        ValueError: If `Q_final` does not have the same shape as `Q`.
        ValueError: If `Q`, `Q_final` is not positive semidefinite.
        ValueError: If `R`, `R_final`, `R_delta`, R_delta_first` is not positive definite.
    """

    Q: np.ndarray
    R: np.ndarray
    tau_p: int
    tau_f: int
    Q_horizon: np.ndarray = field(init=False)
    R_horizon: np.ndarray = field(init=False)
    Q_final: Optional[np.ndarray] = None
    R_final: Optional[np.ndarray] = None
    R_delta: Optional[np.ndarray] = None
    R_delta_horizon: Optional[np.ndarray] = field(init=False)
    R_delta_first: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Validates inputs and constructs block-diagonal matrices for Q and R over the prediction horizon.
        Sets `Q_final`, `R_final` into `Q_horizon`, `R_horizon`.
        """

        # Validate horizon lengths
        for name, tau in {"tau_p": self.tau_p, "tau_f": self.tau_f}.items():
            if not isinstance(tau, int) or tau <= 0:
                raise ValueError(f"{name} must be a positive integer, but got {tau}.")

        # Validate matrices Q and R
        self._validate_square_matrix(self.Q, "Q")
        self._validate_square_matrix(self.R, "R")
        self._check_positive_definite(self.Q, "Q", semi=True)
        self._check_positive_definite(self.R, "R")

        # Construct block-diagonal matrices for Q and R over tau_f steps
        self.Q_horizon = block_diag(*([self.Q] * self.tau_f))
        self.R_horizon = block_diag(*([self.R] * self.tau_f))

        # Validate and apply Q_final if provided
        if self.Q_final is not None:
            self._validate_square_matrix(self.Q_final, "Q_final")
            self._check_positive_definite(self.Q_final, "Q_final", semi=True)
            if self.Q_final.shape != self.Q.shape:
                raise ValueError(
                    f"Q_final must have the same shape as Q: {self.Q.shape}, " f"but got {self.Q_final.shape}."
                )

            self.Q_horizon[-self.Q.shape[0] :, -self.Q.shape[1] :] = self.Q_final

        # Validate and apply R_final if provided
        if self.R_final is not None:
            self._validate_square_matrix(self.R_final, "R_final")
            self._check_positive_definite(self.R_final, "R_final")
            if self.R_final.shape != self.R.shape:
                raise ValueError(
                    f"R_final must have the same shape as R: {self.R.shape}, " f"but got {self.R_final.shape}."
                )

            self.R_horizon[-self.R.shape[0] :, -self.R.shape[1] :] = self.R_final

        # Validate and apply R_delta
        if self.R_delta is not None:
            self._validate_square_matrix(self.R_delta, "R_delta")
            self._check_positive_definite(self.R_delta, "R_delta")

            self.R_delta_horizon = block_diag(*([self.R_delta] * (self.tau_f - 1)))
        else:
            self.R_delta_horizon = None  # Ensure attribute is always defined

        # Validate R_delta_first
        if self.R_delta_first is not None:
            self._validate_square_matrix(self.R_delta_first, "R_delta_first")
            self._check_positive_definite(self.R_delta_first, "R_delta_first")

    @staticmethod
    def _validate_square_matrix(matrix: np.ndarray, name: str):
        """Checks if a given matrix is square and raises an error if not."""
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array, but got {type(matrix).__name__}.")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix, but got shape {matrix.shape}.")

    @staticmethod
    def _check_positive_definite(matrix: np.ndarray, name: str, semi: bool = False):
        """
        Checks if a given matrix is positive definite or positive semidefinite.
        Raises a ValueError if the condition is not met.

        :param matrix: The matrix to check.
        :param name: Name of the matrix (for error messages).
        :param semi: If True, checks for positive semidefiniteness instead of definiteness.
        """

        eigenvalues = np.linalg.eigvalsh(matrix)  # Compute eigenvalues

        if semi:
            if np.any(eigenvalues < 0):  # Semidefinite check
                raise ValueError(f"{name} must be positive semidefinite, but has eigenvalues: {eigenvalues}")
        else:
            if np.any(eigenvalues <= 0):  # Positive definite check
                raise ValueError(f"{name} must be positive definite, but has eigenvalues: {eigenvalues}")


@dataclass
class DPCDimensions:
    """
    Stores key system dimensions required for Data-Driven Predictive Control (DPC) computations.

    Attributes:
        m (int): Number of control inputs.
        p (int): Number of system outputs.
        mp (int): Sum of control inputs (`m`) and system outputs (`p`).
        n_y_f (int): Dimension of the future output trajectory vector.
        n_u_f (int): Dimension of the future input trajectory vector.
        n_z_p (int): Dimension of the past measurements vector.
    """

    m: int
    p: int
    mp: int
    n_y_f: int
    n_u_f: int
    n_z_p: int

    def __post_init__(self):
        """
        Validates that dimensions are positive integers.

        Raises:
            ValueError: If any dimension is not a positive integer.
        """
        for attr_name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{attr_name} must be a positive integer, but got {value}")
