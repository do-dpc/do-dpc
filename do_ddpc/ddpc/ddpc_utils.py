"""
Collection of utils, only static methods for the class DPCUtils
"""

import logging

import numpy as np

from do_ddpc.ddpc.ddpc_structs import DPCDimensions, DPCParameters, DPCClosedFormSolutionMatrices
from do_ddpc.control_utils.control_structs import InputOutputTrajectory

logger = logging.getLogger(__name__)


class StaticOnlyMeta(type):
    """
    Metaclass ensuring all methods in a class are static methods.

    Raises a TypeError if any method is not a static method.

    Example:
        >>> class MyClass(metaclass=StaticOnlyMeta):
        ...     @staticmethod
        ...     def my_static_method():
        ...         pass

        >>> class InvalidClass(metaclass=StaticOnlyMeta):
        ...     def my_instance_method(self):
        ...         pass
        ...
        Traceback (most recent call last):
            ...
        TypeError: Method my_instance_method must be a static method
    """

    def __new__(mcs, name, bases, class_dict):
        """
        Creates a new class ensuring all methods are static.

        Args:
            mcs (type): The metaclass.
            name (str): The name of the new class.
            bases (tuple): The base classes of the new class.
            class_dict (dict): The class dictionary.

        Raises:
            TypeError: If any method is not a static method.

        Returns:
            type: The newly created class.
        """
        for attr_name, attr_value in class_dict.items():
            if callable(attr_value) and not isinstance(attr_value, staticmethod):
                raise TypeError(f"Method {attr_name} must be a static method")
        return super().__new__(mcs, name, bases, class_dict)


class DPCUtils(metaclass=StaticOnlyMeta):
    """
    Class for utility function that are used by DPC or DPC inherited classes.
    """

    @staticmethod
    def check_valid_trajectory_data(training_data: InputOutputTrajectory) -> tuple[int, int, int]:
        """
        Validates the trajectory data for consistency.

        Args:
            training_data: The trajectory data object containing `y` (outputs) and `u` (inputs).

        Returns:
            tuple: (m, p, num_samples), where
                - m (int): Number of control inputs.
                - p (int): Number of system outputs.
                - num_samples (int): Number of collected samples.

        Raises:
            ValueError: If `y` and `u` have different numbers of samples.
            TypeError: If `y` or `u` are not NumPy arrays.
            ValueError: If `y` or `u` are empty.
        """
        if not isinstance(training_data, InputOutputTrajectory):
            raise TypeError("training_data must be an instance of TrajectoryData.")

        if not isinstance(training_data.y, np.ndarray) or not isinstance(training_data.u, np.ndarray):
            raise TypeError("Trajectory data `y` and `u` must be NumPy arrays.")

        if training_data.y.size == 0 or training_data.u.size == 0:
            raise ValueError("Trajectory data `y` and `u` cannot be empty.")

        p, num_samples_y = training_data.y.shape
        m, num_samples_u = training_data.u.shape

        if num_samples_y != num_samples_u:
            raise ValueError(
                f"Mismatch in number of samples: y has {num_samples_y}, but u has {num_samples_u}."
                f" They must be the same."
            )

        return m, p, num_samples_y

    @staticmethod
    def check_valid_controller_parameters(ddpc_params: DPCParameters, m: int, p: int):
        """
        Validates the dimensions of controller parameters Q and R.

        Args:
            ddpc_params (DPCParameters): Struct for the control parameters.
            m (int): Number of control inputs.
            p (int): Number of system outputs.

        Raises:
            ValueError: If Q is not a p × p matrix.
            ValueError: If R is not an m × m matrix.
            ValueError: If R_delta is not None or an m × m matrix.
            ValueError: If R_delta_first is not None or an m × m matrix.
        """
        if not isinstance(ddpc_params, DPCParameters):
            raise TypeError("ddpc_params must be an instance of ControllerParameters.")

        # Validate R matrix
        if not isinstance(ddpc_params.R, np.ndarray):
            raise TypeError("R must be a numpy array.")

        m_1, m_2 = ddpc_params.R.shape
        if m_1 != m_2 or m_1 != m:
            raise ValueError(f"Invalid dimension for R: Expected {m}×{m}, but got {m_1}×{m_2}.")

        # Validate Q matrix
        if not isinstance(ddpc_params.Q, np.ndarray):
            raise TypeError("Q must be a numpy array.")

        p_1, p_2 = ddpc_params.Q.shape
        if p_1 != p_2 or p_1 != p:
            raise ValueError(f"Invalid dimension for Q: Expected {p}×{p}, but got {p_1}×{p_2}.")

        if ddpc_params.R_delta is not None:
            if not isinstance(ddpc_params.R_delta, np.ndarray):
                raise TypeError("R delta must be a numpy array.")

            m_1, m_2 = ddpc_params.R_delta.shape
            if m_1 != m_2 or m_1 != m:
                raise ValueError(f"Invalid dimension for R delta: Expected {m}×{m}, but got {m_1}×{m_2}.")

        if ddpc_params.R_delta_first is not None:
            if not isinstance(ddpc_params.R_delta_first, np.ndarray):
                raise TypeError("R delta must be a numpy array.")

            m_1, m_2 = ddpc_params.R_delta_first.shape
            if m_1 != m_2 or m_1 != m:
                raise ValueError(f"Invalid dimension for R delta: Expected {m}×{m}, but got {m_1}×{m_2}.")

    @staticmethod
    def calculate_dimensions(ddpc_params: DPCParameters, m: int, p: int) -> DPCDimensions:
        """
        Computes the required dimensions for control execution.

        Returns:
            TPCDimensions: Struct containing the computed dimension values.
        """
        return DPCDimensions(
            m=m,
            p=p,
            mp=m + p,
            n_y_f=p * ddpc_params.tau_f,
            n_u_f=m * ddpc_params.tau_f,
            n_z_p=(m + p) * ddpc_params.tau_p,
        )

    @staticmethod
    def construct_difference_matrix(meas_dims: int, horizon: int) -> np.ndarray:
        """
        Constructs the difference operator matrix D of size (meas_dims * (horizon - 1), meas_dims * )

         The structure of D:

            [ 1  0 -1  0  ...  0  0  0  0 ]
            [ 0  1  0 -1  ...  0  0  0  0 ]
            [ .  .  .  .  ...  .  .  .  . ]
            [ 0  0  0  0  ...  1  0 -1  0 ]
            [ 0  0  0  0  ...  0  1  0 -1 ]

        This matrix applies finite differences between consecutive time steps.

        Return:
            Difference matrix D
        """

        # check if meas_dims, horizon are integer and larger or equal to 1.
        m = meas_dims

        D = np.zeros((m * (horizon - 1), m * horizon))

        for i in range(horizon - 1):
            D[m * i : m * (i + 1), m * i : m * (i + 1)] = np.eye(m)
            D[m * i : m * (i + 1), m * (i + 1) : m * (i + 2)] = -np.eye(m)

        return D

    @staticmethod
    def check_valid_closed_form_gains(dims: DPCDimensions, gains: DPCClosedFormSolutionMatrices):
        """
        Validates the dimensions of the closed-form gain matrices.

        Args:
            dims (DPCDimensions): Dimensions of the DPC
            gains (DPCClosedFormSolutionMatrices): The computed closed-form gain matrices.

        Raises:
            ValueError: If any of the gain matrices have incorrect dimensions.
        """

        expected_shapes = {
            "K_z_p": (dims.n_u_f, dims.n_z_p),
            "K_y_r": (dims.n_u_f, dims.n_y_f),
            "K_u_r": (dims.n_u_f, dims.n_u_f),
        }

        for name, expected_shape in expected_shapes.items():
            matrix = getattr(gains, name, None)
            if matrix is None or matrix.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, but got {matrix.shape if matrix is not None else None}."
                )

    @staticmethod
    def is_positive_semidefinite(matrix: np.ndarray, tolerance: float = 0) -> bool:
        """
        Check if a matrix is positive semidefinite.
        """
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues >= -abs(tolerance)))

    @staticmethod
    def is_positive_definite(matrix: np.ndarray) -> bool:
        """
        Check if a matrix is positive definite.
        """
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > 0))

    @staticmethod
    def save_lq_decomposition(matrix: np.ndarray) -> np.ndarray:
        """
        Computes the LQ decomposition of a given matrix.

        This method first attempts to compute the Cholesky decomposition of
        `matrix @ matrix.T`. If the matrix is not symmetric positive definite,
        it falls back to using the QR decomposition.

        Args:
            matrix (np.ndarray): The input matrix to be decomposed.

        Returns:
            np.ndarray: The lower triangular matrix `L` from the LQ decomposition.

        Raises:
            np.linalg.LinAlgError: If both Cholesky and QR decompositions fail.
        """
        try:
            L = np.linalg.cholesky(matrix @ matrix.T)
        except np.linalg.LinAlgError:
            logger.info("Z is not symmetric positive definite, using QR")
            _, L = np.linalg.qr(matrix.T)
            L = L.T
            if L.shape[0] > L.shape[1]:
                L = np.hstack((L, np.zeros((L.shape[0], L.shape[0] - L.shape[1]))))

        return np.real_if_close(L, tol=1e-3)
