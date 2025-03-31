"""
Handles LTI system simulation.

This module provides an `LTISimulator` class for simulating discrete-time
Linear Time-Invariant (LTI) systems and includes factory functions for
creating common system models.

Functions:
    - Various factory functions for creating predefined LTI systems.
"""

from dataclasses import dataclass
from typing import Optional

import control as ctrl  # type: ignore
import numpy as np

from do_dpc.control_utils.noise_generators import WhiteNoiseGenerator
from do_dpc.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StateSpaceModel:
    """
    Represents the system matrices for a state-space model.
    Attributes:
            A (np.ndarray): System state matrix (shape `(n, n)`).
            B (np.ndarray): Input matrix (shape `(n, m)`).
            C (np.ndarray): Output matrix (shape `(p, n)`).
            D (np.ndarray): Feedthrough matrix (shape `(p, m)`).
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    def __post_init__(self):
        """
        Validate the dimensions of the system matrices after initialization.

        Raises:
            ValueError: If any of the matrices do not have the expected dimensions.
        """
        n, m = self.B.shape
        p, _n = self.C.shape
        _p, _m = self.D.shape

        if self.A.shape != (n, n):
            raise ValueError(f"Matrix A must have shape ({n}, {n}), but got {self.A.shape}")
        if self.C.shape[1] != n:
            raise ValueError(f"Matrix C must have shape ({p}, {n}), but got {self.C.shape}")
        if self.D.shape != (p, m):
            raise ValueError(f"Matrix D must have shape ({p}, {m}), but got {self.D.shape}")


# pylint: disable=too-many-instance-attributes
class LTISimulator:
    """
    Simulates a Linear Time-Invariant (LTI) system of the form:

    .. math::
        x[k+1] = A * x[k] + B * u[k] + w[k]
        y[k]   = C * x[k] + D * u[k] + v[k]

    where:
        - :math:`x[k]` is the system state at time step `k`,
        - :math:`u[k]` is the input at time step `k`,
        - :math:`w[k]` is the process noise,
        - :math:`y[k]` is the system output,
        - :math:`v[k]` is the measurement noise.

    The system is simulated with state-space matrices :math:`A`, :math:`B`, :math:`C`, and :math:`D`,
    and optionally, process and measurement noise can be added using white noise generators.

    Attributes:
        sys (StateSpaceModel): Matrix representation of the system.
        x_0 (np.ndarray): Initial state vector (shape `(n,)`), representing the state at time step 0.
        process_noise (Optional[WhiteNoiseGenerator]): Noise affecting the system state.
        measurement_noise (Optional[WhiteNoiseGenerator]): Noise affecting the system output.

    Args:
        sys (StateSpaceModel): Matrix representation of the system.
        x_0 (np.ndarray): Initial state vector (shape `(n,)`).
        process_noise (Optional[WhiteNoiseGenerator]): Noise affecting system state.
        measurement_noise (Optional[WhiteNoiseGenerator]): Noise affecting system output.

    Raises:
        ValueError: If x_0 dimensions are inconsistent.
    """

    def __init__(
        self,
        sys: StateSpaceModel,
        x_0: np.ndarray,
        process_noise: Optional[WhiteNoiseGenerator] = None,
        measurement_noise: Optional[WhiteNoiseGenerator] = None,
    ):
        """
        Initializes the LTI system simulator.

        Args:
            sys (StateSpaceModel): Matrix representation of the system.
            x_0 (np.ndarray): Initial state vector (shape `(n,)`).
            process_noise (Optional[WhiteNoiseGenerator]): Noise affecting system state.
            measurement_noise (Optional[WhiteNoiseGenerator]): Noise affecting system output.

        Raises:
            ValueError: If x_0 dimensions are inconsistent.
        """

        if x_0.shape != (sys.B.shape[0],):
            raise ValueError(f"Initial state x_0 must have shape ({sys.B.shape[0]},), but got {x_0.shape}")

        self.sys = sys
        self.A = sys.A
        self.B = sys.B
        self.C = sys.C
        self.D = sys.D
        self.x = np.array(x_0, dtype=float)
        self.x_0 = x_0

        self.process_noise = process_noise or WhiteNoiseGenerator(std=np.zeros_like(x_0))  # Default: no noise
        self.measurement_noise = measurement_noise or WhiteNoiseGenerator(
            std=np.zeros(self.C.shape[0])
        )  # Default: no noise

        logger.info("LTISimulator initialized with process noise and measurement noise.")

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Advances the LTI system by one-step given a control input.

        Args:
            u (np.ndarray): Control input vector (shape `(m,)`).

        Returns:
            np.ndarray: The system output `y` (shape `(p,)`).

        Raises:
            ValueError: If `u` does not match the expected input shape `(m,)`.
        """

        if u.shape != (self.B.shape[1],):
            raise ValueError(f"Control input must have shape ({self.B.shape[1]},), but got {u.shape}")

        # Process noise
        w = self.process_noise.generate()

        # State update
        self.x = self.A @ self.x + self.B @ u + w  # Apply process noise

        # Measurement noise
        v = self.measurement_noise.generate()

        # Compute output with noise
        y = self.C @ self.x + self.D @ u + v  # Apply measurement noise

        logger.debug("Step completed. New state: %s, Output: %s", self.x, y)
        return y

    def set_initial_x_0(self, x_0: np.ndarray):
        """
        Sets a new initial state for the system.

        Args:
            x_0 (np.ndarray): The new initial state vector.

        Raises:
            ValueError: If the shape of x_0 does not match the system state dimension.
        """
        if x_0.shape != self.x.shape:
            raise ValueError(f"Initial state x_0 must have shape {self.x.shape}, but got {x_0.shape}")

        self.x = np.array(x_0, dtype=float)

    def reset_x_to_x_0(self):
        """
        Resets to the initial state for the system.
        """

        self.x = self.x_0

    def get_state(self) -> np.ndarray:
        """
        Returns the current state of the system.

        Returns:
            np.ndarray: Current state vector.
        """
        return self.x

    def get_output(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the current system output with measurement noise.

        Args:
            u (Optional[np.ndarray]): Control input vector.

        Returns:
            np.ndarray: Current system output vector.
        """
        v = self.measurement_noise.generate()
        if self.D is not None and u is not None:
            return self.C @ self.x + self.D @ u + v
        return self.C @ self.x + v

    def get_output_without_noise(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the current output of the system without measurement noise.

        Args:
            u (np.ndarray, optional): The control input vector (shape (m,)).

        Returns:
            np.ndarray: The current output vector (shape (p,)).
        """
        if self.D is not None and u is not None:
            return self.C @ self.x + self.D @ u
        return self.C @ self.x

    def get_dims(self) -> tuple:
        """
        Returns the dimensions of the LTI system.

        Returns:
            tuple:
                - n (int): Number of state variables.
                - m (int): Number of control inputs.
                - p (int): Number of system outputs.
        """
        n, m = self.B.shape
        p, _ = self.C.shape
        return n, m, p

    def calculate_inf_hor_Kalman_gain_K(self) -> np.ndarray:
        """
        Computes the infinite-horizon Kalman Gain (K) for state estimation.

        System model:
            x[n+1] = A x[n] + B u[n] + G w[n]
            y[n]   = C x[n] + D u[n] + v[n]

        Noise covariance properties:
            E{w w^T} = QN  (Process noise covariance)
            E{v v^T} = RN  (Measurement noise covariance)
            E{w v^T} = NN  (Cross-covariance, assumed zero)

        Assumption:
            G = RN.

        Kalman Gain (K) minimizes the state estimation error:
            x_e[n+1] = A x_e[n] + B u[n] + K(y[n] - C x_e[n] - D u[n])

        Returns:
            K (np.ndarray): Kalman Gain matrix.
        """
        # Compute noise covariance matrices
        QN = np.diag(self.process_noise.std**2)
        RN = (self.measurement_noise.std**2) * np.eye(self.C.shape[0])

        if np.all(QN == 0):
            return np.zeros((self.A.shape[0], self.C.shape[0]))

        # Compute Kalman Gain using discrete-time Linear Quadratic Estimator (LQE)
        K, _, _ = ctrl.dlqe(self.A, RN, self.C, QN, RN)

        # Ensure K has the correct shape
        if K.shape != (self.A.shape[0], self.C.shape[0]):
            raise ValueError(
                f"Kalman Gain K has incorrect shape: {K.shape}, expected: {(self.A.shape[0], self.C.shape[0])}"
            )

        return K


# Factory functions
def create_1D_double_integrator(
    meas_noise_std: Optional[np.ndarray] = None,
    process_noise_std: Optional[np.ndarray] = None,
    meas_noise_seed: Optional[int] = None,
    process_noise_seed: Optional[int] = None,
):
    """
    Creates and returns an LTISimulator instance representing a  1-D double integrator system.

    System dynamics:
        A = [[1, 1], [0, 1]]
        B = [[1], [0]]
        C = [[1, 0], [0, 1]]
        D = [[0], [0]]
        x_0 = [0, 0]

    Optional Noise Parameters:
        meas_noise_std (ndarray, optional): Standard deviation of measurement noise.
        process_noise_std (ndarray, optional): Standard deviation for process noise.

    Returns:
        LTISimulator: A configured instance with the specified noise properties.
    """
    # Initialize noise generators if standard deviation is nonzero
    measurement_noise = (
        WhiteNoiseGenerator(std=meas_noise_std, seed=meas_noise_seed) if meas_noise_std is not None else None
    )
    process_noise = (
        WhiteNoiseGenerator(std=process_noise_std, seed=process_noise_seed) if process_noise_std is not None else None
    )

    # Define system matrices
    sys = StateSpaceModel(
        A=np.array([[1, 1], [0, 1]]), B=np.array([[0], [1]]), C=np.array([[1, 0], [0, 1]]), D=np.array([[0], [0]])
    )

    return LTISimulator(sys, np.array([0, 0]), measurement_noise=measurement_noise, process_noise=process_noise)


def create_pre_stabilized_1D_double_integrator(
    meas_noise_std: Optional[np.ndarray] = None,
    process_noise_std: Optional[np.ndarray] = None,
    meas_noise_seed: Optional[int] = None,
    process_noise_seed: Optional[int] = None,
):
    """
    Creates and returns an LTISimulator instance representing a 1-D double integrator system.
    With an additional stabilizing PD controller with Kp = 0.1, Kd = 0.1.

    System dynamics:
        A = [[0.9, 1], [0, 0.9]]
        B = [[1], [0]]
        C = [[1, 0], [0, 1]]
        D = [[0], [0]]
        x_0 = [0, 0]

    Optional Noise Parameters:
        meas_noise_std (ndarray, optional): Standard deviation of measurement noise.
        process_noise_std (ndarray, optional): Standard deviation for process noise.
        meas_noise_seed (int, optional): Seed for the measurement noise.
        process_noise_seed (int, optional): Seed for the process noise.

    Returns:
        LTISimulator: A configured instance with the specified noise properties.
    """
    # Initialize noise generators if standard deviation is nonzero
    measurement_noise = (
        WhiteNoiseGenerator(std=meas_noise_std, seed=meas_noise_seed) if meas_noise_std is not None else None
    )
    process_noise = (
        WhiteNoiseGenerator(std=process_noise_std, seed=process_noise_seed) if process_noise_std is not None else None
    )

    # Define system matrices
    sys = StateSpaceModel(
        A=np.array([[0.9, 1], [0, 0.9]]), B=np.array([[0], [1]]), C=np.array([[1, 0], [0, 1]]), D=np.array([[0], [0]])
    )

    return LTISimulator(sys, np.array([0, 0]), measurement_noise=measurement_noise, process_noise=process_noise)


def create_3D_double_integrator(
    meas_noise_std: Optional[np.ndarray] = None,
    process_noise_std: Optional[np.ndarray] = None,
    meas_noise_seed: Optional[int] = None,
    process_noise_seed: Optional[int] = None,
):
    """
    Creates and returns an LTISimulator instance representing a  3-D double integrator system.

    The 3 dimensions are independent of each other.

    Optional Noise Parameters:
        meas_noise_std (ndarray, optional): Standard deviation of measurement noise.
        process_noise_std (ndarray, optional): Standard deviation for process noise.

    Returns:
        LTISimulator: A configured instance with the specified noise properties.
    """
    # Initialize noise generators if standard deviation is nonzero
    measurement_noise = (
        WhiteNoiseGenerator(std=meas_noise_std, seed=meas_noise_seed) if meas_noise_std is not None else None
    )
    process_noise = (
        WhiteNoiseGenerator(std=process_noise_std, seed=process_noise_seed) if process_noise_std is not None else None
    )

    A1 = np.array([[1, 1], [0, 1]])
    B1 = np.array([[0], [1]])
    C1 = np.eye(2)
    D1 = np.zeros((2, 1))

    # Define system matrices
    sys = StateSpaceModel(
        A=np.kron(np.eye(3), A1), B=np.kron(np.eye(3), B1), C=np.kron(np.eye(3), C1), D=np.kron(np.eye(3), D1)
    )

    return LTISimulator(sys, np.zeros((6,)), measurement_noise=measurement_noise, process_noise=process_noise)


def create_landau_benchmark(
    meas_noise_std: Optional[np.ndarray] = None,
    process_noise_std: Optional[np.ndarray] = None,
    meas_noise_seed: Optional[int] = None,
    process_noise_seed: Optional[int] = None,
):
    """
    Creates and returns an LTISimulator instance representing the Landau Benchmark.

    All the eigenvalues of this system have mags that are close, but less than, 1

    Optional Noise Parameters:
        meas_noise_std (ndarray, optional): Standard deviation of measurement noise.
        process_noise_std (ndarray, optional): Standard deviation for process noise.
        meas_noise_seed (int, optional): Seed for the measurement noise.
        process_noise_seed (int, optional): Seed for the process noise.

    Returns:
        LTISimulator: A configured instance with the specified noise properties.
    """
    # Initialize noise generators if standard deviation is nonzero
    measurement_noise = (
        WhiteNoiseGenerator(std=meas_noise_std, seed=meas_noise_seed) if meas_noise_std is not None else None
    )
    process_noise = (
        WhiteNoiseGenerator(std=process_noise_std, seed=process_noise_seed) if process_noise_std is not None else None
    )

    A = np.array([[1.4183, -1.5894, 1.3161, -0.8864], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    B = np.array([[1, 0, 0, 0]]).T
    C = np.array([[0, 0, 0.2826, 0.5067]])

    # Define system matrices
    sys = StateSpaceModel(A=A, B=B, C=C, D=np.array([[0]]))

    return LTISimulator(sys, np.array([0, 0, 0, 0]), measurement_noise=measurement_noise, process_noise=process_noise)
