"""
Numerical Subspace State Space System Identification (N4SID or NFourSID) implementation

This module implements N4SID based on the methodology presented in:
- `N4SID <https://www.sciencedirect.com/science/article/pii/0005109894902305>`

The code is extended from: https://github.com/spmvg/nfoursid

N4SID is a subclass of MPC and therefor a subclass of DPC.
"""

import logging
from dataclasses import dataclass

import control as ctrl  # type: ignore
import numpy as np
from numpy.linalg import LinAlgError

from do_dpc.control_utils.lti_systems import StateSpaceModel
from do_dpc.dpc.dpc_structs import DPCParameters
from do_dpc.control_utils.control_structs import InputOutputTrajectory
from do_dpc.dpc.dpc_utils import DPCUtils, StaticOnlyMeta
from do_dpc.dpc.mpc import MPC, MPCSystemMatrices

logger = logging.getLogger(__name__)


@dataclass
class Decomposition:
    """
    Eigenvalue decomposition of a matrix ``matrix`` such that ``left_orthogonal @ eigenvalues @ right_orthogonal``
    equals ``matrix``.

    Attributes:
        left_orthogonal (np.ndarray): The left orthogonal matrix.
        eigenvalues (np.ndarray): The diagonal matrix of eigenvalues.
        right_orthogonal (np.ndarray): The right orthogonal matrix.
    """

    left_orthogonal: np.ndarray
    eigenvalues: np.ndarray
    right_orthogonal: np.ndarray

    def __iter__(self):
        """Allow unpacking."""
        return iter((self.left_orthogonal, self.eigenvalues, self.right_orthogonal))

    def __getitem__(self, index):
        """Allow indexing."""
        return (self.left_orthogonal, self.eigenvalues, self.right_orthogonal)[index]


@dataclass
class SubspaceMatrices:
    """
    Represents needed matrices defined in the paper:
    [PO-MOESP Subspace Identification of Directed Acyclic Graphs with Unknown Topology]
    (https://www.sciencedirect.com/science/article/pii/S0005109814005998).

    Attributes:
        R_22 (np.ndarray)
        R_32 (np.ndarray)
    """

    R_22: np.ndarray
    R_32: np.ndarray

    def __iter__(self):
        """Allow unpacking."""
        return iter((self.R_22, self.R_32))

    def __getitem__(self, index):
        """Allow indexing."""
        return (self.R_22, self.R_32)[index]


class MPCNFourSID(MPC):  # pylint: disable=too-few-public-methods
    """
    Implements the MPC controller using the N4SID
    (Numerical Subspace State Space System Identification) algorithm for system identification.
    The optimal infinite horizon Kalman filter gain is estimated using the data.

     Attributes:
        traj_manager (NFourSIDTrajectoryManager): Manages trajectory data and N4SID calculations.
        n_state (int): The total number of states based on the trajectory data and block rows.
        num_block_rows (int): Number of block rows used in Hankel matrices for system identification.

    Args:
        n_block_rows (int): Number of block rows for Hankel matrices.

    Raises:
        ValueError: If n_block_rows is not an integer or is less than 1.
    """

    def __init__(self, dpc_params: DPCParameters, training_data: InputOutputTrajectory, n_block_rows: int):
        """
        Initializes the MPCNFourSID instance.

        Args:
            dpc_params (DPCParameters): Controller parameters.
            training_data (InputOutputTrajectory): Trajectory data.
            n_block_rows (int): Number of block rows for Hankel matrices.

        Raises:
            ValueError: If n_block_rows is not an integer or is less than 1.
        """

        if not isinstance(n_block_rows, int) or n_block_rows < 1:
            raise ValueError("n_block_rows must be an integer greater than or equal to 1")

        self.traj_manager = NFourSIDTrajectoryManager(training_data, n_block_rows)

        self.n_state = self.traj_manager.p * n_block_rows
        self.num_block_rows = n_block_rows
        super().__init__(dpc_params, training_data, n_state=self.n_state)

    def calculate_system_data(self) -> MPCSystemMatrices:
        """
        Calculates the system data using subspace identification and state space identification.

        Returns:
            MPCSystemMatrices: Identified system data.
        """
        subspace_matrices = self._subspace_identification()
        obs_decomp = self._get_observability_matrix_decomposition(subspace_matrices)

        return self._identify_state_space(obs_decomp)

    def _subspace_identification(self) -> SubspaceMatrices:
        """
        Perform subspace identification based on the PO-MOESP method.
        The instrumental variable contains past outputs and past inputs.
        The implementation uses a QR-decomposition for numerical efficiency and is based on page 329 of [1].

        [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
        Cambridge university press, 2007.

        Returns:
            SubspaceMatrices: Intermediate step for the system identification
        """
        _, r = map(lambda matrix: matrix.T, np.linalg.qr(self.traj_manager.get_uy_future_past_hankel(), mode="reduced"))

        y_rows, u_rows = self.traj_manager.get_y_rows_u_rows()

        return SubspaceMatrices(R_22=r[u_rows:-y_rows, u_rows:-y_rows], R_32=r[-y_rows:, u_rows:-y_rows])

    def _identify_state_space(self, obs_decomp: Decomposition) -> MPCSystemMatrices:
        """
        Identifies the state space model from the observability decomposition.
        Utilizes the covariance matrix to calculate the infinite horizon Kalman filter gain `K`

        Args:
            obs_decomp (Decomposition): Observability matrix decomposition.

        Returns:
            MPCSystemMatrices: Identified state space model data.
        """
        x_est = (np.power(obs_decomp.eigenvalues, 0.5) @ obs_decomp.right_orthogonal)[:, :-1]

        fut_y, fut_u = self.traj_manager.get_future_y_u()
        combined_x_y = np.concatenate([x_est[:, 1:], fut_y[:, :-1]])
        combined_x_u = np.concatenate([x_est[:, :-1], fut_u[:, :-1]])

        ABCD = (np.linalg.pinv(combined_x_u @ combined_x_u.T) @ combined_x_u @ combined_x_y.T).T

        residuals = combined_x_y - ABCD @ combined_x_u

        sys = self._unpack_abcd_matrices(ABCD)
        K = self._calculate_infinite_horizon_kalman_gain_K(residuals, sys.A, sys.C)

        return MPCSystemMatrices(K=K, sys=sys)

    def _unpack_abcd_matrices(self, ABCD: np.ndarray) -> StateSpaceModel:
        """
        Unpacks the ABCD matrices from a given combined matrix and returns a StateSpaceModel.

        Args:
            ABCD (np.ndarray): The combined matrix containing A, B, C, and D matrices.

        Returns:
            StateSpaceModel: The state-space model with separate A, B, C, and D matrices.
        """
        return StateSpaceModel(
            A=ABCD[: self.n_state, : self.n_state],
            B=ABCD[: self.n_state, self.n_state :],
            C=ABCD[self.n_state :, : self.n_state],
            D=ABCD[self.n_state :, self.n_state :],
        )

    def _calculate_infinite_horizon_kalman_gain_K(
        self, residuals: np.ndarray, A: np.ndarray, C: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the infinite horizon Kalman gain matrix K using the control library.

        Note:
            The cross covariance is not yet implemented.
            Covariance matrix P = [R S^T; S Q]
            Q is the process noise covariance.
            R is the measurement noise covariance.
            S is the cross covariance.

        Args:
            residuals (np.ndarray): Residuals from the state-space identification.
            A (np.ndarray): State transition matrix.
            C (np.ndarray): Output matrix.

        Raises:
            Warning: If the Kalman gain could not be calculated. A zero matrix is returned as the Kalman gain.

        Returns:
            np.ndarray: The infinite horizon Kalman gain matrix K.
        """
        P = residuals @ residuals.T / residuals.shape[1]
        Q = P[: self.n_state, : self.n_state]
        R = P[self.n_state :, self.n_state :]
        # S = P[: self.x_dim, self.x_dim:]

        try:
            K, _, _ = ctrl.dlqe(A, Q, C, Q, R)
        except LinAlgError as e:
            logger.warning(
                "LinAlgError encountered in dlqe computation: %s. "
                "This might be due to the absence of noise in the system. "
                "Returning a zero matrix for K.",
                e,
            )
            K = np.zeros((A.shape[0], C.shape[0]))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Unexpected error in dlqe computation: %s", e)
            K = np.zeros((A.shape[0], C.shape[0]))

        return K

    def _get_observability_matrix_decomposition(self, subspace_matrices: SubspaceMatrices) -> Decomposition:
        """
        Calculates the eigenvalue decomposition of the observability matrix.

        Returns:
            Decomposition: The decomposition of the observability matrix.
        """
        R_22, R_32 = subspace_matrices

        observability = R_32 @ np.linalg.pinv(R_22) @ self.traj_manager.get_uy_hankel()
        observ_decomp = NFourSIDUtils.reduce_decomposition(
            NFourSIDUtils.eigenvalue_decomposition(observability), self.n_state
        )
        return observ_decomp


class NFourSIDTrajectoryManager:
    """
    Manages trajectory data for N4SID subspace identification.

    Attributes:
        n_block_rows (int): Number of block rows for Hankel matrices.
        m (int): Number of inputs.
        p (int): Number of outputs.
        y (np.ndarray): Output data.
        u (np.ndarray): Input data.
        u_hankel (np.ndarray): Hankel matrix of input data.
        y_hankel (np.ndarray): Hankel matrix of output data.
    """

    def __init__(self, training_data: InputOutputTrajectory, n_block_rows: int):
        """
        Initializes the trajectory manager with trajectory data and block rows.

        Args:
            training_data (InputOutputTrajectory): Trajectory data.
            n_block_rows (int): Number of block rows for Hankel matrices.
        """

        self.n_block_rows = n_block_rows
        self.m, self.p, _ = DPCUtils.check_valid_trajectory_data(training_data)
        self.y, self.u = training_data.y.T, training_data.u.T
        self.u_hankel = NFourSIDUtils.block_hankel_matrix(self.u, n_block_rows)
        self.y_hankel = NFourSIDUtils.block_hankel_matrix(self.y, n_block_rows)

    def get_uy_hankel(self) -> np.ndarray:
        """
        Combines the input and output Hankel matrices.

        Returns:
            np.ndarray: Combined Hankel matrix.
        """
        return np.concatenate([self.u_hankel, self.y_hankel])

    def get_uy_future_past_hankel(self) -> np.ndarray:
        """
        Constructs the future-past Hankel matrix for subspace identification.

        Returns:
            np.ndarray: Future-past Hankel matrix.
        """
        u_past, u_future = self.u_hankel[:, : -self.n_block_rows], self.u_hankel[:, self.n_block_rows :]
        y_past, y_future = self.y_hankel[:, : -self.n_block_rows], self.y_hankel[:, self.n_block_rows :]
        return np.concatenate([u_future, u_past, y_past, y_future]).T

    def get_y_rows_u_rows(self):
        """
        Gets the number of rows for outputs and inputs in the Hankel matrix.

        Returns:
            tuple: Number of rows for outputs and inputs.
        """
        return self.p * self.n_block_rows, self.m * self.n_block_rows

    def get_future_y_u(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the future part of the trajectory data after a certain number of block rows.

        Returns:
            tuple: A tuple containing the future outputs and future inputs.
                - future_outputs (np.ndarray): The future outputs after the specified block rows.
                - future_inputs (np.ndarray): The future inputs after the specified block rows.
        """
        return self.y[self.n_block_rows :, :].T, self.u[self.n_block_rows :, :].T


class NFourSIDUtils(metaclass=StaticOnlyMeta):
    """
    Utility functions for N4SID subspace identification.
    """

    @staticmethod
    def eigenvalue_decomposition(matrix: np.ndarray) -> Decomposition:
        """
        Calculates the eigenvalue decomposition of a matrix.

        Args:
            matrix (np.ndarray): The matrix to decompose.

        Returns:
            Decomposition: The decomposition of the matrix.
        """
        u, eigenvalues, vh = np.linalg.svd(matrix)
        eigenvalues_mat = np.zeros((u.shape[0], vh.shape[0]))
        np.fill_diagonal(eigenvalues_mat, eigenvalues)
        return Decomposition(u, eigenvalues_mat, vh)

    @staticmethod
    def reduce_decomposition(decomposition: Decomposition, rank: int) -> Decomposition:
        """
        Reduces an eigenvalue decomposition to retain only the largest eigenvalues.

        Args:
            decomposition (Decomposition): The original decomposition.
            rank (int): The number of eigenvalues to retain.

        Returns:
            Decomposition: The reduced decomposition.
        """
        u, s, vh = decomposition
        return Decomposition(u[:, :rank], s[:rank, :rank], vh[:rank, :])

    @staticmethod
    def block_hankel_matrix(matrix: np.ndarray, num_block_rows: int) -> np.ndarray:
        """
        Constructs a block Hankel matrix from the input matrix.

        Args:
            matrix (np.ndarray): Input matrix.
            num_block_rows (int): Number of block rows.

        Returns:
            np.ndarray: Block Hankel matrix.
        """
        hankel_rows_dim = num_block_rows * matrix.shape[1]
        hankel_cols_dim = matrix.shape[0] - num_block_rows + 1

        hankel = np.zeros((hankel_rows_dim, hankel_cols_dim))
        for block_row_index in range(hankel_cols_dim):
            flattened_block_rows = matrix[block_row_index : block_row_index + num_block_rows, :].flatten()
            hankel[:, block_row_index] = flattened_block_rows
        return hankel
