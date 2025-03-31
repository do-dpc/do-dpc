"""
This module implements the MPCOracle class for benchmarking purposes.

The MPCOracle class does not estimate a model using the collected trajectory data.
Instead, it uses the actual system matrices A, B, C, D and the optimal infinite Kalman Gain K.
Hence, it is named "oracle" as it has perfect knowledge of the system.
"""

import numpy as np

from do_dpc.control_utils.control_structs import InputOutputTrajectory
from do_dpc.dpc.dpc import DPC
from do_dpc.dpc.dpc_structs import DPCParameters
from do_dpc.dpc.mpc import MPC, MPCSystemMatrices


@DPC.register("MPCOracle")
class MPCOracle(MPC):
    """
    MPCOracle class for benchmarking purposes.

    This class uses the actual system matrices and the optimal infinite horizon Kalman Gain K.
    It does not estimate a model from the collected trajectory data.

    Note:
        This controller is useful for benchmarking the performance of other data-driven controllers by
        providing the ideal scenario where the system matrices and Kalman gain are known.

    Attributes:
        sys_data (MPCSystemMatrices): Contains the actual system matrices and optimal Kalman Gain K.
        n_state (int): The number of states in the system.
        dpc_params (DPCParameters): Controller configuration parameters passed from the parent class.

    Args:
        dpc_params (DPCParameters): Controller configuration parameters.
        n_state (int): Number of states in the system.
        sys_data (MPCSystemMatrices): Actual system matrices and Kalman Gain.

    Raises:
        ValueError: If `n_state` does not match the number of states in `sys_data`.
    """

    def __init__(
        self,
        dpc_params: DPCParameters,
        n_state: int,
        sys_data: MPCSystemMatrices,
    ):
        """
        Initializes the MPCOracle controller.

        Args:
            n_state (int): Number of states in the system.
            sys_data (MPCSystemMatrices): Actual system matrices and Kalman Gain.

        Raises:
            ValueError: If `n_state` does not match the number of states in `sys_data`.
        """
        if n_state != sys_data.sys.A.shape[0]:
            raise ValueError(
                f"n_state must match the number of states in sys_data, expected: {sys_data.sys.A.shape[0]}"
            )

        self.sys_data = sys_data
        super().__init__(dpc_params, self._create_empty_training_data(100), n_state=n_state)

    def calculate_system_data(self) -> MPCSystemMatrices:
        """
        Returns the actual system data for the MPCOracle.

        Returns:
            MPCSystemMatrices: An instance containing the system matrices and Kalman Gain.
        """
        return self.sys_data

    def _create_empty_training_data(self, num_samples: int) -> InputOutputTrajectory:
        """
        Creates an empty training dataset with zeroed input and output arrays.

        Parameters:
            num_samples (int): The number of samples for the training data.
                              Must be a positive integer.

        Returns:
            InputOutputTrajectory: A data structure containing zero-initialized
                                   input (u) and output (y) arrays.

        Raises:
            ValueError: If num_samples is not a positive integer.
        """
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        p, _ = self.sys_data.sys.C.shape
        _, m = self.sys_data.sys.B.shape

        y = np.zeros((p, num_samples))
        u = np.zeros((m, num_samples))

        return InputOutputTrajectory(y=y, u=u)
