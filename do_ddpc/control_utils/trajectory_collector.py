"""
Handles control input generation and trajectory data collection for data-driven control.
"""

from typing import Optional

import numpy as np
from tqdm import tqdm  # type: ignore

from do_ddpc.control_utils.control_structs import InputOutputTrajectory
from do_ddpc.control_utils.lti_systems import LTISimulator
from do_ddpc.control_utils.noise_generators import WhiteNoiseGenerator
from do_ddpc.control_utils.pid_control_utils import PIDCombo
from do_ddpc.utils.logging_config import get_logger

logger = get_logger(__name__)

N_SAMPLES = 1000
EXC_DEV = 1


class TrajectoryCollector:
    """
    Collects system output (`y`) and control input (`u`) trajectories.

    This class handles the logging and structured collection of trajectory data for
    data-driven control applications.
    """

    def __init__(self, m: int, p: int, traj_length: int):
        """
        Initializes the TrajectoryCollector.

        Args:
            m (int): Number of control inputs (dimension of `u`).
            p (int): Number of system outputs (dimension of `y`).
            traj_length (int): Length of the trajectory (number of time steps).
        """
        self.m = m
        self.p = p
        self.traj_length = traj_length

        # Initialize trajectory arrays with NaNs to track missing values.
        self.y = np.full((p, traj_length), np.nan)
        self.u = np.full((m, traj_length), np.nan)

        logger.info("TrajectoryCollector initialized with m=%s, p=%s, traj_length=%s", m, p, traj_length)

    def store_measurements(self, y_next: np.ndarray, u_next: np.ndarray):
        """
        Stores the next system output (`y_next`) and the corresponding control input (`u_next`)
        in the trajectory dataset.

        This function assigns the provided system output `y_next` and control input `u_next`
        to the first available (empty) column in the trajectory storage. If the trajectory
        is already full, a warning is logged.

        Args:
            y_next (np.ndarray): The new system output observation (shape `(p,)`).
            u_next (np.ndarray): The control input applied at this step (shape `(m,)`).

        Raises:
            ValueError: If `y_next` does not match the expected shape `(p,)`.
            ValueError: If `u_next` does not match the expected shape `(m,)`.
        """
        if y_next.shape != (self.p,):
            raise ValueError(f"y_next must have shape ({self.p},) but got {y_next.shape}")

        if u_next.shape != (self.m,):
            raise ValueError(f"u_next must have shape ({self.m},) but got {u_next.shape}")

        u_idx = np.argmax(np.isnan(self.u[0]))
        if np.isnan(self.u[0, u_idx]):
            self.u[:, u_idx] = u_next
            logger.debug("Stored control input u at column index %s", u_idx)
        else:
            logger.warning("Trajectory collection for u is already complete.")

        y_idx = np.argmax(np.isnan(self.y[0]))
        if np.isnan(self.y[0, y_idx]):
            self.y[:, y_idx] = y_next
            logger.debug("Stored system output y at column index %s", y_idx)
        else:
            logger.warning("Trajectory collection for y is already complete.")

    def get_trajectory_data(self) -> InputOutputTrajectory:
        """
        Retrieves the collected trajectory data.

        Ensures that data collection is complete before returning the trajectory.
        If NaN values are present, logs a warning, removes them, and proceeds with returning the cleaned data.

        Returns:
            InputOutputTrajectory: The collected trajectory data with `y`, `u`, `m`, and `p`, with NaN values removed.
        """
        if np.isnan(self.u).any() or np.isnan(self.y).any():
            logger.warning("NaN values detected in trajectory data. Cleaning data by removing NaNs.")
            self.y = np.nan_to_num(self.y)
            self.u = np.nan_to_num(self.u)

        logger.info("Trajectory collection complete. Returning cleaned trajectory data.")

        return InputOutputTrajectory(y=self.y, u=self.u)


# pylint: disable=R0913,R0917
def collect_trajectory_data(
    sys: LTISimulator,
    m: int,
    p: int,
    pid_combo: Optional[PIDCombo] = None,
    n_samples: int = N_SAMPLES,
    exc_dev: float = EXC_DEV,
) -> InputOutputTrajectory:
    """
    Collects trajectory data from an LTI system with optional PID control.

    Args:
        sys (LTISimulator): The system to collect data from.
        m (int): Number of control inputs.
        p (int): Number of system outputs.
        pid_combo (PIDCombo, optional): PID controller with function to get state to error.
        n_samples (int, optional): Number of samples for the data collection.
        exc_dev (float, optional): Excitation Deviation for the input generation.

    Returns:
        InputOutputTrajectory: Collected trajectory data containing system outputs and control inputs.
    """
    seed = 7473
    u_mean = np.zeros((m,))
    traj_col = TrajectoryCollector(m, p, n_samples)
    ctrl_gen = WhiteNoiseGenerator(mean=u_mean, std=exc_dev * np.ones_like(u_mean), seed=seed)
    y_next = sys.get_output()

    for _ in tqdm(range(n_samples), desc="Collecting Training Data", ncols=80, position=0):
        u_next = ctrl_gen.generate()
        if pid_combo:
            err, der_err = pid_combo.converter_function(y_next, None)
            u_next -= pid_combo.MIMO_PID.compute_with_derivative(err, der_err)
        traj_col.store_measurements(y_next, u_next)
        y_next = sys.step(u_next)

    return traj_col.get_trajectory_data()
