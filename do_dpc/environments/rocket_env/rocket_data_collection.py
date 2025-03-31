"""
Module for Data collection for the Rocket lander.

RocketInputGenerator Class Definition.
"""

import sys

import numpy as np
from tqdm import tqdm  # type: ignore

from do_dpc.control_utils.control_structs import InputOutputTrajectory, Bounds
from do_dpc.control_utils.noise_generators import WhiteNoiseGenerator
from do_dpc.control_utils.pid_control_utils import PIDCombo
from do_dpc.control_utils.pid_profiles import ROCKET_PID_COMBO
from do_dpc.control_utils.trajectory_collector import TrajectoryCollector
from do_dpc.environments.rocket_env.rocket_env_facade import RocketEnvFacade
from do_dpc.environments.rocket_env.rocket_utils import is_rocket_outside_reasonable_bounds
from do_dpc.utils.logging_config import get_logger

N_SAMPLES = 800

logger = get_logger(__name__)


class RocketInputGenerator:
    """
    Generates pre-stabilized random inputs for rocket control.

    Attributes:
        pid_combo (PIDCombo): PID controller handling multiple input-output mappings.
        exc_gen (WhiteNoiseGenerator): Gaussian noise generator for random inputs.
        input_limits (Bounds): Constraints on the generated input values.
    """

    def __init__(self, input_limits: Bounds, pid_combo: PIDCombo, exc_gen: WhiteNoiseGenerator):
        """
        Initializes the input generator with PID-based correction and noise generation.

        Attributes:
            pid_combo (PIDCombo): PID controller handling multiple input-output mappings.
            exc_gen (WhiteNoiseGenerator): Gaussian noise generator for random inputs.
            input_limits (Bounds): Constraints on the generated input values.

        Args:
            input_limits (Bounds): Minimum and maximum input limits.
            pid_combo (PIDCombo): Combined PID controllers for error correction.
            exc_gen (WhiteNoiseGenerator): Gaussian noise generator for random inputs.
        """
        self.pid_combo = pid_combo
        self.exc_gen = exc_gen
        self.input_limits = input_limits

    def compute_action(self, output) -> np.ndarray:
        """
        Computes the appropriate action based on the PID controller or noise generator.

        If the rocket is within reasonable bounds, PID-based control is applied.
        Otherwise, a random noise-based action is generated.

        Args:
            output (np.ndarray): Current system output to evaluate.

        Returns:
            np.ndarray: Clipped action values within allowed limits.
        """
        if is_rocket_outside_reasonable_bounds(output):
            err, der_err = self.pid_combo.converter_function(output, None)
            action = -self.pid_combo.MIMO_PID.compute_with_derivative(err, der_err)
            return self.clip_action(action)

        return self.clip_action(self.exc_gen.generate())

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clips the action values to stay within the defined input limits.

        Args:
            action (np.ndarray): Computed action values.

        Returns:
            np.ndarray: Clipped action values.
        """
        return np.maximum(np.minimum(action, self.input_limits.max_values), self.input_limits.min_values)


def collect_trajectory_data_env(
    env: RocketEnvFacade, m: int, p: int, n_samples: int = N_SAMPLES
) -> InputOutputTrajectory:
    """
    Collects trajectory data from the given environment.

    Args:
        env: The environment object with `get_output()` and `done` attributes.
        m (int): Number of system inputs.
        p (int): Number of system outputs.
        n_samples (int, optional): Number of samples to collect. Defaults to `N_SAMPLES`.

    Returns:
        InputOutputTrajectory: The collected trajectory data, or None if data collection fails.

    Raises:
        AttributeError: If `env` does not have the required attributes or methods.
        ValueError: If `n_samples` is not a positive integer.
        RuntimeError: If the environment is already done before collecting data.
    """
    sys.stdout.flush()

    if not hasattr(env, "get_output") or not hasattr(env, "done"):
        raise AttributeError("The provided environment must have 'get_output()' and 'done' attributes.")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    if env.done:
        raise RuntimeError("The environment is already done. Cannot collect trajectory data.")

    traj_col = TrajectoryCollector(m, p, n_samples)
    y_next = env.get_output()

    obs_init = y_next[:2].copy()

    exc_gen = WhiteNoiseGenerator(np.array([0.33, 0, 0]), np.array([0.5, 1, 1]), seed=1)

    stabilized_random_inputs = RocketInputGenerator(env.get_input_bounds(), ROCKET_PID_COMBO, exc_gen)

    for i in tqdm(range(n_samples), desc="Collecting Training Data", ncols=80):
        if env.done:
            if env.done:
                logger.error("Data collection stopped prematurely: Environment reached 'done' output.")
                logger.info("Iteration: %d", i)
                env.close()
                raise RuntimeError("Could not complete the data collection as the environment is done.")

        output = y_next.copy()
        output[:2] -= obs_init[:2]

        u_next = stabilized_random_inputs.compute_action(output)

        traj_col.store_measurements(y_next, u_next)

        y_next = env.step(u_next)

    logger.info("Data Collection complete")
    logger.info("Total Samples: %d", n_samples)
    return traj_col.get_trajectory_data()
