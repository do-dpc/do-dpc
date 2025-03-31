"""
Facade for the Rocket Environment.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from do_ddpc.control_utils.control_structs import Bounds
from do_ddpc.control_utils.lti_systems import StateSpaceModel
from do_ddpc.environments.legacy_coco_rocket.system_model import SystemModel
from do_ddpc.environments.rocket_env.rocket_env_cfg_facade import RocketEnvCfgFacade
from do_ddpc.environments.rocket_env.rocket_utils import calculate_normalized_thrust_to_hover
from do_ddpc.utils.path_manager import get_path_manager

GYM_ID = "do_ddpc_env/RocketLander-v0"

# Get a singleton instance of PathManager
path_manager = get_path_manager()


@dataclass
class RocketEnvironmentArguments:
    """
    Defines the initial conditions and environment settings for a rocket simulation.

    Attributes:
        initial_position (Optional[Tuple[float, float, float]]):
            The initial position of the rocket in 3D space (x, y, z).
            If not provided, defaults to (0.5, 0.4, 0).
        initial_state (Optional[Tuple[float, float, float, float, float, float]]):
            The initial state of the rocket, including position, velocity, and orientation.
        enable_wind (bool):
            If True, wind effects are included in the simulation. Default is False.
        enable_moving_barge (bool):
            If True, the landing barge moves during the simulation. Default is False.
    """

    initial_position: Optional[Tuple[float, float, float]] = None
    initial_state: Optional[Tuple[float, float, float, float, float, float]] = None
    enable_wind: bool = False
    enable_moving_barge: bool = False

    def __post_init__(self):
        """
        Validates input arguments and sets default values if necessary.

        Ensures that either `initial_position` or `initial_state` is provided.
        If both are None, `initial_position` is set to a default value.

        Raises:
            ValueError: If `initial_position` does not have exactly 3 elements.
            ValueError: If `initial_state` does not have exactly 6 elements.
        """

        if self.initial_position is None and self.initial_state is None:
            self.initial_position = (0.5, 0.4, 0)

        if self.initial_position is not None and len(self.initial_position) != 3:
            raise ValueError(f"initial_position must be a tuple of length 3, but got {self.initial_position}")

        if self.initial_state is not None and len(self.initial_state) != 6:
            raise ValueError(f"initial_state must be a tuple of length 6, but got {self.initial_state}")

    def as_dict(self) -> dict:
        """
        Converts the class instance into a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        return asdict(self)


class RocketEnvFacade:
    """
    A high-level interface for interacting with the rocket environment.

    This class manages the environment setup, resets, simulation steps, and provides
    relevant system outputs, including references and bounds.

    Attributes:
        env (gym.Env): The rocket environment instance.
        lin_sys (StateSpaceModel): Linearized system model of the rocket.
        landed_successfully (bool): Tracks whether the rocket has landed successfully.
        done (bool): Indicates if the episode has ended.
        use_virtual_actuators (bool): Whether virtual actuators are used.
        y (np.ndarray): The current system output.

    Args:
        env_args (RocketEnvironmentArguments): Arguments for configuring the environment.
        video_name_prefix (str, optional): Prefix for recorded video files. Defaults to "ddpc-ctrl".
        record_video (bool, optional): Whether to record videos of the environment. Defaults to True.
        use_virtual_actuators (bool, optional): Whether to use virtual actuators. Defaults to False.
        seed (int, optional): Random seed for environment initialization. Defaults to 0.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        env_args: RocketEnvironmentArguments,
        video_name_prefix: str = "ddpc-ctrl",
        record_video: bool = True,
        use_virtual_actuators=False,
        seed: int = 0,
    ):
        """
        Initializes the rocket environment and applies video recording if enabled.
        """
        self.env = gym.make(GYM_ID, render_mode="rgb_array", args=env_args.as_dict(), max_episode_steps=10000)

        if record_video:
            self.env = gym.wrappers.RecordVideo(
                self.env,
                str(path_manager.get_video_path()),
                episode_trigger=lambda x: True,
                name_prefix=video_name_prefix,
            )

        self.lin_sys = self._get_linearized_system()

        self.landed_successfully = False

        self.done = False

        self.use_virtual_actuators = use_virtual_actuators

        self.y = self.reset(seed)

    def get_y_u_reference(self, set_theta_ref_to_zero: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the reference output and input for landing.

        Args:
            set_theta_ref_to_zero (bool, optional): Whether to set the reference angle to zero. Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]: Reference output `y_r` and reference input `u_r`.
        """
        landing_pos = self.env.unwrapped.get_landing_position()  # type: ignore

        if set_theta_ref_to_zero:
            y_r = np.array((landing_pos[0], landing_pos[1], 0, 0, 0, 0))
        else:
            y_r = np.array((landing_pos[0], landing_pos[1], 0, 0, landing_pos[2], 0))

        u_r = np.array([calculate_normalized_thrust_to_hover(), 0, 0])

        return y_r, u_r

    @staticmethod
    def get_dims() -> Tuple[int, int]:
        """Returns m, p"""
        return 3, 6

    def get_output(self) -> np.ndarray:
        """
        Retrieves the current system output.

        Returns:
            np.ndarray: The system output `y`.
        """
        return self.y

    def reset(self, seed: int = 0) -> np.ndarray:
        """
        Resets the environment and returns the initial system output.

        Args:
            seed (int, optional): The random seed for environment reset. Defaults to 0.

        Returns:
            np.ndarray: The initial system output `y`.
        """
        x, _ = self.env.reset(seed=seed)
        self.done = False
        return self._calculate_y(x)

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()

    def step(self, u_next: np.ndarray) -> np.ndarray:
        """
        Takes a step in the environment with the given input.

        Args:
            u_next (np.ndarray): The next input action.

        Returns:
            np.ndarray: The updated system output `y`.
        """
        x, reward, self.done, _, _ = self.env.step(u_next)

        if reward == 100:
            self.landed_successfully = True

        self.y = self._calculate_y(x, u_next)

        return self.y

    def get_input_bounds(self) -> Bounds:
        """
        Retrieves the bounds for valid input values.

        Returns:
            Bounds: The upper and lower input limits.
        """
        if self.use_virtual_actuators:
            max_values = RocketEnvCfgFacade.max_virtual_actuator_values()
            min_values = RocketEnvCfgFacade.min_virtual_actuator_values()
        else:
            max_values = RocketEnvCfgFacade.max_actuator_values_normalized()
            min_values = RocketEnvCfgFacade.min_actuator_values_normalized()

        return Bounds(max_values, min_values)

    def _get_linearized_system(self) -> StateSpaceModel:
        """
        Computes the linearized state-space model of the rocket.

        Returns:
            StateSpaceModel: The discretized linear state-space representation.
        """
        model = SystemModel(self.env.unwrapped)  # type: ignore
        model.calculate_linear_system_matrices()
        model.discretize_system_matrices(sample_time=0.1)
        A, B = model.get_discrete_linear_system_matrices()
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0], B.shape[1]))

        return StateSpaceModel(A, B, C, D)

    def _calculate_y(self, x: np.ndarray, u_next: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the output `y` for the given state `x` and optional input `u_next`.

        If `u_next` is not provided (`None`) and the system's `D` matrix is **not** all zeros,
        an error is raised because the system expects an input.

        Args:
            x (np.ndarray): The state vector.
            u_next (Optional[np.ndarray]): The next input vector. Defaults to None.

        Returns:
            np.ndarray: The computed output `y`.

        Note:
            The last two states of x are ignored as they represent the touching of the grounds as bool.
            These two states are not part of the state space representation.

        Raises:
            ValueError: If `u_next` is required (because `D` is nonzero) but is `None`.
        """
        x = x[:-2]

        if u_next is None:
            if np.any(self.lin_sys.D):
                raise ValueError("Missing input: 'u_next' is required because D is not a zero matrix.")
            return self.lin_sys.C @ x

        return self.lin_sys.C @ x + self.lin_sys.D @ u_next
