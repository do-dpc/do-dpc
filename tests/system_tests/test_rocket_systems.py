# pylint: disable=redefined-outer-name
"""
Tests for the Rocket specific environment
"""
import numpy as np
import pytest

from do_ddpc.ddpc.gamma_ddpc import GammaDPC
from do_ddpc.ddpc.mpc import MPCSystemMatrices
from do_ddpc.ddpc.mpc_oracle import MPCOracle
from do_ddpc.ddpc.spc import SPC
from do_ddpc.ddpc.tpc import TPC
from do_ddpc.environments.rocket_env.rocket_controller_profiles import CTRL_PARAMS_ROCKET
from do_ddpc.environments.rocket_env.rocket_data_collection import collect_trajectory_data_env
from do_ddpc.environments.rocket_env.rocket_env_facade import RocketEnvFacade, RocketEnvironmentArguments
from do_ddpc.environments.rocket_env.rocket_utils import calculate_normalized_thrust_to_hover

WORKING_CTRL = [TPC, SPC, GammaDPC, MPCOracle]


@pytest.fixture(scope="module")
def trajectory_data():
    """
    Collects and saves trajectory data once per test session.
    Returns:
        training_data: Collected trajectory data for testing controllers.
    """
    env = RocketEnvFacade(RocketEnvironmentArguments(), video_name_prefix="data_collection", record_video=False)
    training_data = collect_trajectory_data_env(env, 3, 6)
    env.close()
    return training_data


@pytest.mark.parametrize("controller_cls", WORKING_CTRL)
def test_rocket_systems(controller_cls, trajectory_data):
    """
    Tests each controller with a rocket simulation.

    Args:
        controller_cls (Type[Controller]): The controller class to test.
        trajectory_data (np.ndarray): Collected trajectory data (from fixture).
    """
    env = RocketEnvFacade(RocketEnvironmentArguments(initial_position=(0.8, 0.9, 0)), record_video=False)

    if controller_cls is MPCOracle:
        x_0 = np.array([0.8 * (33.333 / 2), 0.9 * 26.666, 0, 0, 0, 0])
        sys = MPCSystemMatrices(K=np.eye(6), sys=env.lin_sys)
        controller_instance = controller_cls(CTRL_PARAMS_ROCKET, n_state=sys.sys.A.shape[0], sys_data=sys)
        controller_instance.set_state_x(x_0)
    else:
        controller_instance = controller_cls(CTRL_PARAMS_ROCKET, trajectory_data)
    controller_instance.add_input_constraints(env.get_input_bounds())
    controller_instance.build_optimization_problem()

    TIMESTEPS = 2000
    u_stable = np.array([calculate_normalized_thrust_to_hover(), 0, 0])

    u_next, y_current, y_prev = u_stable, env.get_output(), env.get_output()
    z_p_current = np.concatenate((y_prev, u_next))

    for _ in range(CTRL_PARAMS_ROCKET.tau_p):
        controller_instance.update_past_measurements(z_p_current)
    y_prev = env.step(u_next)
    try:
        for i in range(TIMESTEPS):
            if env.landed_successfully:
                print(f"Landed successfully after {i} steps")
                break

            if env.done:
                print(f"Crashed after {i} steps")
                break

            y_r, u_r = env.get_y_u_reference()
            controller_instance.update_tracking_reference(y_r, u_r)

            z_p_current = np.concatenate((y_prev, u_next))
            controller_instance.update_past_measurements(z_p_current)

            controller_instance.solve()
            u_next = controller_instance.get_next_control_action()
            y_prev, y_current = y_current, env.step(u_next)

    finally:
        assert env.landed_successfully, "Rocket did not land successfully!"
        env.close()
