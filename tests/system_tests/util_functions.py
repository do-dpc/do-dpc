# pylint: disable=duplicate-code
"""
This module contains different utility functions that are used by the DPC tests.
"""

from typing import Optional, Type

import numpy as np

from do_ddpc.control_utils.control_structs import Bounds
from do_ddpc.control_utils.lti_systems import LTISimulator
from do_ddpc.control_utils.pid_control_utils import PIDCombo
from do_ddpc.control_utils.trajectory_collector import collect_trajectory_data
from do_ddpc.ddpc.ddpc import DPC
from do_ddpc.ddpc.ddpc_structs import DPCParameters
from do_ddpc.ddpc.mpc import MPCSystemMatrices
from do_ddpc.ddpc.mpc_nfour_sid import MPCNFourSID
from do_ddpc.ddpc.mpc_oracle import MPCOracle

# Simulation Parameters
N_SAMPLES = 1000
EXC_DEV = 1
A_TOL = 0.5
TIMESTEPS = 20
BOUND_TOL = 0.1


def run_simulation(
    system: LTISimulator, controller_instance: DPC, target_y: np.ndarray, m: int, u_bounds: Optional[Bounds] = None
):
    """
    Runs a closed-loop simulation using a predictive controller on an LTI system.

    Args:
        system (LTISimulator): The system being controlled.
        controller_instance: The predictive controller.
        target_y (np.ndarray): The desired reference output.
        m (int): Number of control inputs.
        u_bounds (Bounds, optional): Input bounds.

    Returns:
        np.ndarray: Final output of the system after simulation.
    """
    system.reset_x_to_x_0()
    controller_instance.update_tracking_reference(y_r_new=target_y, u_r_new=np.zeros((m,)))
    u_next, y_current, y_prev = np.zeros((m,)), system.step(np.zeros((m,))), system.step(np.zeros((m,)))

    for t in range(TIMESTEPS):
        z_p_current = np.concatenate((y_prev, u_next))
        controller_instance.update_past_measurements(z_p_current)
        controller_instance.solve()
        u_next = controller_instance.get_next_control_action()
        y_prev, y_current = y_current, system.step(u_next)

        assert np.all(np.abs(u_next) < 100), f"Control input {u_next} is too large at step {t}"

        if u_bounds:
            assert np.all(u_next + BOUND_TOL >= u_bounds.min_values), f"Step {t}: Control input below min bound."
            assert np.all(u_next - BOUND_TOL <= u_bounds.max_values), f"Step {t}: Control input exceeds max bound."

        if np.allclose(y_current, target_y, atol=A_TOL):
            print(f"System stabilized at step {t}: y_current = {y_current}")
            break

    return y_current


# pylint: disable=too-many-arguments, too-many-positional-arguments
def create_and_test_controller(
    system: LTISimulator,
    controller_cls: Type[DPC],
    ddpc_params: DPCParameters,
    n_block_rows: int,
    target: np.ndarray,
    u_bounds: Optional[Bounds] = None,
    y_bounds: Optional[Bounds] = None,
    pid_combo: Optional[PIDCombo] = None,
):
    """
    Creates and tests a predictive controller on a given system.

    Args:
        system (LTISimulator): The system to control.
        controller_cls (Type): The predictive controller class.
        ddpc_params (DPCParameters): Control parameters
        n_block_rows (int): Needed for n4sid.
        target (np.ndarray): Target for the output reference.
        u_bounds (Bounds, optional): Input constraints.
        y_bounds (Bounds, optional): Output constraints.
        pid_combo (PIDCombo, optional): PID controller with function to get state to error.

    Returns:
        None. Runs assertions to validate performance.
    """
    _, m, p = system.get_dims()
    training_data = collect_trajectory_data(system, m, p, pid_combo)

    if controller_cls is MPCOracle:
        sys_data = MPCSystemMatrices(K=system.calculate_inf_hor_Kalman_gain_K(), sys=system.sys)
        controller = controller_cls(ddpc_params, n_state=system.x.shape[0], sys_data=sys_data)  # type: ignore
        controller.set_state_x(x_new=system.x_0)
    elif controller_cls is MPCNFourSID:
        controller = controller_cls(ddpc_params, training_data, n_block_rows=n_block_rows)
    else:
        controller = controller_cls(ddpc_params, training_data)

    if u_bounds:
        controller.add_input_constraints(u_bounds)
    if y_bounds:
        controller.add_terminal_output_constraints(y_bounds)

    controller.build_optimization_problem()
    y_final = run_simulation(system, controller, target, m, u_bounds)

    if not y_bounds and not u_bounds:
        assert np.isclose(
            y_final[0], target[0], atol=A_TOL
        ), f"Final output {y_final[0]} did not reach target {target[0]}."

    if y_bounds:
        assert np.all(
            y_final + BOUND_TOL >= y_bounds.min_values
        ), f"System output:{y_final} below min bound:{y_bounds.min_values}."
        assert np.all(
            y_final - BOUND_TOL <= y_bounds.max_values
        ), f"System output:{y_final} exceeds max bound: {y_bounds.max_values}."
