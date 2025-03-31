"""
This module tests the basic functionality of the DPC library.
"""

import numpy as np
import pytest

from do_ddpc.control_utils.lti_systems import (
    create_pre_stabilized_1D_double_integrator,
    LTISimulator,
)
from do_ddpc.control_utils.noise_generators import WhiteNoiseGenerator
from do_ddpc.control_utils.trajectory_collector import TrajectoryCollector
from do_ddpc.ddpc.ddpc_structs import DPCParameters
from do_ddpc.control_utils.control_structs import InputOutputTrajectory
from do_ddpc.ddpc.tpc import TPC
from tests.system_tests.util_functions import collect_trajectory_data, run_simulation, BOUND_TOL

CTRL_BASIC = DPCParameters(Q=np.diag([100, 1]), R=0.001 * np.eye(1), tau_p=3, tau_f=3)
BASIC_SYSTEM = create_pre_stabilized_1D_double_integrator()
BASIC_DPC_CTRL = TPC
TARGET = np.array((7, 0))


def test_data_collection_with_double_integrator(
    trajectory_collector: TrajectoryCollector,
    white_noise_generator: WhiteNoiseGenerator,
    double_integrator: LTISimulator,
):
    """
    Tests data collection process using a double integrator system.

    - Generates control inputs.
    - Steps through the system.
    - Stores system outputs.
    - Ensures the trajectory data is complete and valid.

    Args:
        trajectory_collector (TrajectoryCollector): Data collector instance.
        white_noise_generator (WhiteNoiseGenerator): Generates control inputs.
        double_integrator (LTISimulator): The double integrator system.

    Returns:
        None. Runs assertions to validate data collection.
    """
    for _ in range(10):
        u_next = white_noise_generator.generate()
        y_next = double_integrator.step(u_next)
        trajectory_collector.store_measurements(y_next, u_next)

    trajectory_data = trajectory_collector.get_trajectory_data()
    assert isinstance(trajectory_data, InputOutputTrajectory)
    assert trajectory_data.y.shape == (2, 10)
    assert trajectory_data.u.shape == (1, 10)
    assert not np.isnan(trajectory_data.y).any()
    assert not np.isnan(trajectory_data.u).any()


def test_problem_is_infeasibility():
    """
    Tests that the optimization problem raises a RuntimeError when it becomes dual infeasible.
    """
    system = create_pre_stabilized_1D_double_integrator()
    _, m, p = system.get_dims()
    training_data = collect_trajectory_data(system, m, p)

    controller = BASIC_DPC_CTRL(CTRL_BASIC, training_data)

    tau_f = 3
    A = np.zeros((2, (m + p) * tau_f))
    A[0, 0] = 1
    A[1, 0] = -1
    b = np.array([1, -2])

    controller.add_linear_constraints(A, b)

    controller.build_optimization_problem()

    # Check for RuntimeError due to infeasibility
    with pytest.raises(RuntimeError, match="Optimization failed: Problem status is infeasible"):
        run_simulation(system, controller, TARGET, m)


def test_linear_constraints():
    """
    Tests the linear constraint function.
    Added terminal output constraints to test them.
    """
    system = BASIC_SYSTEM
    _, m, p = system.get_dims()
    training_data = collect_trajectory_data(system, m, p)

    controller = BASIC_DPC_CTRL(CTRL_BASIC, training_data)

    tau_f = 3
    A = np.zeros((2, (m + p) * tau_f))
    A[0, p * (tau_f - 1)] = 1
    A[1, p * (tau_f - 1)] = -1
    b = np.array([5, 0])

    controller.add_linear_constraints(A, b)

    controller.build_optimization_problem()

    y_final = run_simulation(system, controller, TARGET, m)

    assert np.all(y_final[0] + BOUND_TOL >= 0), "System output below min bound."
    assert np.all(y_final[0] - BOUND_TOL <= 5), "System output exceeds max bound."
