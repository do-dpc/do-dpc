# pylint: disable=duplicate-code
"""
DeePC Test Module.

This module tests the DeePC implementation, considering two key aspects:
1. The number of samples (`N`) is reduced to ensure tests run within a reasonable time.
2. The presence of tunable parameters, which require validation.

Different DeePC setups are tested to verify correct functionality.
"""

from typing import Optional

import numpy as np
import pytest

from do_dpc.control_utils.lti_systems import (
    create_pre_stabilized_1D_double_integrator,
    create_1D_double_integrator,
    LTISimulator,
)
from do_dpc.control_utils.pid_control_utils import PIDCombo
from do_dpc.dpc.dpc_structs import DPCParameters
from do_dpc.control_utils.control_structs import Bounds
from do_dpc.dpc.deepc import DeePCSpecificParameters, DeePC
from tests.system_tests.test_maps import PID_COMBO_CONFIG
from tests.system_tests.util_functions import collect_trajectory_data, run_simulation

# Simulation Parameters
N_SAMPLES = 60
EXC_DEV = 1
TARGET = np.array([7, 0])
A_TOL = 0.5
TIMESTEPS = 20
BOUND_TOL = 0.1
Y_BOUNDS = Bounds(max_values=np.array([5, np.inf]), min_values=np.array([0, -np.inf]))
U_BOUNDS = Bounds(max_values=np.array([0.01]), min_values=np.array([-0.2]))

# DeePC Tunable Parameters
PARAMS_NO_REG = DeePCSpecificParameters(suppress_warnings=True)
PARAMS_SIGMA = DeePCSpecificParameters(lambda_sigma=1e5, suppress_warnings=True)
PARAMS_SIGMA_G_1 = DeePCSpecificParameters(lambda_sigma=1e5, lambda_g_1=10)
PARAMS_SIGMA_G_2 = DeePCSpecificParameters(lambda_sigma=1e5, lambda_g_2=1)
PARAMS_SIGMA_G_1_KERNEL = DeePCSpecificParameters(lambda_sigma=1e5, lambda_g_1=10, lambda_p=100)
PARAMS_SIGMA_G_2_KERNEL = DeePCSpecificParameters(lambda_sigma=1e5, lambda_g_2=1, lambda_p=100)

# Concatenated Parameters, note that without lambda_sigma and g, DeePC does not work in the present of noise
DEEPC_PARAMS_ALL = [
    PARAMS_NO_REG,
    PARAMS_SIGMA,
    PARAMS_SIGMA_G_1,
    PARAMS_SIGMA_G_2,
    PARAMS_SIGMA_G_1_KERNEL,
    PARAMS_SIGMA_G_2_KERNEL,
]
DEEPC_PARAMS_NOISE_RES = [PARAMS_SIGMA_G_1, PARAMS_SIGMA_G_2, PARAMS_SIGMA_G_1_KERNEL, PARAMS_SIGMA_G_2_KERNEL]


def create_and_test_controller(
    system: LTISimulator,
    deepc_params: DeePCSpecificParameters,
    u_bounds: Optional[Bounds] = None,
    y_bounds: Optional[Bounds] = None,
    pid_combo: Optional[PIDCombo] = None,
):
    """
    Creates and tests a predictive controller on a given system.

    Args:
        system (LTISimulator): The system to control.
        deepc_params (DeePCSpecificParameters): Set of tuned parameters
        u_bounds (Bounds, optional): Input constraints.
        y_bounds (Bounds, optional): Output constraints.
        pid_combo (PIDCombo, optional): PID controller with function to get state to error.

    Returns:
        None. Runs assertions to validate performance.
    """
    _, m, p = system.get_dims()
    training_data = collect_trajectory_data(system, m, p, pid_combo, n_samples=N_SAMPLES)
    controller = DeePC(DPCParameters(Q=np.diag([100, 1]), R=0.001 * np.eye(m), tau_p=3, tau_f=3), training_data)

    controller.set_specific_parameters(deepc_params)

    if u_bounds:
        controller.add_input_constraints(u_bounds)
    if y_bounds:
        controller.add_terminal_output_constraints(y_bounds)

    controller.build_optimization_problem()
    y_final = run_simulation(system, controller, TARGET, m, u_bounds)

    if not y_bounds and not u_bounds:
        assert np.isclose(
            y_final[0], TARGET[0], atol=A_TOL
        ), f"Final output {y_final[0]} did not reach target {TARGET[0]}."

    if y_bounds:
        assert np.all(y_final + BOUND_TOL >= y_bounds.min_values), "System output below min bound."
        assert np.all(y_final - BOUND_TOL <= y_bounds.max_values), "System output exceeds max bound."


@pytest.mark.parametrize("deepc_params", DEEPC_PARAMS_ALL)
def test_unconstrained_stable_double_integrator(deepc_params: DeePCSpecificParameters):
    """
    Tests the stable double integrator without any constraints
    """
    create_and_test_controller(create_pre_stabilized_1D_double_integrator(), deepc_params)


@pytest.mark.parametrize("deepc_params", DEEPC_PARAMS_NOISE_RES)
def test_low_meas_noise_unstable_double_integrator(deepc_params: DeePCSpecificParameters):
    """
    Tests the unstable double integrator with low measurement noise
    """
    meas_noise_std = np.array([0.01])
    meas_noise_seed = 6473

    create_and_test_controller(
        create_1D_double_integrator(meas_noise_std=meas_noise_std, meas_noise_seed=meas_noise_seed),
        deepc_params,
        pid_combo=PID_COMBO_CONFIG["1D_double_integrator"],
    )


@pytest.mark.parametrize("deepc_params", DEEPC_PARAMS_NOISE_RES)
def test_meas_noise_stable_double_integrator(deepc_params: DeePCSpecificParameters):
    """
    Tests the unstable double integrator with low measurement noise
    """
    meas_noise_std = np.array([0.02])
    meas_noise_seed = 6409

    create_and_test_controller(
        create_pre_stabilized_1D_double_integrator(meas_noise_std=meas_noise_std, meas_noise_seed=meas_noise_seed),
        deepc_params,
    )


@pytest.mark.parametrize("deepc_params", DEEPC_PARAMS_ALL)
def test_unconstrained_unstable_double_integrator(deepc_params: DeePCSpecificParameters):
    """
    Tests the unstable double integrator without any constraints
    """
    create_and_test_controller(
        create_1D_double_integrator(), deepc_params, pid_combo=PID_COMBO_CONFIG["1D_double_integrator"]
    )


@pytest.mark.parametrize("deepc_params", DEEPC_PARAMS_ALL)
def test_constrained_input_stable_double_integrator(deepc_params: DeePCSpecificParameters):
    """
    Tests the unstable double integrator without input constraints
    """
    create_and_test_controller(create_pre_stabilized_1D_double_integrator(), deepc_params, u_bounds=U_BOUNDS)


@pytest.mark.parametrize("deepc_params", DEEPC_PARAMS_ALL)
def test_constrained_output_stable_double_integrator(deepc_params: DeePCSpecificParameters):
    """
    Tests the unstable double integrator with terminal output constraints
    """
    create_and_test_controller(create_pre_stabilized_1D_double_integrator(), deepc_params, y_bounds=Y_BOUNDS)
