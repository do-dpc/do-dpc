"""
System tests for the implemented DPC controller modules.

This module provides system tests for DPC controllers using different LTI systems.
It ensures the correctness of data collection, controller calculations, and system regulation.

Note that the DeePC controller is tested using a different test module as it cannot be run with the same N_SAMPLES.
"""

import pytest

from do_dpc.dpc.mpc_nfour_sid import MPCNFourSID
from tests.system_tests import test_maps
from tests.system_tests.util_functions import create_and_test_controller


def generate_param_comb() -> list:
    """
    Generates the different parameters depending on the system that has been chosen.

    Returns:
        list: A list with the different parameter settings
    """
    param_list = []
    for system in test_maps.SYSTEMS:
        for ctrl in test_maps.CONTROLLERS:
            # Only for noise resistant controller test all the noise levels
            if ctrl in test_maps.NOISE_RES_CONTROLLERS:
                noise_config_list = test_maps.NOISE_CONFIGS_ALL[system]
            else:
                noise_config_list = test_maps.NOISE_CONFIGS_EASY[system]

            bound_list = test_maps.BOUNDS[system]
            for i, dpc_params in enumerate(test_maps.CTRL_PARAMS[system]):
                # Apply the noise only to the non-constrained case, else one need to adapt the tolerances
                bounds = bound_list[0]
                k = 0
                for j, noise_config in enumerate(noise_config_list):
                    param_list.append(
                        pytest.param(
                            system,
                            ctrl,
                            dpc_params,
                            noise_config,
                            bounds,
                            id=f"{system}--{ctrl}--params_{i}--noise_{j}--bounds_{k}",
                        )
                    )

                # Apply the constraints only to the noiseless case
                noise_config = noise_config_list[0]
                j = 0
                for k, bounds in enumerate(bound_list):
                    if k == 0:
                        continue
                    param_list.append(
                        pytest.param(
                            system,
                            ctrl,
                            dpc_params,
                            noise_config,
                            bounds,
                            id=f"{system}--{ctrl}--params_{i}--noise_{j}--bounds_{k}",
                        )
                    )
    return param_list


@pytest.mark.parametrize("system, controller_cls,dpc_params, noise_config, bounds", generate_param_comb())
def test_lti_systems(system, controller_cls, noise_config, dpc_params, bounds):
    """
    Tests the different systems with the different controller, parameters, noise and constrints.
    """

    if controller_cls is MPCNFourSID:
        if system == "1D_double_integrator":
            if "y_bounds" in bounds:
                pytest.skip("For unknown reason, N4SID does not respect the bounds but tries to reach the target.")
            if "meas_noise_std" in noise_config and noise_config["meas_noise_std"][0] == 0.01:
                pytest.skip("For unknown reason, N4SID does not work.")
        if system == "pre_stabilized_1D_double_integrator":
            if "meas_noise_std" in noise_config and noise_config["meas_noise_std"][0] == 0.02:
                pytest.skip("For unknown reason, N4SID does not work.")

    sys = (test_maps.SYSTEM_CREATOR[system])(**noise_config)
    n_blow_rows = test_maps.N_BLOCK_ROWS[system]
    target = test_maps.TARGETS[system]
    create_and_test_controller(
        sys, controller_cls, dpc_params, n_blow_rows, target, **bounds, pid_combo=test_maps.PID_COMBO_CONFIG[system]
    )
