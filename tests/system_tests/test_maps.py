"""
This module contains the maps for the tests.
With those maps, the parametrized test and be created.
"""

import numpy as np

from do_ddpc.control_utils.control_structs import Bounds
from do_ddpc.control_utils.lti_systems import (
    create_pre_stabilized_1D_double_integrator,
    create_1D_double_integrator,
    create_3D_double_integrator,
    create_landau_benchmark,
)
from do_ddpc.control_utils.pid_profiles import THREE_D_DOUBLE_INT_PID_COMBO, ONE_D_DOUBLE_INT_PID_COMBO
from do_ddpc.ddpc.ddpc_structs import DPCParameters
from do_ddpc.ddpc.gamma_ddpc import GammaDPC
from do_ddpc.ddpc.mpc_oracle import MPCOracle
from do_ddpc.ddpc.spc import SPC
from do_ddpc.ddpc.tpc import TPC

SYSTEMS = ["pre_stabilized_1D_double_integrator", "1D_double_integrator", "3D_double_integrator", "landau_benchmark"]

SYSTEM_CREATOR = {
    "pre_stabilized_1D_double_integrator": create_pre_stabilized_1D_double_integrator,
    "1D_double_integrator": create_1D_double_integrator,
    "3D_double_integrator": create_3D_double_integrator,
    "landau_benchmark": create_landau_benchmark,
}

# Define controller classes
# MPCN4SID is for the moment removed as it made some problems with the SCS solver
CONTROLLERS = [TPC, SPC, GammaDPC, MPCOracle]
NOISE_RES_CONTROLLERS = [TPC, MPCOracle]

# Define control parameters for different systems
CTRL_PARAMS = {
    "pre_stabilized_1D_double_integrator": [
        DPCParameters(Q=np.diag([100, 1]), R=0.001 * np.eye(1), tau_p=3, tau_f=3),
        DPCParameters(
            Q=np.diag([100, 1]),
            R=0.001 * np.eye(1),
            tau_p=3,
            tau_f=3,
            Q_final=np.diag([1000, 12]),
            R_final=np.diag([1]),
        ),
        DPCParameters(Q=np.diag([100, 1]), R=0.001 * np.eye(1), tau_p=3, tau_f=3, R_delta_first=10 * np.eye(1)),
    ],
    "1D_double_integrator": [
        DPCParameters(Q=np.diag([100, 1]), R=0.001 * np.eye(1), tau_p=3, tau_f=3),
        DPCParameters(Q=np.diag([100, 1]), R=0.001 * np.eye(1), tau_p=3, tau_f=3, R_delta=10 * np.eye(1)),
        DPCParameters(
            Q=np.diag([100, 1]),
            R=0.001 * np.eye(1),
            tau_p=3,
            tau_f=3,
            R_delta=1 * np.eye(1),
            R_delta_first=10 * np.eye(1),
        ),
    ],
    "3D_double_integrator": [DPCParameters(Q=np.diag([100, 1, 100, 1, 100, 1]), R=0.001 * np.eye(3), tau_p=3, tau_f=3)],
    "landau_benchmark": [DPCParameters(Q=np.diag([200]), R=0.001 * np.eye(1), tau_p=5, tau_f=20)],
}

PID_COMBO_CONFIG = {
    "pre_stabilized_1D_double_integrator": None,
    "1D_double_integrator": ONE_D_DOUBLE_INT_PID_COMBO,
    "3D_double_integrator": THREE_D_DOUBLE_INT_PID_COMBO,
    "landau_benchmark": None,
}

# Define noise configurations
NOISE_CONFIGS_EASY = {
    "pre_stabilized_1D_double_integrator": [
        {},
        {"meas_noise_std": np.array([0.01]), "meas_noise_seed": 6473},
        {"meas_noise_std": np.array([0.02]), "meas_noise_seed": 6471},
        {"meas_noise_std": np.array([0.03]), "meas_noise_seed": 6471},
        {
            "meas_noise_std": np.array([0.01]),
            "process_noise_std": np.array([0.01, 0.005]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
        {
            "meas_noise_std": np.array([0.01]),
            "process_noise_std": np.array([0.02, 0.01]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
    ],
    "1D_double_integrator": [
        {},
        {"meas_noise_std": np.array([0.005]), "meas_noise_seed": 6473},
        {"meas_noise_std": np.array([0.01]), "meas_noise_seed": 6471},
        {
            "meas_noise_std": np.array([0.005]),
            "process_noise_std": np.array([0.01, 0.005]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
    ],
    "3D_double_integrator": [{}, {"meas_noise_std": np.array([0.005]), "meas_noise_seed": 6473}],
    "landau_benchmark": [{}, {"meas_noise_std": np.array([0.005]), "meas_noise_seed": 6473}],
}

NOISE_CONFIGS_ALL = {
    "pre_stabilized_1D_double_integrator": [
        {},
        {"meas_noise_std": np.array([0.02]), "meas_noise_seed": 6473},
        {"meas_noise_std": np.array([0.03]), "meas_noise_seed": 6471},
        {
            "meas_noise_std": np.array([0.01]),
            "process_noise_std": np.array([0.01, 0.005]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
        {
            "meas_noise_std": np.array([0.02]),
            "process_noise_std": np.array([0.05, 0.01]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
    ],
    "1D_double_integrator": [
        {},
        {"meas_noise_std": np.array([0.01]), "meas_noise_seed": 6473},
        {"meas_noise_std": np.array([0.02]), "meas_noise_seed": 6471},
        {
            "meas_noise_std": np.array([0.01]),
            "process_noise_std": np.array([0.01, 0.005]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
        {
            "meas_noise_std": np.array([0.02]),
            "process_noise_std": np.array([0.05, 0.01]),
            "meas_noise_seed": 6423,
            "process_noise_seed": 6211,
        },
    ],
    "3D_double_integrator": [{}, {"meas_noise_std": np.array([0.005]), "meas_noise_seed": 6473}],
    "landau_benchmark": [{}, {"meas_noise_std": np.array([0.005]), "meas_noise_seed": 6473}],
}

# Define constraints
BOUNDS = {
    "pre_stabilized_1D_double_integrator": [
        {},
        {"y_bounds": Bounds(max_values=np.array([5, np.inf]), min_values=np.array([0, -np.inf]))},
        {"u_bounds": Bounds(max_values=np.array([0.01]), min_values=np.array([-0.2]))},
    ],
    "1D_double_integrator": [
        {},
        {"y_bounds": Bounds(max_values=np.array([5, np.inf]), min_values=np.array([0, -np.inf]))},
        {"u_bounds": Bounds(max_values=np.array([0.01]), min_values=np.array([-0.2]))},
    ],
    "3D_double_integrator": [
        {},
        {"u_bounds": Bounds(max_values=np.array([0.01, 0.01, 0.01]), min_values=np.array([-0.2, -0.2, -0.2]))},
    ],
    "landau_benchmark": [
        {},
        {"u_bounds": Bounds(max_values=np.array([0.01]), min_values=np.array([-0.2]))},
    ],
}

TARGETS = {
    "pre_stabilized_1D_double_integrator": np.array([7, 0]),
    "1D_double_integrator": np.array([7, 0]),
    "3D_double_integrator": np.array([7, 0, 7, 0, 7, 0]),
    "landau_benchmark": np.array([1]),
}

N_BLOCK_ROWS = {
    "pre_stabilized_1D_double_integrator": 2,
    "1D_double_integrator": 2,
    "3D_double_integrator": 6,
    "landau_benchmark": 4,
}
