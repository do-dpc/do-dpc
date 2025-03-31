"""
Working set of control parameters
"""

import numpy as np

from do_ddpc.ddpc.ddpc_structs import DPCParameters

Q = np.diag([5, 6, 30, 40, 1000, 80])
R = np.diag([1, 0.1, 0.1])
Q_final = np.diag([300, 500, 100, 2, 4000, 800])

tau_p = 1
tau_f = 10

CTRL_PARAMS_ROCKET = DPCParameters(Q=Q, R=R, tau_p=tau_p, tau_f=tau_f, Q_final=Q_final)
