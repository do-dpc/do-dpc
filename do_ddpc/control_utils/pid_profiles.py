"""
PID Profiles for controlling different environments.
"""

from do_ddpc.control_utils.pid import PIDGains, MIMOPIDController, PIDController
from do_ddpc.control_utils.pid_control_utils import (
    PIDCombo,
    one_D_double_integrator_output_to_err_der_err,
    three_D_double_integrator_output_to_err_der_err,
    rocket_output_to_err_der_err,
)

PID_GAIN_DOUBLE_INTEGRATOR = PIDGains(Kp=2.5, Ki=0, Kd=3)

# Rocket PID Gains
PID_GAIN_ROCKET_STATE_X = PIDGains(Kp=10, Ki=0, Kd=10)
PID_GAIN_ROCKET_STATE_Y = PIDGains(Kp=5, Ki=0, Kd=6)
PID_GAIN_ROCKET_STATE_THETA = PIDGains(Kp=0.085, Ki=0.001, Kd=10.55)

DOUBLE_INT_PID = PIDController(PID_GAIN_DOUBLE_INTEGRATOR, dt=1)

# PID Combos
ONE_D_DOUBLE_INT_PID_COMBO = PIDCombo(
    MIMOPIDController([DOUBLE_INT_PID]), one_D_double_integrator_output_to_err_der_err
)

THREE_D_DOUBLE_INT_PID_COMBO = PIDCombo(
    MIMOPIDController([DOUBLE_INT_PID, DOUBLE_INT_PID, DOUBLE_INT_PID]),
    three_D_double_integrator_output_to_err_der_err,
)

ROCKET_PID_COMBO = PIDCombo(
    MIMOPIDController(
        [
            PIDController(PID_GAIN_ROCKET_STATE_X, dt=1),
            PIDController(PID_GAIN_ROCKET_STATE_Y, dt=1),
            PIDController(PID_GAIN_ROCKET_STATE_THETA, dt=1),
        ]
    ),
    rocket_output_to_err_der_err,
)
