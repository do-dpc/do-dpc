"""
Tests for the PID controller and MIMO PID controller
"""

import numpy as np
import pytest

from do_dpc.control_utils.pid import PIDController, PIDGains, MIMOPIDController


def test_pid_initialization():
    """Test if PIDController initializes correctly and raises errors for invalid dt."""
    gains = PIDGains(Kp=1.0, Ki=0.5, Kd=0.1)

    # Valid initialization
    pid = PIDController(gains, dt=0.1)
    assert pid.dt == 0.1
    assert pid.gains.Kp == 1.0
    assert pid.gains.Ki == 0.5
    assert pid.gains.Kd == 0.1

    # Invalid dt should raise ValueError
    with pytest.raises(ValueError, match="Time step dt must be positive."):
        PIDController(gains, dt=0)


def test_pid_proportional_action():
    """Test if PID responds correctly to proportional gain."""
    gains = PIDGains(Kp=2.0, Ki=0.0, Kd=0.0)
    pid = PIDController(gains, dt=0.1)

    error = 5.0
    control_signal = pid.compute(error)

    assert control_signal == pytest.approx(10.0, rel=1e-6)


def test_pid_integral_action():
    """Test if PID correctly accumulates integral action over multiple steps."""
    gains = PIDGains(Kp=0.0, Ki=1.0, Kd=0.0)
    pid = PIDController(gains, dt=0.1)

    error = 2.0
    for _ in range(10):  # Apply the same error for 10 steps
        control_signal = pid.compute(error)

    assert control_signal == pytest.approx(2.0, rel=1e-6)  # (2 * 0.1 * 10 = 2.0)


def test_pid_derivative_action():
    """Test if PID correctly applies derivative action."""
    gains = PIDGains(Kp=0.0, Ki=0.0, Kd=1.0)
    pid = PIDController(gains, dt=0.1, alpha=1.0)  # No filtering for test

    error_t1 = 0.0
    error_t2 = 5.0

    # First step, no derivative
    control_signal_1 = pid.compute(error_t1)
    assert control_signal_1 == pytest.approx(0.0, rel=1e-6)

    # Second step, derivative should be (5 - 0) / 0.1 = 50
    control_signal_2 = pid.compute(error_t2)
    assert control_signal_2 == pytest.approx(50.0, rel=1e-6)


def test_pid_derivative_filtering():
    """Test if the low-pass filter smoothens derivative response."""
    gains = PIDGains(Kp=0.0, Ki=0.0, Kd=1.0)
    pid = PIDController(gains, dt=0.1, alpha=0.5)  # Filter factor 0.5

    pid.compute(0.0)  # First step
    control_signal = pid.compute(5.0)  # Second step, should be smoothed

    expected_derivative = 50 * 0.5  # Low-pass filter applied
    assert control_signal == pytest.approx(expected_derivative, rel=1e-6)


def test_pid_compute_with_derivative():
    """Test `compute_with_derivative` function."""
    gains = PIDGains(Kp=1.0, Ki=0.5, Kd=0.1)
    pid = PIDController(gains, dt=0.1)

    error = 3.0
    error_derivative = -1.0

    control_signal = pid.compute_with_derivative(error, error_derivative)

    expected_output = (1.0 * 3.0) + (0.5 * 3.0 * 0.1) + (0.1 * -1.0)  # Kp*e + Ki*sum(e*dt) + Kd*de
    assert control_signal == pytest.approx(expected_output, rel=1e-6)


def test_mimo_pid_initialization():
    """Test if MIMO_PIDController initializes correctly."""
    gains1 = PIDGains(Kp=1.0, Ki=0.5, Kd=0.1)
    gains2 = PIDGains(Kp=2.0, Ki=1.0, Kd=0.2)

    pid1 = PIDController(gains1, dt=0.1)
    pid2 = PIDController(gains2, dt=0.1)

    mimo_pid = MIMOPIDController([pid1, pid2])
    assert len(mimo_pid.pid_controllers) == 2
    assert mimo_pid.pid_controllers[0].gains.Kp == 1.0
    assert mimo_pid.pid_controllers[1].gains.Kp == 2.0


def test_mimo_pid_compute():
    """Test if MIMO_PIDController computes control actions correctly."""
    gains1 = PIDGains(Kp=1.0, Ki=0.5, Kd=0.1)
    gains2 = PIDGains(Kp=2.0, Ki=1.0, Kd=0.2)

    pid1 = PIDController(gains1, dt=0.1)
    pid2 = PIDController(gains2, dt=0.1)

    mimo_pid = MIMOPIDController([pid1, pid2])

    errors = np.array([1.0, 2.0])
    control_signals = mimo_pid.compute(errors)

    expected_outputs = np.array([1.15, 4.6])  # Kp * error for each controller
    np.testing.assert_allclose(control_signals, expected_outputs, rtol=1e-6)


def test_mimo_pid_compute_with_derivative():
    """Test if MIMO_PIDController computes control actions correctly with derivatives."""
    gains1 = PIDGains(Kp=1.0, Ki=0.5, Kd=0.1)
    gains2 = PIDGains(Kp=2.0, Ki=1.0, Kd=0.2)

    pid1 = PIDController(gains1, dt=0.1)
    pid2 = PIDController(gains2, dt=0.1)

    mimo_pid = MIMOPIDController([pid1, pid2])

    errors = np.array([1.0, 2.0])
    error_derivatives = np.array([0.5, -0.5])
    control_signals = mimo_pid.compute_with_derivative(errors, error_derivatives)

    expected_outputs = np.array([1.1, 4.1])
    np.testing.assert_allclose(control_signals, expected_outputs, rtol=1e-6)
