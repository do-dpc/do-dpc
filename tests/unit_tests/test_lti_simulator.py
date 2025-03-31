"""
Unit tests for the LTISimulator class.

This module tests the behavior of the LTISimulator, including
state initialization, step updates, noise handling, and input validation.
"""

import numpy as np
import pytest

from do_dpc.control_utils.lti_systems import StateSpaceModel, LTISimulator
from do_dpc.control_utils.noise_generators import WhiteNoiseGenerator


def test_initial_state(double_integrator):
    """Test if the initial state of the double integrator system is correctly set."""
    assert np.array_equal(double_integrator.get_state(), np.array([0, 0]))


def test_step_update(double_integrator):
    """Test if the state updates correctly after one step with a control input."""
    u = np.array([1.0])
    y = double_integrator.step(u)

    assert y.shape == (2,), "Output shape should match the system output dimension."
    assert double_integrator.get_state().shape == (2,), "State shape should remain consistent."


def test_multiple_steps(double_integrator):
    """Test if the system state updates correctly over multiple steps."""
    u = np.array([1.0])

    for _ in range(5):
        y = double_integrator.step(u)

    assert y.shape == (2,)
    assert double_integrator.get_state().shape == (2,)


def test_invalid_control_input(double_integrator):
    """Test if providing an incorrectly shaped input raises ValueError."""
    with pytest.raises(ValueError):
        double_integrator.step(np.array([1.0, 2.0]))  # Wrong shape


def test_noisy_system():
    """Test if a system with noise updates correctly."""
    system = StateSpaceModel(
        A=np.array([[1, 1], [0, 1]]), B=np.array([[0], [1]]), C=np.array([[1, 0], [0, 1]]), D=np.array([[0], [0]])
    )

    measurement_noise = WhiteNoiseGenerator(std=np.array([0.02, 0.02]))
    process_noise = WhiteNoiseGenerator(std=np.array([0.05, 0.01]))

    simulator = LTISimulator(
        system, x_0=np.array([0, 0]), measurement_noise=measurement_noise, process_noise=process_noise
    )

    u = np.array([1.0])
    y = simulator.step(u)

    assert y.shape == (2,), "Output shape should match the system output dimension."
    assert simulator.get_state().shape == (2,), "State shape should remain consistent even with noise."


def test_reset_state(double_integrator):
    """Test if resetting the state brings it back to the initial condition."""
    u = np.array([1.0])
    double_integrator.step(u)  # Change state
    double_integrator.reset_x_to_x_0()  # Reset state

    assert np.array_equal(double_integrator.get_state(), np.array([0, 0])), "State should reset to initial conditions."


def test_output_with_control_input(double_integrator):
    """Test if the output is computed correctly when a control input is provided."""
    u = np.array([1.0])
    y = double_integrator.get_output(u)

    assert y.shape == (2,), "Output shape should match the expected system output dimension."


def test_output_without_control_input(double_integrator):
    """Test if the output is computed correctly without a control input."""
    y = double_integrator.get_output()

    assert y.shape == (2,), "Output shape should match the expected system output dimension."


def test_output_without_noise(double_integrator):
    """Test if the noise-free output computation works correctly."""
    u = np.array([1.0])
    y = double_integrator.get_output_without_noise(u)

    assert y.shape == (2,), "Output shape should match the expected system output dimension."


def test_system_dimensions(double_integrator):
    """Test if the system dimensions are correctly returned."""
    n, m, p = double_integrator.get_dims()

    assert n == 2, "State dimension should be 2."
    assert m == 1, "Input dimension should be 1."
    assert p == 2, "Output dimension should be 2."


def test_state_update_with_process_noise():
    """Test if the state updates correctly when process noise is present."""
    system = StateSpaceModel(
        A=np.array([[1, 1], [0, 1]]), B=np.array([[0], [1]]), C=np.array([[1, 0], [0, 1]]), D=np.array([[0], [0]])
    )

    process_noise = WhiteNoiseGenerator(std=np.array([0.1, 0.1]))  # Nonzero process noise

    simulator = LTISimulator(system, x_0=np.array([0, 0]), process_noise=process_noise)

    u = np.array([1.0])
    simulator.step(u)

    assert simulator.get_state().shape == (2,), "State shape should remain consistent."


def test_state_update_with_measurement_noise():
    """Test if the output updates correctly when measurement noise is present."""
    system = StateSpaceModel(
        A=np.array([[1, 1], [0, 1]]), B=np.array([[0], [1]]), C=np.array([[1, 0], [0, 1]]), D=np.array([[0], [0]])
    )

    measurement_noise = WhiteNoiseGenerator(std=np.array([0.1, 0.1]))  # Nonzero measurement noise

    simulator = LTISimulator(system, x_0=np.array([0, 0]), measurement_noise=measurement_noise)

    u = np.array([1.0])
    y = simulator.get_output(u)

    assert y.shape == (2,), "Output shape should remain consistent even with noise."
