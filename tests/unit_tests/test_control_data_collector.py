"""
Unit tests for control input generation and trajectory data collection.
"""

import numpy as np
import pytest

from do_dpc.control_utils.control_structs import InputOutputTrajectory


@pytest.mark.parametrize("white_noise_generator", [{"m": 2}], indirect=True)
def test_generate_u(white_noise_generator):
    """Test that the control input generator produces values with correct shape."""
    u_generated = white_noise_generator.generate()
    assert u_generated.shape == (2,)


@pytest.mark.parametrize("trajectory_collector", [{"traj_length": 5}], indirect=True)
def test_store_and_retrieve_trajectory(trajectory_collector):
    """Test storing and retrieving trajectory data with enough data points."""
    y_sample = np.array([0.5, -0.2])
    u_sample = np.array([1.2])

    for _ in range(5):
        trajectory_collector.store_measurements(y_sample, u_sample)

    trajectory_data = trajectory_collector.get_trajectory_data()

    assert isinstance(trajectory_data, InputOutputTrajectory)
    assert trajectory_data.y.shape == (2, 5)
    assert trajectory_data.u.shape == (1, 5)


def test_incomplete_trajectory_reset(trajectory_collector):
    """Test that an incomplete trajectory replaces u and y nan values"""
    y_sample = np.array([0.5, -0.2])
    u_sample = np.array([1.2])

    trajectory_collector.store_measurements(y_sample, u_sample)

    traj_data = trajectory_collector.get_trajectory_data()

    # Check that NaN values have been replaced
    assert not np.isnan(traj_data.y).any(), "y trajectory still contains NaN values"
    assert not np.isnan(traj_data.u).any(), "u trajectory still contains NaN values"


def test_invalid_input_shape(trajectory_collector):
    """Test that storing an incorrectly shaped input raises a ValueError."""
    with pytest.raises(ValueError):
        trajectory_collector.store_measurements(np.array([1.0]), np.array([1.0, 2.0]))
