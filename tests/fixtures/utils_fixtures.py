"""
Global fixtures for tests.

Pytest automatically discovers fixtures from this module.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from do_ddpc.control_utils.lti_systems import create_1D_double_integrator
from do_ddpc.control_utils.noise_generators import WhiteNoiseGenerator
from do_ddpc.control_utils.trajectory_collector import TrajectoryCollector
from do_ddpc.utils.path_manager import PathManager


@pytest.fixture
def path_manager(tmp_path):
    """
    Provides a temporary PathManager instance.

    Uses `tmp_path` to prevent modifications to actual directories during tests.

    Returns:
        PathManager: A PathManager instance with a temporary base directory.
    """
    return PathManager(base_dir=tmp_path)


@pytest.fixture
def double_integrator():
    """
    Returns an LTISimulator object with the matrices of a double integrator.
    """
    return create_1D_double_integrator()


@pytest.fixture
def white_noise_generator(request):
    """
    Fixture for white noise input generation.

    Parameters:
        m (int, default=1): Number of control inputs.
        u_mean (list, default=[0.0] * m): Mean value for the control input.
        exc_dev (float, default=0.1): Standard deviation of the white noise applied.

    Returns:
        ControlInputGenerator: An instance configured with the given parameters.
    """
    params = getattr(request, "param", {})

    m = params.get("m", 1)
    u_mean = np.array(params.get("u_mean", [0.0] * m))
    exc_dev = params.get("exc_dev", 0.1)

    return WhiteNoiseGenerator(mean=u_mean, std=exc_dev * np.ones_like(u_mean))


@pytest.fixture
def trajectory_collector(request):
    """
    Fixture for trajectory data collection.

    Parameters:
        m (int, default=1): Number of control inputs.
        p (int, default=2): Number of system outputs.
        traj_length (int, default=10): Length of the trajectory (number of time steps).

    Returns:
        TrajectoryCollector: An instance configured with the given parameters.
    """
    params = getattr(request, "param", {})

    m = params.get("m", 1)
    p = params.get("p", 2)
    traj_length = params.get("traj_length", 10)

    return TrajectoryCollector(m, p, traj_length)


# Define a simple dataclass for testing
@dataclass
class ExampleData:
    """Stores system trajectory data."""

    y: np.ndarray
    u: np.ndarray
    time_step: float
    iterations: int


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    return ExampleData(y=np.array([[1, 2, 3], [4, 5, 6]]), u=np.array([[0.1, 0.2, 0.3]]), time_step=0.1, iterations=100)
