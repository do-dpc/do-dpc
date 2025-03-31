"""
Unit tests for the ControllerParameters class.

This module tests the behavior of ControllerParameters, including
matrix validation, positive (semi)definiteness checks, and block-diagonal construction.
"""

import numpy as np
import pytest

from do_dpc.dpc.dpc_structs import DPCParameters


def create_positive_definite_matrix(size: int) -> np.ndarray:
    """Generates a random positive definite matrix of given size."""
    A = np.random.randn(size, size)
    return A.T @ A  # Ensures positive definiteness


def create_positive_semidefinite_matrix(size: int) -> np.ndarray:
    """Generates a random positive semidefinite matrix of given size."""
    A = np.random.randn(size, size)
    return A @ A.T  # Ensures positive semidefiniteness


@pytest.fixture
def valid_controller_params():
    """Fixture providing a valid instance of ControllerParameters."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)
    return DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)


def test_valid_initialization():
    """Test if ControllerParameters initializes correctly with valid inputs."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)
    params = DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)

    assert params.Q_horizon.shape == (9, 9), "Q_horizon shape should be (3*tau_f, 3*tau_f)."
    assert params.R_horizon.shape == (6, 6), "R_horizon shape should be (2*tau_f, 2*tau_f)."


@pytest.mark.parametrize("tau_p, tau_f", [(-1, 3), (3, 0), (0, 0)])
def test_invalid_horizon_values(tau_p, tau_f):
    """Test if negative or zero horizon values raise ValueError."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)
    with pytest.raises(ValueError, match="must be a positive integer"):
        DPCParameters(Q=Q, R=R, tau_p=tau_p, tau_f=tau_f)


def test_invalid_Q_not_square():
    """Test if a non-square Q matrix raises ValueError."""
    Q = np.random.randn(3, 2)  # Not square
    R = create_positive_definite_matrix(2)
    with pytest.raises(ValueError, match="must be a square matrix"):
        DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)


def test_invalid_R_not_square():
    """Test if a non-square R matrix raises ValueError."""
    Q = create_positive_semidefinite_matrix(3)
    R = np.random.randn(2, 3)  # Not square
    with pytest.raises(ValueError, match="must be a square matrix"):
        DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)


def test_Q_not_positive_semidefinite():
    """Test if a Q matrix that is not positive semidefinite raises ValueError."""
    Q = np.array([[1, 2], [2, -3]])  # Not positive semidefinite
    R = create_positive_definite_matrix(2)
    with pytest.raises(ValueError, match="must be positive semidefinite"):
        DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)


def test_R_not_positive_definite():
    """Test if an R matrix that is not positive definite raises ValueError."""
    Q = create_positive_semidefinite_matrix(3)
    R = np.array([[0, 0], [0, -1]])  # Not positive definite
    with pytest.raises(ValueError, match="must be positive definite"):
        DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)


def test_Q_final_overwrites_last_block():
    """Test if Q_final correctly overwrites the last block of Q_horizon."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)
    Q_final = create_positive_semidefinite_matrix(3)

    params = DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3, Q_final=Q_final)

    assert np.array_equal(params.Q_horizon[-3:, -3:], Q_final), "Q_final should replace the last block in Q_horizon."


def test_R_final_overwrites_last_block():
    """Test if R_final correctly overwrites the last block of R_horizon."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)
    R_final = create_positive_definite_matrix(2)

    params = DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3, R_final=R_final)

    assert np.array_equal(params.R_horizon[-2:, -2:], R_final), "R_final should replace the last block in R_horizon."


def test_R_delta_horizon_construction():
    """Test if R_delta correctly constructs a block-diagonal matrix."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)
    R_delta = create_positive_definite_matrix(2)

    params = DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3, R_delta=R_delta)

    expected_shape = ((3 - 1) * 2, (3 - 1) * 2)  # (tau_f - 1) * R.shape[0]
    assert params.R_delta_horizon.shape == expected_shape, "R_delta_horizon shape mismatch."


def test_default_R_delta_horizon():
    """Test if R_delta_horizon is None when R_delta is not provided."""
    Q = create_positive_semidefinite_matrix(3)
    R = create_positive_definite_matrix(2)

    params = DPCParameters(Q=Q, R=R, tau_p=2, tau_f=3)

    assert params.R_delta_horizon is None, "R_delta_horizon should be None if R_delta is not provided."
