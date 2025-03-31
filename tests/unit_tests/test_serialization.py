"""
Unit tests for the serialization functionality.

This module tests the functionality of saving and loading dataclass objects
in `.npz` format, including handling supported and unsupported field types,
missing fields, 0D arrays, and multiple datasets.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from do_ddpc.utils.serialization import save_dataclass_npz, load_dataclass_npz
from tests.fixtures.utils_fixtures import ExampleData


def test_save_and_load_npz(tmpdir, sample_data):
    """Test saving and loading dataclass using NPZ format."""
    # Create a temporary file in the tmpdir
    filename = tmpdir.join("test_trajectory.npz")

    # Save the dataclass object
    save_dataclass_npz(sample_data, str(filename))

    # Load the dataclass object
    loaded_data = load_dataclass_npz(ExampleData, str(filename))

    # Assertions to check if the loaded data matches the saved data
    assert np.array_equal(loaded_data.y, sample_data.y)
    assert np.array_equal(loaded_data.u, sample_data.u)
    assert loaded_data.time_step == sample_data.time_step
    assert loaded_data.iterations == sample_data.iterations


def test_invalid_field_type(tmpdir):
    """Test that an error is raised when a non-supported field type is saved."""

    # Define a dataclass with an unsupported type (e.g., a string)
    @dataclass
    class InvalidData:
        """Invalid Dataclass"""

        x: np.ndarray
        y: str  # Invalid field type

    invalid_data = InvalidData(x=np.array([1, 2]), y="invalid")

    # Create a temporary file in the tmpdir
    filename = tmpdir.join("invalid_data.npz")

    # Try saving it, expect a ValueError
    with pytest.raises(ValueError, match="Unsupported data type <class 'str'> for field 'y'"):
        save_dataclass_npz(invalid_data, str(filename))


def test_load_non_existent_file(tmpdir):
    """Test that FileNotFoundError is raised when trying to load a non-existent file."""
    filename = tmpdir.join("non_existent_file.npz")

    # Try loading a non-existent file, expect a FileNotFoundError
    with pytest.raises(FileNotFoundError, match=f"File '{filename}' not found."):
        load_dataclass_npz(ExampleData, str(filename))


def test_load_missing_field(tmpdir):
    """Test that loading data without all fields raises an error."""
    # Create a partial npz file (missing some fields)
    filename = tmpdir.join("partial_trajectory.npz")
    partial_data = {
        "y": np.array([[1, 2], [3, 4]]),
        "u": np.array([[0.1, 0.2]]),
    }
    np.savez_compressed(str(filename), **partial_data)

    # Try loading data with missing fields (e.g., 'time_step' and 'iterations')
    with pytest.raises(TypeError):
        load_dataclass_npz(ExampleData, str(filename))


def test_convert_0d_array_to_scalar(tmpdir):
    """Test that a 0D NumPy array is correctly converted to a scalar when loading."""

    @dataclass
    class ScalarData:
        """Scalar dataclass"""

        value: np.ndarray

    # Save a 0D NumPy array
    filename = tmpdir.join("scalar_data.npz")
    scalar_data = ScalarData(value=np.array(42.0))  # 0D array
    save_dataclass_npz(scalar_data, str(filename))

    # Load and ensure that it's converted to a scalar
    loaded_scalar_data = load_dataclass_npz(ScalarData, str(filename))
    assert loaded_scalar_data.value == 42.0  # It should be a float, not a numpy array


def test_save_and_load_multiple_datasets(tmpdir, sample_data):
    """Test saving and loading multiple different dataclass instances."""
    # Create temporary file paths
    filename1 = tmpdir.join("test_trajectory1.npz")
    filename2 = tmpdir.join("test_trajectory2.npz")

    # Save the first dataclass object
    save_dataclass_npz(sample_data, str(filename1))

    # Modify data
    sample_data.y = np.array([[7, 8, 9], [10, 11, 12]])
    sample_data.u = np.array([[0.4, 0.5, 0.6]])

    # Save the second dataclass object
    save_dataclass_npz(sample_data, str(filename2))

    # Load both datasets and check if they match their respective original versions
    loaded_data1 = load_dataclass_npz(ExampleData, str(filename1))
    loaded_data2 = load_dataclass_npz(ExampleData, str(filename2))

    assert not np.array_equal(loaded_data1.y, loaded_data2.y)
    assert not np.array_equal(loaded_data1.u, loaded_data2.u)
    assert loaded_data1.time_step == sample_data.time_step
    assert loaded_data2.time_step == sample_data.time_step
