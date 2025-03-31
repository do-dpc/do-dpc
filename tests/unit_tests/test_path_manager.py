"""
Unit tests for the PathManager class.

Tests ensure that log and data paths are correctly formatted
and that necessary directories are created.
"""

from do_ddpc.utils.path_manager import get_path_manager


def test_directories_created(path_manager):
    """Test that log and data directories are created."""
    assert path_manager.log_dir.exists()
    assert path_manager.data_dir.exists()


def test_get_log_file(path_manager):
    """Test log file path generation."""
    log_file = path_manager.get_log_file()
    assert path_manager.log_dir in log_file.parents
    assert log_file.suffix == ".log"


def test_get_data_path(path_manager):
    """Test data file path generation."""
    filename = "test_data.csv"
    data_file = path_manager.get_data_path(filename)
    assert path_manager.data_dir in data_file.parents
    assert data_file.name == filename


def test_singleton_instance():
    """Test that get_path_manager() returns the same instance."""
    instance1 = get_path_manager()
    instance2 = get_path_manager()
    assert instance1 is instance2
