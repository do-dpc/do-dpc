"""
Manages file paths for logs and data storage.

This module provides a `PathManager` class for handling file and directory paths,
ensuring required directories exist, and formatting filenames with timestamps.

Functions:
    - get_path_manager(): Returns a singleton instance of `PathManager`.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

VIDEO_FOLDER = "video"
EXAMPLE_FOLDER = "docs/source/example_gallery"


class PathManager:
    """
    Manages file paths and directories, ensuring they exist and formatting filenames.

    Responsibilities:
    - Provides paths for logs, data, and other files.
    - Ensures necessary folders exist.
    - Adds timestamps to filenames when needed.

    Attributes:
        logger (logging.Logger): Logger instance for logging information and errors.
        base_dir (Path): Base directory for logs and data.
        log_dir (Path): Directory path for log files.
        data_dir (Path): Directory path for data files.

    Args:
        base_dir (Path, optional): Base directory for logs and data. Defaults to the current working directory.

    Raises:
        ValueError: If `base_dir` is not a valid directory path.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initializes the PathManager.

        Args:
            base_dir (Path, optional): Base directory for logs and data. Defaults to the current working directory.
        """
        self.logger = logging.getLogger(__name__)

        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = base_dir

        self.log_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"

        # Ensure directories exist
        self._ensure_directory(self.log_dir)
        self._ensure_directory(self.data_dir)

    def _ensure_directory(self, path: Path):
        """
        Ensures a directory exists, creating it if necessary.

        Args:
            path (Path): The directory path that needs to be checked and created if missing.

        Raises:
            OSError: If the directory creation fails due to permissions or other system errors.
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Directory ensured: %s", path)
        except Exception as e:
            self.logger.error("Failed to create directory : %s", path)
            raise OSError(f"Could not create directory: {path}") from e

    def get_log_file(self, with_date: bool = True) -> Path:
        """
        Returns the full path to the log file.

        Args:
            with_date (bool, optional): If True, adds the current date in YYYY-MM-DD format to the filename.

        Returns:
            Path: Full path to the log file.
        """
        filename = "app.log"
        if with_date:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}_app.log"

        log_file = self.log_dir / filename
        self.logger.debug("Log file path retrieved: %s", log_file)
        return log_file

    def get_data_path(self, filename: str, with_date: bool = False) -> Path:
        """
        Returns the full path for a data file.

        Args:
            filename (str): Name of the data file.
            with_date (bool, optional): If True, prepends the date in YYYY-MM-DD format to the filename.

        Returns:
            Path: Full path to the data file.
        """
        if with_date:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}_{filename}"

        data_path = self.data_dir / filename
        self.logger.debug("Data file path retrieved: %s", data_path)
        return data_path

    def get_examples_path(self) -> Path:
        """
        Returns the full path to the example folder.

        Returns:
            Path: Full path to the example folder.
        """
        example_path = self.base_dir.joinpath(EXAMPLE_FOLDER)
        self._ensure_directory(example_path)
        return example_path

    def get_video_path(self) -> Path:
        """
        Returns the full path to the video folder.

        Returns:
            Path: Full path to the video folder.
        """

        return self.get_examples_path().joinpath(VIDEO_FOLDER)


# pylint: disable=invalid-name
# Singleton instance
_path_manager_instance = None


def get_path_manager() -> PathManager:
    """
    Returns a singleton instance of PathManager.

    Returns:
        PathManager: The global instance of PathManager.
    """
    # pylint: disable=global-statement
    global _path_manager_instance
    if _path_manager_instance is None:
        _path_manager_instance = PathManager()
    return _path_manager_instance
