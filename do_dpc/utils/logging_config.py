"""
Configures application-wide logging with colored console output and file logging.
"""

import logging

from do_dpc.utils.path_manager import get_path_manager

# Get a singleton instance of PathManager
path_manager = get_path_manager()


class SimpleColoredFormatter(logging.Formatter):
    """
    Custom log formatter with basic color coding for different log levels.
    ANSI escape codes are used for terminal output only.
    """

    COLOR_CODES = {
        logging.DEBUG: "\033[90m",  # Gray
        logging.INFO: "\033[34m",  # Blue
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bright Red
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, self.COLOR_CODES["RESET"])
        message = super().format(record)
        return f"{color}{message}{self.COLOR_CODES['RESET']}"


def setup_logging():
    """
    Configures logging using the log file path from PathManager.
    Uses colored logs for console output and plain text for file logs.
    """
    log_file = path_manager.get_log_file(with_date=True)

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    # Create formatter for file logging (plain text)
    file_formatter = logging.Formatter(log_format)

    # Create formatter for console logging (with colors)
    colored_formatter = SimpleColoredFormatter(log_format)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
    )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger for a specific module.

    Args:
        name (str): The name of the logger (typically `__name__`).

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


# Automatically set up logging when this module is imported
setup_logging()
