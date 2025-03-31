"""
Unit tests for logging configuration.
"""

import logging

from do_dpc.utils.logging_config import get_logger, setup_logging


def test_get_logger():
    """Test that get_logger returns a valid logger instance."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_logging_setup(caplog):
    """Test if logging setup correctly initializes log handlers."""
    setup_logging()
    logger = get_logger("test_logger")
    logger.info("Test log message")

    assert any("Test log message" in record.message for record in caplog.records)
