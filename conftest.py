"""
Global fixture configuration.

This automatically detects and imports all fixture files inside `tests/fixtures/`.
"""

pytest_plugins = ["tests.fixtures.utils_fixtures"]
