"""Pytest configuration for atdata tests."""

import warnings
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest to suppress known warnings from test infrastructure.

    Suppresses RuntimeWarnings from s3fs/moto async incompatibility that occur
    during test cleanup and coverage instrumentation. These are expected when
    mocking S3 operations and don't indicate real issues.
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", pytest.PytestUnraisableExceptionWarning)
