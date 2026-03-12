"""
pytest configuration for AutoResearch tests.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that require Kaggle/Anthropic API keys"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take more than 10 seconds"
    )
