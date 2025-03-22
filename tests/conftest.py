"""
Common test fixtures for pytest.
"""
import os
import pytest
import boto3
import asyncio
from unittest.mock import MagicMock, patch

@pytest.fixture
def s3_client_mock():
    """Mock S3 client for testing."""
    with patch('boto3.client') as mock_client:
        yield mock_client

# Note: We removed the event_loop fixture to avoid warnings
# pytest-asyncio provides its own event_loop fixture