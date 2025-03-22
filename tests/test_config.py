"""
Tests for configuration module.
"""
import os
import pytest
from unittest.mock import patch
from src.config.config import validate_credentials, setup_logging

class TestConfig:
    """Tests for the config module."""
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key", 
        "JINA_API_KEY": "test-key",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-key",
        "REGION": "us-east-1",
        "RDS_USERNAME": "test-user",
        "RDS_PASSWORD": "test-pass"
    })
    def test_validate_credentials_success(self):
        """Test validation with all credentials present."""
        with patch('src.config.config.load_dotenv'):
            result = validate_credentials()
            assert result is None
    
    def test_validate_credentials_missing(self):
        """Test validation with missing credentials."""
        # Directly patch the module-level variables instead of os.environ
        with patch('src.config.config.OPENAI_API_KEY', ''), \
             patch('src.config.config.JINA_API_KEY', ''), \
             patch('src.config.config.AWS_ACCESS_KEY_ID', 'test-key'), \
             patch('src.config.config.AWS_SECRET_ACCESS_KEY', 'test-key'), \
             patch('src.config.config.REGION', 'us-east-1'), \
             patch('src.config.config.RDS_USERNAME', 'test-user'), \
             patch('src.config.config.RDS_PASSWORD', 'test-pass'), \
             patch('src.config.config.load_dotenv'):
            
            result = validate_credentials()
            assert result is not None
            assert 'OPENAI_API_KEY' in result
            assert 'JINA_API_KEY' in result
    
    def test_setup_logging(self):
        """Test logging setup."""
        with patch('src.config.config.logging.basicConfig') as mock_config:
            logger = setup_logging(level=10)  # DEBUG level
            mock_config.assert_called_once()
            assert logger is not None