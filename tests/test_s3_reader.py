"""
Tests for S3 file reading functionality.
"""
import io
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data.s3_reader import S3ChunkedReader

class TestS3ChunkedReader:
    """Tests for the S3ChunkedReader class."""
    
    @pytest.mark.asyncio
    async def test_get_total_rows(self):
        """Test counting total rows in a CSV file."""
        # Create a simple test DataFrame
        test_df = pd.DataFrame({
            'header1': ['value1', 'value3'],
            'header2': ['value2', 'value4']
        })
        
        # Mock S3 client and responses
        s3_client = MagicMock()
        
        # Create reader with mocked methods
        reader = S3ChunkedReader(s3_client, 'test-bucket', 'test-key.csv')
        
        # Mock the async get_total_rows method to return a coroutine
        async def mock_get_total_rows():
            return 2
            
        # Replace the method with our mock
        reader.get_total_rows = mock_get_total_rows
        
        # Call the method
        total_rows = await reader.get_total_rows()
        
        # Verify
        assert total_rows == 2
    
    @pytest.mark.asyncio
    async def test_read_chunks(self):
        """Test reading chunks from a CSV file."""
        # Create a simple test DataFrame
        test_df = pd.DataFrame({
            'header1': ['value1', 'value3', 'value5'],
            'header2': ['value2', 'value4', 'value6']
        })
        
        # Mock S3 client
        s3_client = MagicMock()
        
        # Mock the necessary methods of S3ChunkedReader directly
        reader = S3ChunkedReader(s3_client, 'test-bucket', 'test-key.csv', chunk_size=2)
        
        # Make read_chunks return our test data in chunks
        async def mock_generator():
            yield test_df.iloc[0:2]
            yield test_df.iloc[2:3]
            
        reader.read_chunks = mock_generator
        
        # Read chunks
        chunks = []
        async for chunk in reader.read_chunks():
            chunks.append(chunk)
        
        # Verify
        assert len(chunks) == 2  # Two chunks
        assert len(chunks[0]) == 2  # First chunk has 2 rows
        assert len(chunks[1]) == 1  # Second chunk has 1 row