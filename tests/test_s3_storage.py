"""
Tests for S3 storage functionality.
"""
import io
import csv
import pytest
from unittest.mock import MagicMock, patch, call
from src.data.s3_storage import S3TempStorage, IncrementalCSVWriter

class TestS3TempStorage:
    """Tests for the S3TempStorage class."""
    
    def test_init(self):
        """Test initialization."""
        s3_client = MagicMock()
        storage = S3TempStorage(s3_client, 'test-bucket', 'test-prefix')
        
        assert storage.s3_client == s3_client
        assert storage.bucket == 'test-bucket'
        assert storage.prefix == 'test-prefix/'
    
    @pytest.mark.asyncio
    async def test_write_csv_rows_new_file(self):
        """Test writing rows to a new CSV file."""
        # Setup
        s3_client = MagicMock()
        storage = S3TempStorage(s3_client, 'test-bucket', 'test-prefix')
        
        # Mock NoSuchKey exception for a new file
        s3_client.get_object.side_effect = s3_client.exceptions.NoSuchKey({}, 'NoSuchKey')
        
        # Test data
        fieldnames = ['id', 'name']
        rows = [{'id': 1, 'name': 'Test1'}, {'id': 2, 'name': 'Test2'}]
        
        # Execute
        await storage.write_csv_rows('test.csv', fieldnames, rows, mode='a')
        
        # Verify
        s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-prefix/test.csv'
        )
        s3_client.put_object.assert_called_once()
        args = s3_client.put_object.call_args[1]
        assert args['Bucket'] == 'test-bucket'
        assert args['Key'] == 'test-prefix/test.csv'
        
        # Verify CSV content
        content = args['Body'].decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        result_rows = list(reader)
        assert len(result_rows) == 2
        assert result_rows[0]['id'] == '1'
        assert result_rows[0]['name'] == 'Test1'
    
    @pytest.mark.asyncio
    async def test_read_csv(self):
        """Test reading a CSV file."""
        # Setup
        s3_client = MagicMock()
        storage = S3TempStorage(s3_client, 'test-bucket', 'test-prefix')
        
        # Mock response
        mock_body = io.StringIO("id,name\n1,Test1\n2,Test2\n")
        mock_response = {'Body': MagicMock()}
        mock_response['Body'].read.return_value = mock_body.getvalue().encode('utf-8')
        s3_client.get_object.return_value = mock_response
        
        # Execute
        rows = await storage.read_csv('test.csv')
        
        # Verify
        s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-prefix/test.csv'
        )
        assert len(rows) == 2
        assert rows[0]['id'] == '1'
        assert rows[0]['name'] == 'Test1'
    
    def test_delete_file(self):
        """Test deleting a file."""
        # Setup
        s3_client = MagicMock()
        storage = S3TempStorage(s3_client, 'test-bucket', 'test-prefix')
        
        # Execute
        storage.delete_file('test.csv')
        
        # Verify
        s3_client.delete_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-prefix/test.csv'
        )
    
    @pytest.mark.asyncio
    async def test_combine_csv_files(self):
        """Test combining CSV files."""
        # Setup
        s3_client = MagicMock()
        storage = S3TempStorage(s3_client, 'test-bucket', 'test-prefix')
        
        # Mock reading files
        async def mock_read_csv(key):
            if key == 'file1.csv':
                return [{'id': '1', 'name': 'Test1'}]
            else:
                return [{'id': '2', 'name': 'Test2'}]
        
        storage.read_csv = mock_read_csv
        
        # Execute
        result = await storage.combine_csv_files(
            ['file1.csv', 'file2.csv'],
            'combined.csv',
            ['id', 'name']
        )
        
        # Verify
        assert result is True
        s3_client.put_object.assert_called_once()
        args = s3_client.put_object.call_args[1]
        assert args['Bucket'] == 'test-bucket'
        assert args['Key'] == 'test-prefix/combined.csv'
        
        # Check content
        content = args['Body'].decode('utf-8')
        assert 'id,name' in content
        assert '1,Test1' in content.replace('\r', '')
        assert '2,Test2' in content.replace('\r', '')

class TestIncrementalCSVWriter:
    """Tests for the IncrementalCSVWriter class."""
    
    @pytest.mark.asyncio
    async def test_write_row(self):
        """Test writing a single row."""
        # Setup
        s3_storage = MagicMock()
        writer = IncrementalCSVWriter(s3_storage, 'test.csv', ['id', 'name'])
        writer.flush_buffer = MagicMock()
        
        # Execute - buffer not full yet
        await writer.write_row({'id': 1, 'name': 'Test1'})
        
        # Verify - buffer not flushed yet
        assert len(writer.buffer) == 1
        writer.flush_buffer.assert_not_called()
        
        # Fill the buffer to trigger flush
        writer.buffer_size = 2  # Set smaller buffer for testing
        await writer.write_row({'id': 2, 'name': 'Test2'})
        
        # Verify flush was called
        writer.flush_buffer.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_flush_buffer(self):
        """Test flushing the buffer."""
        # Setup
        s3_storage = MagicMock()
        writer = IncrementalCSVWriter(s3_storage, 'test.csv', ['id', 'name'])
        
        # Add some rows
        writer.buffer = [
            {'id': '1', 'name': 'Test1'},
            {'id': '2', 'name': 'Test2'}
        ]
        
        # Execute
        await writer.flush_buffer()
        
        # Verify
        s3_storage.write_csv_rows.assert_called_once_with(
            'test.csv',
            ['id', 'name'],
            writer.buffer,
            mode='a'
        )
        assert writer.buffer == []
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the writer."""
        # Setup
        s3_storage = MagicMock()
        writer = IncrementalCSVWriter(s3_storage, 'test.csv', ['id', 'name'])
        writer.flush_buffer = MagicMock()
        
        # Add a row to ensure flush is called
        writer.buffer = [{'id': '1', 'name': 'Test1'}]
        
        # Execute
        await writer.close()
        
        # Verify
        writer.flush_buffer.assert_called_once()