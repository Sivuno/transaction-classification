"""
Tests for DynamoDB storage functionality.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from src.data.dynamodb_storage import DynamoDBStorage, DynamoDBWriter
from decimal import Decimal

class TestDynamoDBStorage:
    """Tests for the DynamoDBStorage class."""
    
    def test_init(self):
        """Test initialization."""
        with patch('boto3.resource') as mock_resource:
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table
            mock_table.meta.client.describe_table.return_value = {'Table': {'TableName': 'test-table'}}
            
            storage = DynamoDBStorage('test-table', 'us-east-1', 'test-key', 'test-secret')
            
            assert storage.table_name == 'test-table'
            assert storage.region_name == 'us-east-1'
            assert storage.aws_access_key_id == 'test-key'
            assert storage.aws_secret_access_key == 'test-secret'
            assert storage.table == mock_table
            
            mock_resource.assert_called_once_with(
                'dynamodb',
                region_name='us-east-1',
                aws_access_key_id='test-key',
                aws_secret_access_key='test-secret'
            )
    
    def test_create_table(self):
        """Test table creation."""
        with patch('boto3.resource') as mock_resource:
            # Mock the resource not finding the table
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table
            mock_table.meta.client.describe_table.side_effect = [
                {'ResponseMetadata': {'HTTPStatusCode': 404}, 'Error': {'Code': 'ResourceNotFoundException'}},
                {'Table': {'TableName': 'test-table'}}
            ]
            
            # Mock the create_table method
            mock_create_table = MagicMock()
            mock_resource.return_value.create_table.return_value = mock_create_table
            
            # Mock the waiter
            mock_waiter = MagicMock()
            mock_table.meta.client.get_waiter.return_value = mock_waiter
            
            # Create the storage
            storage = DynamoDBStorage('test-table')
            
            # Verify create_table was called
            mock_resource.return_value.create_table.assert_called_once()
            mock_waiter.wait.assert_called_once_with(TableName='test-table')
    
    def test_convert_to_dynamodb_format(self):
        """Test conversion to DynamoDB format."""
        # Test with various types
        test_item = {
            'string': 'test',
            'int': 123,
            'float': 123.45,
            'bool': True,
            'none': None,
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'nested': {'a': [1, 2], 'b': {'c': 3}}
        }
        
        result = DynamoDBStorage._convert_to_dynamodb_format(test_item)
        
        # Check types
        assert isinstance(result['string'], str)
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], Decimal)
        assert isinstance(result['bool'], bool)
        assert result['none'] is None
        assert isinstance(result['list'], list)
        assert isinstance(result['dict'], dict)
        assert isinstance(result['nested'], dict)
        assert isinstance(result['nested']['a'], list)
        assert isinstance(result['nested']['b'], dict)
        
        # Check values
        assert result['string'] == 'test'
        assert result['int'] == 123
        assert result['float'] == Decimal('123.45')
        assert result['bool'] is True
        assert result['none'] is None
        assert result['list'] == [1, 2, 3]
        assert result['dict'] == {'a': 1, 'b': 2}
        assert result['nested'] == {'a': [1, 2], 'b': {'c': 3}}
    
    @pytest.mark.asyncio
    async def test_write_item(self):
        """Test writing a single item."""
        with patch('boto3.resource') as mock_resource:
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table
            mock_table.meta.client.describe_table.return_value = {'Table': {'TableName': 'test-table'}}
            
            storage = DynamoDBStorage('test-table')
            
            # Test writing an item
            item = {'Transaction_ID': '123', 'Category_ID': 'ABC'}
            result = await storage.write_item(item)
            
            assert result is True
            mock_table.put_item.assert_called_once()
            
            # Check that ProcessTimestamp was added
            call_args = mock_table.put_item.call_args[1]['Item']
            assert 'ProcessTimestamp' in call_args
            assert 'Transaction_ID' in call_args
            assert 'Category_ID' in call_args
    
    @pytest.mark.asyncio
    async def test_batch_write_items(self):
        """Test batch writing items."""
        with patch('boto3.resource') as mock_resource:
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table
            mock_table.meta.client.describe_table.return_value = {'Table': {'TableName': 'test-table'}}
            
            # Mock batch writer
            mock_batch_writer = MagicMock()
            mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
            
            storage = DynamoDBStorage('test-table')
            
            # Test batch writing items
            items = [
                {'Transaction_ID': '123', 'Category_ID': 'ABC'},
                {'Transaction_ID': '456', 'Category_ID': 'DEF'}
            ]
            
            success_count, failed_items = await storage.batch_write_items(items)
            
            assert success_count == 2
            assert failed_items == []
            assert mock_batch_writer.put_item.call_count == 2
    
    @pytest.mark.asyncio
    async def test_query_items(self):
        """Test querying items."""
        with patch('boto3.resource') as mock_resource:
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table
            mock_table.meta.client.describe_table.return_value = {'Table': {'TableName': 'test-table'}}
            
            # Mock query response
            mock_table.query.return_value = {
                'Items': [
                    {'Transaction_ID': '123', 'Category_ID': 'ABC'},
                    {'Transaction_ID': '123', 'Category_ID': 'DEF'}
                ]
            }
            
            storage = DynamoDBStorage('test-table')
            
            # Test querying items
            items = await storage.query_items('123')
            
            assert len(items) == 2
            assert items[0]['Transaction_ID'] == '123'
            assert items[1]['Transaction_ID'] == '123'
            mock_table.query.assert_called_once()


class TestDynamoDBWriter:
    """Tests for the DynamoDBWriter class."""
    
    @pytest.mark.asyncio
    async def test_write_row(self):
        """Test writing a single row."""
        # Setup
        storage = MagicMock()
        writer = DynamoDBWriter(storage)
        writer.flush_buffer = MagicMock()
        
        # Execute - buffer not full yet
        await writer.write_row({'Transaction_ID': '123', 'Category_ID': 'ABC'})
        
        # Verify - buffer not flushed yet
        assert len(writer.buffer) == 1
        writer.flush_buffer.assert_not_called()
        
        # Fill the buffer to trigger flush
        writer.buffer_size = 2  # Set smaller buffer for testing
        await writer.write_row({'Transaction_ID': '456', 'Category_ID': 'DEF'})
        
        # Verify flush was called
        writer.flush_buffer.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_flush_buffer(self):
        """Test flushing the buffer."""
        # Setup
        storage = MagicMock()
        storage.batch_write_items.return_value = (2, [])
        writer = DynamoDBWriter(storage)
        
        # Add some rows
        writer.buffer = [
            {'Transaction_ID': '123', 'Category_ID': 'ABC'},
            {'Transaction_ID': '456', 'Category_ID': 'DEF'}
        ]
        
        # Execute
        await writer.flush_buffer()
        
        # Verify
        storage.batch_write_items.assert_called_once_with(writer.buffer)
        assert writer.buffer == []
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the writer."""
        # Setup
        storage = MagicMock()
        writer = DynamoDBWriter(storage)
        writer.flush_buffer = MagicMock()
        
        # Add a row to ensure flush is called
        writer.buffer = [{'Transaction_ID': '123', 'Category_ID': 'ABC'}]
        
        # Execute
        await writer.close()
        
        # Verify
        writer.flush_buffer.assert_called_once() 