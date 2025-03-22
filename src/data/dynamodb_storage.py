import logging
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class DynamoDBStorage:
    """
    Storage handler for DynamoDB.
    """
    def __init__(self, table_name, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        """
        Initialize DynamoDB storage handler.
        
        Args:
            table_name: DynamoDB table name
            region_name: AWS region name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.table_name = table_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.dynamodb = None
        self.table = None
        self._initialize_dynamodb()
        logger.debug(f"Initialized DynamoDBStorage with table={table_name}")
        
    def _initialize_dynamodb(self):
        """Initialize DynamoDB resource and table."""
        try:
            # Create DynamoDB resource
            self.dynamodb = boto3.resource(
                'dynamodb',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            
            # Get table reference
            self.table = self.dynamodb.Table(self.table_name)
            
            # Verify table exists by making a simple query
            self.table.meta.client.describe_table(TableName=self.table_name)
            logger.info(f"Successfully connected to DynamoDB table {self.table_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.error(f"Table {self.table_name} does not exist. Creating table...")
                self._create_table()
            else:
                logger.error(f"Error initializing DynamoDB: {e}")
                raise
    
    def _create_table(self):
        """Create DynamoDB table if it doesn't exist."""
        try:
            # Create table with Transaction_ID as partition key and a timestamp sort key
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {'AttributeName': 'Transaction_ID', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'ProcessTimestamp', 'KeyType': 'RANGE'}  # Sort key
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'Transaction_ID', 'AttributeType': 'S'},
                    {'AttributeName': 'ProcessTimestamp', 'AttributeType': 'N'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            
            # Wait for table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
            self.table = table
            logger.info(f"Created DynamoDB table {self.table_name}")
        except ClientError as e:
            logger.error(f"Error creating DynamoDB table: {e}")
            raise
    
    @staticmethod
    def _convert_to_dynamodb_format(item):
        """
        Convert Python types to DynamoDB compatible format.
        
        Args:
            item: Dictionary to convert
            
        Returns:
            Dictionary with DynamoDB compatible types
        """
        if isinstance(item, dict):
            return {k: DynamoDBStorage._convert_to_dynamodb_format(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [DynamoDBStorage._convert_to_dynamodb_format(i) for i in item]
        elif isinstance(item, float):
            return Decimal(str(item))
        elif isinstance(item, (int, str, bool, type(None))):
            return item
        else:
            return str(item)
    
    async def write_item(self, item):
        """
        Write a single item to DynamoDB.
        
        Args:
            item: Dictionary containing the item data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add timestamp if not present
            if 'ProcessTimestamp' not in item:
                item['ProcessTimestamp'] = int(time.time())
            
            # Convert to DynamoDB format
            dynamodb_item = self._convert_to_dynamodb_format(item)
            
            # Write to DynamoDB
            self.table.put_item(Item=dynamodb_item)
            return True
        except Exception as e:
            logger.error(f"Error writing item to DynamoDB: {e}")
            return False
    
    async def batch_write_items(self, items):
        """
        Write multiple items to DynamoDB in batches.
        
        Args:
            items: List of dictionaries containing item data
            
        Returns:
            tuple: (success_count, failed_items)
        """
        if not items:
            return 0, []
        
        success_count = 0
        failed_items = []
        
        # Process in batches of 25 (DynamoDB batch write limit)
        for i in range(0, len(items), 25):
            batch = items[i:i+25]
            
            # Add timestamp if not present
            for item in batch:
                if 'ProcessTimestamp' not in item:
                    item['ProcessTimestamp'] = int(time.time())
            
            # Convert to DynamoDB format
            dynamodb_batch = [self._convert_to_dynamodb_format(item) for item in batch]
            
            try:
                # Prepare batch write request
                with self.table.batch_writer() as writer:
                    for item in dynamodb_batch:
                        writer.put_item(Item=item)
                
                success_count += len(batch)
            except Exception as e:
                logger.error(f"Error in batch write to DynamoDB: {e}")
                failed_items.extend(batch)
        
        return success_count, failed_items
    
    async def query_items(self, transaction_id=None, limit=100):
        """
        Query items from DynamoDB.
        
        Args:
            transaction_id: Optional transaction ID to filter by
            limit: Maximum number of items to return
            
        Returns:
            List of items matching the query
        """
        try:
            if transaction_id:
                response = self.table.query(
                    KeyConditionExpression=Key('Transaction_ID').eq(transaction_id),
                    Limit=limit
                )
            else:
                response = self.table.scan(Limit=limit)
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error querying DynamoDB: {e}")
            return []
    
    async def delete_item(self, transaction_id, process_timestamp):
        """
        Delete an item from DynamoDB.
        
        Args:
            transaction_id: Transaction ID
            process_timestamp: Process timestamp
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.table.delete_item(
                Key={
                    'Transaction_ID': transaction_id,
                    'ProcessTimestamp': process_timestamp
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting item from DynamoDB: {e}")
            return False


class DynamoDBWriter:
    """
    Writer that buffers rows and writes them to DynamoDB in batches.
    """
    def __init__(self, dynamodb_storage, batch_size=25):
        """
        Initialize DynamoDB writer.
        
        Args:
            dynamodb_storage: Instance of DynamoDBStorage
            batch_size: Number of items to buffer before writing
        """
        logger.debug(f"Initializing DynamoDBWriter with batch_size={batch_size}")
        self.dynamodb_storage = dynamodb_storage
        self.buffer = []
        self.buffer_size = batch_size
        self.lock = asyncio.Lock()
        self.total_rows = 0
        self.max_retries = 3
        self.base_delay = 1  # Base delay in seconds for exponential backoff

    async def write_row(self, row_data):
        """
        Write a single row to the DynamoDB buffer with retry logic.
        
        Args:
            row_data: Dictionary containing the row data to write
        """
        logger.debug(f"Adding row to buffer (current size: {len(self.buffer)})")
        
        for attempt in range(self.max_retries):
            try:
                async with self.lock:
                    self.buffer.append(row_data)
                    self.total_rows += 1
                    
                    # If buffer is full, write to DynamoDB
                    if len(self.buffer) >= self.buffer_size:
                        await self.flush_buffer()
                    return
                    
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Error writing row to buffer (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Failed to write row after {self.max_retries} attempts: {e}")
                raise

    async def flush_buffer(self):
        """Write buffered rows to DynamoDB with retry logic."""
        if not self.buffer:
            return
            
        logger.debug(f"Flushing buffer with {len(self.buffer)} rows")
        
        for attempt in range(self.max_retries):
            try:
                success_count, failed_items = await self.dynamodb_storage.batch_write_items(self.buffer)
                
                if failed_items:
                    logger.warning(f"Failed to write {len(failed_items)} items to DynamoDB")
                    # Keep failed items in buffer for retry
                    self.buffer = failed_items
                else:
                    self.buffer = []
                
                logger.debug(f"Buffer flushed successfully: {success_count} items written")
                return
                
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Error flushing buffer (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying flush in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Failed to flush buffer after {self.max_retries} attempts: {e}")
                raise

    async def close(self):
        """Ensure any remaining buffered rows are written and clean up resources."""
        logger.debug(f"Closing DynamoDB writer, flushing remaining {len(self.buffer)} rows")
        if self.buffer:
            await self.flush_buffer()
        logger.info(f"DynamoDB writer closed. Total rows written: {self.total_rows}")
        
    async def __aenter__(self):
        """Support for async context manager."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup when used as context manager."""
        await self.close()

    async def get_total_rows(self):
        """Return the total number of rows written so far."""
        return self.total_rows 