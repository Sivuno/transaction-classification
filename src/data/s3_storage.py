import io
import csv
import logging
import asyncio
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class S3TempStorage:
    """
    Storage handler for temporary files in S3.
    """
    def __init__(self, s3_client, bucket, prefix):
        """
        Initialize S3 temporary storage handler.
        
        Args:
            s3_client: Boto3 S3 client instance
            bucket: S3 bucket name
            prefix: Prefix for all temporary files (e.g., 'processing/')
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'
        logger.debug(f"Initialized S3TempStorage with bucket={bucket}, prefix={prefix}")
        
    async def write_csv_rows(self, key, fieldnames, rows, mode='a'):
        """
        Write rows to a CSV file in S3.
        
        Args:
            key: S3 key for the file
            fieldnames: List of CSV column names
            rows: List of dictionaries containing row data
            mode: 'w' for write (overwrite), 'a' for append
        """
        logger.debug(f"Writing {len(rows)} rows to {key} in mode {mode}")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        
        if mode == 'w':
            writer.writeheader()
        
        for row in rows:
            writer.writerow(row)
            
        csv_bytes = buffer.getvalue().encode('utf-8')
        
        if mode == 'a':
            try:
                existing_content = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=f"{self.prefix}{key}"
                )['Body'].read()
                
                if not existing_content.endswith(b'\n'):
                    csv_bytes = b'\n' + csv_bytes
                    
                csv_bytes = existing_content + csv_bytes
            except self.s3_client.exceptions.NoSuchKey:
                header_buffer = io.StringIO()
                writer = csv.DictWriter(header_buffer, fieldnames=fieldnames)
                writer.writeheader()
                csv_bytes = header_buffer.getvalue().encode('utf-8') + csv_bytes
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{key}",
                Body=csv_bytes
            )
            logger.debug(f"Successfully wrote data to {key}")
        except Exception as e:
            logger.error(f"Failed to write to S3: {e}")
            raise
        
    async def read_csv(self, key):
        """
        Read CSV content from S3.
        
        Args:
            key: S3 key of the file to read
            
        Returns:
            List of dictionaries containing the CSV data
        """
        logger.debug(f"Reading CSV from {key}")
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{key}"
            )
            content = response['Body'].read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            return list(reader)
        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            raise
        
    def delete_file(self, key):
        """
        Delete a file from S3.
        
        Args:
            key: S3 key of the file to delete
        """
        logger.debug(f"Deleting file {key}")
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{key}"
            )
        except Exception as e:
            logger.error(f"Failed to delete file from S3: {e}")
            raise
        
    def list_files(self, prefix=None):
        """
        List files in the temp storage area.
        
        Args:
            prefix: Optional additional prefix to filter results
            
        Returns:
            List of S3 keys for matching files
        """
        full_prefix = f"{self.prefix}{prefix if prefix else ''}"
        logger.debug(f"Listing files with prefix {full_prefix}")
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            files = []
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get('Contents', []):
                    files.append(obj['Key'])
                    
            return files
        except Exception as e:
            logger.error(f"Failed to list files in S3: {e}")
            raise
        
    async def combine_csv_files(self, file_keys, output_key, fieldnames):
        """
        Combine multiple CSV files into one with robust error handling.
        
        Args:
            file_keys: List of S3 keys to combine
            output_key: S3 key for the combined output file
            fieldnames: List of CSV column names
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug(f"Combining {len(file_keys)} files into {output_key}")
        
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        
        total_rows = 0
        files_processed = 0
        try:
            for key in file_keys:
                try:
                    logger.debug(f"Reading file {key} for combination")
                    rows = await self.read_csv(key)
                    files_processed += 1
                    for row in rows:
                        writer.writerow(row)
                        total_rows += 1
                    logger.debug(f"Added {len(rows)} rows from {key}")
                except Exception as e:
                    logger.warning(f"Error reading file {key} for combination, skipping: {e}")
                    continue
            
            if files_processed == 0:
                logger.warning(f"No files were successfully processed during combination")
                return False
                
            csv_bytes = buffer.getvalue().encode('utf-8')
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{output_key}",
                Body=csv_bytes
            )
            logger.info(f"Successfully combined {total_rows} rows from {files_processed} files into {output_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to combine CSV files: {e}")
            raise
        
    def cleanup_temp_files(self, prefix=None):
        """
        Delete all temporary files with given prefix.
        
        Args:
            prefix: Optional prefix to filter files for deletion
        """
        logger.debug(f"Cleaning up temp files with prefix {prefix}")
        try:
            files = self.list_files(prefix)
            logger.info(f"Found {len(files)} files to clean up")
            
            for i in range(0, len(files), 1000):
                batch = files[i:i + 1000]
                objects = [{'Key': key} for key in batch]
                
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects}
                )
            logger.debug("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            raise

class IncrementalCSVWriter:
    """
    CSV writer that buffers rows and writes them to S3 in batches.
    """
    def __init__(self, s3_storage, output_key, fieldnames):
        """
        Initialize CSV writer for S3 storage.
        
        Args:
            s3_storage: Instance of S3TempStorage for file operations
            output_key: Key (path) for the output file in S3
            fieldnames: List of column names for the CSV
        """
        logger.debug(f"Initializing IncrementalCSVWriter for S3 key {output_key}")
        self.s3_storage = s3_storage
        self.output_key = output_key
        self.fieldnames = fieldnames
        self.buffer = []
        self.buffer_size = 100  # Number of rows to buffer before writing
        self.lock = asyncio.Lock()
        self.total_rows = 0
        self.max_retries = 3
        self.base_delay = 1  # Base delay in seconds for exponential backoff

    async def write_row(self, row_data):
        """
        Write a single row to the CSV buffer with retry logic.
        
        Args:
            row_data: Dictionary containing the row data to write
        """
        logger.debug(f"Adding row to buffer (current size: {len(self.buffer)})")
        
        for attempt in range(self.max_retries):
            try:
                async with self.lock:
                    # Ensure all values are properly encoded strings
                    encoded_row = {
                        k: str(v).encode('utf-8', errors='replace').decode('utf-8') 
                        if v is not None else ''
                        for k, v in row_data.items()
                    }
                    
                    self.buffer.append(encoded_row)
                    self.total_rows += 1
                    
                    # If buffer is full, write to S3
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
        """Write buffered rows to S3 with retry logic."""
        if not self.buffer:
            return
            
        logger.debug(f"Flushing buffer with {len(self.buffer)} rows")
        
        for attempt in range(self.max_retries):
            try:
                await self.s3_storage.write_csv_rows(
                    self.output_key,
                    self.fieldnames,
                    self.buffer,
                    mode='a'
                )
                self.buffer = []
                logger.debug("Buffer flushed successfully")
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
        logger.debug(f"Closing CSV writer, flushing remaining {len(self.buffer)} rows")
        if self.buffer:
            await self.flush_buffer()
        logger.info(f"CSV writer closed. Total rows written: {self.total_rows}")
        
    async def __aenter__(self):
        """Support for async context manager."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup when used as context manager."""
        await self.close()

    async def get_total_rows(self):
        """Return the total number of rows written so far."""
        return self.total_rows