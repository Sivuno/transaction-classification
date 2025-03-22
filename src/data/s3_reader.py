import csv
import io
import logging
import pandas as pd
import re
from typing import Generator, Dict, Any

logger = logging.getLogger(__name__)

class S3ChunkedReader:
    """
    Memory-efficient reader for large CSV files stored in S3.
    Implements streaming to handle files that exceed memory capacity.
    """
    def __init__(self, s3_client, bucket, key, chunk_size=500, buffer_size=10485760):  # 10MB buffer
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self._header = None
        
    async def get_total_rows(self):
        """
        Get total row count using streaming to handle large files.
        
        Returns:
            int: Total number of rows in the file
        """
        try:
            # Get file size
            response = self.s3_client.head_object(Bucket=self.bucket, Key=self.key)
            file_size = response['ContentLength']
            
            # Initialize counters
            total_rows = 0
            current_position = 0
            buffer = ""
            incomplete_row = ""
            first_chunk = True
            
            while current_position < file_size:
                end_position = min(current_position + self.buffer_size, file_size)
                
                range_response = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=self.key,
                    Range=f'bytes={current_position}-{end_position-1}'
                )
                
                chunk_content = range_response['Body'].read().decode('utf-8', errors='replace')
                buffer = incomplete_row + chunk_content
                
                # Split into rows and handle incomplete last row
                rows = buffer.split('\n')
                if current_position + self.buffer_size < file_size:
                    incomplete_row = rows[-1]
                    rows = rows[:-1]
                else:
                    incomplete_row = ""
                
                # Skip header in first chunk
                if first_chunk:
                    rows = rows[1:]
                    first_chunk = False
                
                total_rows += len(rows)
                current_position = end_position
                
                # Close the response to free resources
                range_response['Body'].close()
            
            logger.info(f"Counted {total_rows} rows in {self.key}")
            return total_rows
            
        except Exception as e:
            logger.error(f"Error counting rows in {self.key}: {e}")
            raise
            
    async def read_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """
        Stream the file in memory-efficient chunks with robust CSV parsing.
        
        Yields:
            pandas.DataFrame: Chunks of data from the file
        """
        try:
            # First, get just the header
            header_response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=self.key,
                Range='bytes=0-4096'  # Read first 4KB to get header
            )
            header_content = header_response['Body'].read().decode('utf-8', errors='replace')
            self._header = next(csv.reader(io.StringIO(header_content)))
            
            # Get total file size for range requests
            response = self.s3_client.head_object(Bucket=self.bucket, Key=self.key)
            file_size = response['ContentLength']
            
            # Process file in chunks using range requests
            current_position = 0
            buffer = ""
            incomplete_row = ""
            
            while current_position < file_size:
                end_position = min(current_position + self.buffer_size, file_size)
                
                range_response = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=self.key,
                    Range=f'bytes={current_position}-{end_position-1}'
                )
                
                chunk_content = range_response['Body'].read().decode('utf-8', errors='replace')
                buffer = incomplete_row + chunk_content
                
                # Find the last complete row in this buffer
                rows = buffer.split('\n')
                if current_position + self.buffer_size < file_size:
                    incomplete_row = rows[-1]
                    rows = rows[:-1]
                else:
                    incomplete_row = ""
                
                if current_position == 0:
                    rows = rows[1:]  # Skip header in first chunk
                
                # Process complete rows
                if rows:
                    # Step A: Handle special CSV parsing cases with improved regex
                    cleaned_rows = []
                    for row in rows:
                        # Handle quoted fields containing escaped quotes
                        row = re.sub(
                            r'(".*?)\\+"(?!,|\n|$)',  # Match escaped quotes inside fields
                            lambda m: m.group(1) + '"',
                            row
                        )
                        # Handle quoted fields with trailing backslashes
                        row = re.sub(
                            r'(".*?)(\\+)"(?=,|\n|$)',
                            lambda m: m.group(1) + '"',
                            row
                        )
                        # Handle unquoted fields with trailing backslashes
                        row = re.sub(
                            r'([^",\n]+)(\\+)(?=,|\n|$)',
                            lambda m: m.group(1),
                            row
                        )
                        cleaned_rows.append(row)
                    
                    # Step B: Construct the CSV string from header and cleaned rows
                    csv_text = '\n'.join([','.join(self._header)] + cleaned_rows)
                    
                    # Step C: Parse the cleaned CSV text with enhanced parameters
                    df_chunk = pd.read_csv(
                        io.StringIO(csv_text),
                        engine='python',
                        on_bad_lines='warn',
                        quoting=csv.QUOTE_MINIMAL,
                        escapechar='\\',
                        doublequote=True,  # Added parameter to handle doubled quotes
                        encoding_errors='replace'
                    )
                    
                    # Process in smaller chunks for memory efficiency
                    for i in range(0, len(df_chunk), self.chunk_size):
                        sub_chunk = df_chunk.iloc[i:i + self.chunk_size].copy()
                        
                        # Convert types and handle missing values
                        for col in sub_chunk.columns:
                            if col == 'Transaction_Value':
                                sub_chunk[col] = pd.to_numeric(sub_chunk[col], errors='coerce')
                        
                        sub_chunk = sub_chunk.fillna({
                            'Transaction_ID': 'UNKNOWN',
                            'Supplier_Name': 'UNKNOWN',
                            'Transaction_Description': 'UNKNOWN',
                            'Transaction_Value': 0.0,
                            'customer_id': 'UNKNOWN',
                            'customer_industry_description': 'UNKNOWN'
                        })
                        
                        yield sub_chunk
                
                current_position = end_position
                
                # Close the response to free resources
                range_response['Body'].close()
                
        except Exception as e:
            logger.error(f"Error reading file {self.key}: {str(e)}")
            raise