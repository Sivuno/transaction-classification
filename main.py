#!/usr/bin/env python3
import asyncio
import sys
import time
from typing import List, Dict, Any
import pandas as pd

from src.config.config import (
    validate_credentials, setup_logging, L3_MATCHES, L4_MATCHES,
    S3_INPUT_PREFIX, S3_OUTPUT_BUCKET, S3_PROCESSING_PREFIX, S3_OUTPUT_PREFIX,
    USE_DYNAMODB, DYNAMODB_TRANSACTIONS_TABLE, DYNAMODB_RESULTS_TABLE
)
from src.services.database import (
    init_database_connections, wake_up_databases, close_database_connections, 
    start_heartbeat, stop_heartbeat
)
from src.utils.aws import (
    create_s3_client, list_csv_files_in_s3_folder, parse_s3_uri,
    create_dynamodb_resource
)
from src.data.s3_reader import S3ChunkedReader
from src.utils.progress import GlobalProgress
from src.models.classifier import process_transactions

# Set up logging
logger = setup_logging()

async def get_total_rows_robust(files_to_process):
    """
    Get total row count across all files with robust error handling.
    
    Args:
        files_to_process: List of S3 URIs to process
        
    Returns:
        int: Total number of rows across all files
    """
    total_rows = 0
    s3_client = create_s3_client()
    
    for file_uri in files_to_process:
        try:
            bucket, key = parse_s3_uri(file_uri)
            reader = S3ChunkedReader(s3_client, bucket, key)
            file_rows = await reader.get_total_rows()
            total_rows += file_rows
            logger.info(f"Counted {file_rows} rows in {file_uri}")
        except Exception as e:
            logger.error(f"Error counting rows in {file_uri}: {e}")
            continue
            
    return total_rows

async def process_s3_file(file_uri, chunk_size=500):
    """
    Process an S3 file in chunks to avoid memory issues.
    
    Args:
        file_uri: S3 URI of the file to process
        chunk_size: Number of rows to process in each chunk
        
    Yields:
        DataFrame: Chunks of the file as DataFrames
    """
    logger.debug(f"Processing S3 file: {file_uri}")
    s3_client = create_s3_client()
    bucket, key = parse_s3_uri(file_uri)
    
    try:
        reader = S3ChunkedReader(s3_client, bucket, key, chunk_size=chunk_size)
        async for chunk_df in reader.read_chunks():
            yield chunk_df
    except Exception as e:
        logger.error(f"Error processing S3 file {file_uri}: {e}")
        raise

async def main():
    """Main function for processing transactions."""
    logger.info("Starting transaction classification process")
    
    # Check credentials
    error = validate_credentials()
    if error:
        logger.error(error)
        sys.exit(1)
    
    # Initialize database connections
    if not await init_database_connections():
        logger.error("Failed to initialize database connections")
        sys.exit(1)
    
    # Wake up databases
    if not await wake_up_databases():
        logger.error("Failed to wake up databases")
        sys.exit(1)
        
    # Start database heartbeat to keep connections alive
    await start_heartbeat()
    
    timestamp = int(time.time())
    
    # Initialize storage based on configuration
    if USE_DYNAMODB:
        from src.data.dynamodb_storage import DynamoDBStorage
        logger.info("Using DynamoDB for storage")
        storage = DynamoDBStorage(
            table_name=DYNAMODB_RESULTS_TABLE,
            region_name=None,  # Use values from config
            aws_access_key_id=None,  # Use values from config
            aws_secret_access_key=None  # Use values from config
        )
        output_key = f"results_{timestamp}"
    else:
        from src.data.s3_storage import S3TempStorage
        logger.info("Using S3 for storage")
        storage = S3TempStorage(
            s3_client=create_s3_client(),
            bucket=S3_OUTPUT_BUCKET,
            prefix=S3_PROCESSING_PREFIX
        )
        output_key = f"final_llm_results_{timestamp}.csv"
    
    # Define CSV field names
    fieldnames = [
        "Transaction_ID", "Matched_Transaction_ID", "Supplier_Name", "Transaction_Description",
        "LLM_Description", "Transaction_Value", "Category_ID", 
        "Category_Description", "level", "Reason", "Confidence_Score",
        "Embedding_Matches", "Level_1_Categories", "Single_Word", "Match_Source",
        "transaction_type", "sourceable_flag", "Match_Score", "searchable", "rfp_description"
    ]
    
    start_time = time.time()
    temp_files = []
    
    try:
        # Find files to process
        files_to_process = list_csv_files_in_s3_folder(S3_INPUT_PREFIX)
        if not files_to_process:
            logger.error(f"No CSV files found in {S3_INPUT_PREFIX}")
            sys.exit(1)
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Get total row count
        total_transactions = await get_total_rows_robust(files_to_process)
        
        # Initialize progress tracker
        global_progress = GlobalProgress(total_transactions)
        global_progress.initialize_progress_bar()
        
        # Process all files
        all_failed_rows = []
        for file_number, file_uri in enumerate(files_to_process, 1):
            logger.info(f"Processing file {file_number}/{len(files_to_process)}: {file_uri}")
            
            chunk_number = 0
            try:
                async for chunk_df in process_s3_file(file_uri, chunk_size=500):
                    chunk_number += 1
                    
                    if USE_DYNAMODB:
                        temp_key = f"results_{timestamp}_{file_number}_{chunk_number}"
                    else:
                        temp_key = f"temp_results_{timestamp}_{file_number}_{chunk_number}.csv"
                        temp_files.append(temp_key)
                    
                    # Process the chunk and collect failed rows
                    failed_rows, _ = await process_transactions(
                        chunk_df, 
                        L3_MATCHES, 
                        L4_MATCHES, 
                        fieldnames, 
                        storage,
                        global_progress,
                        temp_key
                    )
                    all_failed_rows.extend(failed_rows)
            except Exception as e:
                logger.error(f"Error processing file {file_uri}: {e}")
                continue
        
        # Retry failed transactions
        max_retry_rounds = 3
        retry_round = 1
        
        while all_failed_rows and retry_round <= max_retry_rounds:
            logger.info(f"Retry round {retry_round}/{max_retry_rounds}: Retrying {len(all_failed_rows)} failed transactions")
            
            # Convert failed rows to DataFrame
            retry_df = pd.DataFrame(all_failed_rows)
            all_failed_rows = []  # Reset for this round
            
            if USE_DYNAMODB:
                retry_key = f"retry_{timestamp}_{retry_round}"
            else:
                retry_key = f"retry_results_{timestamp}_{retry_round}.csv"
                temp_files.append(retry_key)
            
            # Process retry batch
            failed_rows, _ = await process_transactions(
                retry_df,
                L3_MATCHES,
                L4_MATCHES,
                fieldnames,
                storage,
                global_progress,
                retry_key
            )
            
            all_failed_rows = failed_rows
            retry_round += 1
        
        # Combine results if using S3
        if not USE_DYNAMODB and temp_files:
            from src.data.s3_storage import S3TempStorage
            s3_storage = storage
            
            logger.info(f"Combining {len(temp_files)} temporary files into final output")
            success = await s3_storage.combine_csv_files(
                temp_files,
                output_key,
                fieldnames
            )
            
            if success:
                logger.info(f"Successfully combined results into {output_key}")
                
                # Move to final output location
                s3_client = create_s3_client()
                source_key = f"{S3_PROCESSING_PREFIX}{output_key}"
                dest_key = f"{S3_OUTPUT_PREFIX}{output_key}"
                
                try:
                    s3_client.copy_object(
                        Bucket=S3_OUTPUT_BUCKET,
                        CopySource={'Bucket': S3_OUTPUT_BUCKET, 'Key': source_key},
                        Key=dest_key
                    )
                    logger.info(f"Copied final results to {dest_key}")
                    
                    # Delete source file
                    s3_client.delete_object(
                        Bucket=S3_OUTPUT_BUCKET,
                        Key=source_key
                    )
                except Exception as e:
                    logger.error(f"Error copying final results: {e}")
            else:
                logger.error("Failed to combine temporary files")
        
        # Clean up temporary files if using S3
        if not USE_DYNAMODB:
            try:
                from src.data.s3_storage import S3TempStorage
                s3_storage = storage
                s3_storage.cleanup_temp_files()
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
        
        # Log final statistics
        elapsed_time = time.time() - start_time
        processed_count = global_progress.processed_count
        failed_count = global_progress.failed_count
        
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {processed_count} transactions successfully")
        logger.info(f"Failed to process {failed_count} transactions")
        
        if failed_count > 0:
            logger.warning(f"Some transactions failed to process. Check logs for details.")
        
        if USE_DYNAMODB:
            logger.info(f"Results stored in DynamoDB table: {DYNAMODB_RESULTS_TABLE}")
        else:
            logger.info(f"Results stored in S3: s3://{S3_OUTPUT_BUCKET}/{S3_OUTPUT_PREFIX}{output_key}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        sys.exit(1)
    finally:
        # Stop database heartbeat
        await stop_heartbeat()
        
        # Close database connections
        await close_database_connections()
        
        logger.info("Transaction classification process completed")

if __name__ == "__main__":
    asyncio.run(main())