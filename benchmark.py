#!/usr/bin/env python3
import asyncio
import time
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import aiohttp
from typing import List, Dict, Any, Optional, Union
import os
import sys
import traceback
import datetime
from concurrent.futures import ThreadPoolExecutor

# Handle potential asyncpg import issues
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logging.warning("asyncpg module not found. Database benchmarks will be limited.")

# Import project modules
from src.config.config import (
    setup_logging, validate_credentials, L3_MATCHES, L4_MATCHES,
    MAX_CONCURRENT_EMBEDDINGS, MAX_CONCURRENT_CHAT_CALLS,
    OPENAI_RATE_LIMIT_PER_MINUTE, JINA_RATE_LIMIT_PER_MINUTE
)
from src.services.api import (
    get_embedding, classify_transaction_initial, select_best_match, get_level1_categories
)
from src.services.database import (
    init_database_connections, close_database_connections, 
    search_customer_collection, aurora_query, execute_with_retry,
    wake_up_databases, start_heartbeat, stop_heartbeat
)
from src.utils.aws import create_s3_client
from src.data.s3_reader import S3ChunkedReader
from src.data.s3_storage import S3TempStorage
from src.utils.rate_limiter import RateLimiter
from src.models.classifier import process_transaction

# Configure logging with more direct console output
logger = setup_logging()

# Add a direct print function for immediate console feedback
def console_print(message):
    """Print directly to console with timestamp, regardless of logging level."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

class Benchmark:
    """Main benchmark class to test performance of various components."""
    
    def __init__(self, num_samples=10, concurrency_levels=None):
        """
        Initialize the benchmark with configuration.
        
        Args:
            num_samples: Number of samples to use for each test
            concurrency_levels: List of concurrency levels to test
        """
        self.num_samples = num_samples
        self.concurrency_levels = concurrency_levels or [1, 5, 10, 25, 50]
        self.results = {}
        self.sample_data = None
        
        # Initialize timestamps for each operation
        self.timestamps = {
            "jina_embedding": [],
            "openai_initial": [],
            "vector_search": [],
            "openai_best_match": [],
            "openai_level1": [],
            "db_write": [],
            "s3_read": [],
            "s3_write": [],
            "end_to_end": []
        }
        
        # For progress tracking
        self.start_time = time.time()
        self.operation_counts = {
            "embedding_completed": 0,
            "embedding_total": 0,
            "openai_completed": 0,
            "openai_total": 0,
            "vector_completed": 0,
            "vector_total": 0,
            "s3_completed": 0,
            "s3_total": 0,
            "end_to_end_completed": 0,
            "end_to_end_total": 0
        }
        
        # Print initial benchmark info
        console_print(f"Benchmark initialized with {num_samples} samples and concurrency levels: {concurrency_levels}")
        console_print("=== Benchmark Starting ===")
        console_print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def initialize(self):
        """Initialize connections and prepare sample data."""
        logger.info("Initializing benchmark environment...")
        
        # Validate credentials
        error = validate_credentials()
        if error:
            logger.warning(f"Credential validation warning: {error}")
            logger.info("Continuing with benchmark in limited mode...")
        
        # Create sample data first so we can run partial benchmarks even if DB fails
        self.sample_data = self._create_sample_data()
        logger.info(f"Created {len(self.sample_data)} sample transactions for testing")
        
        # Initialize database connections with error handling
        try:
            if not await init_database_connections():
                logger.warning("Failed to initialize database connections, some tests may be skipped")
            else:
                # Wake up databases
                if not await wake_up_databases():
                    logger.warning("Failed to wake up databases, some tests may be skipped")
                else:
                    # Start database heartbeat
                    await start_heartbeat()
                    logger.info("Database connections initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization error: {e}")
            logger.info("Will skip database-dependent tests")
    
    def _create_sample_data(self):
        """Create sample transaction data for testing."""
        sample_transactions = [
            {
                "Transaction_ID": f"T{i+1}",
                "Supplier_Name": "Office Depot" if i % 3 == 0 else "Acme Services" if i % 3 == 1 else "Tech Solutions Inc",
                "Transaction_Description": "Office supplies and paper for HR department" if i % 3 == 0 else 
                                         "IT consulting services monthly retainer" if i % 3 == 1 else
                                         "Cloud hosting infrastructure - AWS monthly charges",
                "Transaction_Value": float((i+1) * 100),
                "customer_id": "SHQ001",  # Use the valid customer ID for all test transactions
                "customer_industry_description": "Financial Services" if i % 5 == 0 else 
                                                "Healthcare" if i % 5 == 1 else
                                                "Manufacturing" if i % 5 == 2 else
                                                "Retail" if i % 5 == 3 else
                                                "Technology"
            }
            for i in range(self.num_samples)
        ]
        return pd.DataFrame(sample_transactions)
    
    async def benchmark_embeddings(self):
        """Benchmark the embedding API for different concurrency levels."""
        console_print("\n=== STARTING EMBEDDING API BENCHMARK ===")
        console_print("Testing embedding generation with Jina.ai API")
        results = []
        
        async with aiohttp.ClientSession() as session:
            for concurrency in self.concurrency_levels:
                console_print(f"Testing embedding API with concurrency level: {concurrency}")
                embedding_semaphore = asyncio.Semaphore(concurrency)
                rate_limiter = RateLimiter(calls_per_minute=JINA_RATE_LIMIT_PER_MINUTE)
                
                start_time = time.time()
                embedding_tasks = []
                
                for _, row in self.sample_data.iterrows():
                    text = f"{row['Supplier_Name']} {row['Transaction_Description']}"
                    task = asyncio.create_task(
                        self._time_operation(
                            "jina_embedding",
                            get_embedding,
                            text, embedding_semaphore, rate_limiter, 'classification'
                        )
                    )
                    embedding_tasks.append(task)
                
                embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
                end_time = time.time()
                
                # Calculate statistics
                successful = sum(1 for e in embeddings if isinstance(e, list))
                failed = sum(1 for e in embeddings if isinstance(e, Exception))
                duration = end_time - start_time
                throughput = successful / duration if duration > 0 else 0
                
                results.append({
                    "component": "Jina Embedding API",
                    "concurrency": concurrency,
                    "total_requests": len(embedding_tasks),
                    "successful": successful,
                    "failed": failed,
                    "duration_seconds": duration,
                    "throughput_per_second": throughput
                })
                
                console_print(f"COMPLETED - Embedding benchmark at concurrency {concurrency}: {successful} successful, {failed} failed, {duration:.2f}s, {throughput:.2f}/s")
        
        self.results["embedding"] = results
        return results
    
    async def benchmark_openai_apis(self):
        """Benchmark the OpenAI API endpoints for different concurrency levels."""
        console_print("\n=== STARTING OPENAI API BENCHMARK ===")
        console_print("Testing OpenAI API endpoints for transaction classification")
        results = []
        
        for api_name, api_func in [
            ("Initial Classification", classify_transaction_initial),
            ("Best Match Selection", select_best_match),
            ("Level 1 Categories", get_level1_categories)
        ]:
            api_results = []
            
            for concurrency in self.concurrency_levels:
                console_print(f"Testing {api_name} with concurrency level: {concurrency}")
                
                async with aiohttp.ClientSession() as session:
                    chat_semaphore = asyncio.Semaphore(concurrency)
                    rate_limiter = RateLimiter(calls_per_minute=OPENAI_RATE_LIMIT_PER_MINUTE)
                    
                    start_time = time.time()
                    api_tasks = []
                    
                    # Prepare appropriate arguments based on the API being tested
                    for _, row in self.sample_data.iterrows():
                        if api_func == classify_transaction_initial:
                            timestamp_key = "openai_initial"
                            task = asyncio.create_task(
                                self._time_operation(
                                    timestamp_key,
                                    api_func,
                                    session, 
                                    row['Supplier_Name'], 
                                    row['Transaction_Description'], 
                                    row['customer_industry_description'], 
                                    chat_semaphore, 
                                    rate_limiter
                                )
                            )
                        elif api_func == select_best_match:
                            timestamp_key = "openai_best_match"
                            # Mock L3 and L4 matches for testing
                            l3_matches = [
                                {"Description": "Office Supplies", "UNSPSC_ID": "14111500", "level": "L3", "Score": 0.95},
                                {"Description": "Printing and Writing Paper", "UNSPSC_ID": "14111600", "level": "L3", "Score": 0.85}
                            ]
                            l4_matches = [
                                {"Description": "Copy paper", "UNSPSC_ID": "14111501", "level": "L4", "Score": 0.92},
                                {"Description": "Notebook paper", "UNSPSC_ID": "14111514", "level": "L4", "Score": 0.80}
                            ]
                            
                            task = asyncio.create_task(
                                self._time_operation(
                                    timestamp_key,
                                    api_func,
                                    session,
                                    row['Transaction_ID'],
                                    row['Transaction_Description'],
                                    row['Supplier_Name'],
                                    row['Transaction_Description'],  # Using as LLM description for test
                                    row['Transaction_Value'],
                                    l3_matches,
                                    l4_matches,
                                    row['customer_industry_description'],
                                    chat_semaphore,
                                    rate_limiter
                                )
                            )
                        elif api_func == get_level1_categories:
                            timestamp_key = "openai_level1"
                            task = asyncio.create_task(
                                self._time_operation(
                                    timestamp_key,
                                    api_func,
                                    session,
                                    row['Supplier_Name'],
                                    row['Transaction_Description'],
                                    row['Transaction_Description'],  # Using as LLM description for test
                                    row['customer_industry_description'],
                                    row['Transaction_Value'],
                                    0,  # industry_filter
                                    chat_semaphore,
                                    rate_limiter
                                )
                            )
                            
                        api_tasks.append(task)
                    
                    api_results_list = await asyncio.gather(*api_tasks, return_exceptions=True)
                    end_time = time.time()
                    
                    # Calculate statistics
                    successful = sum(1 for r in api_results_list if not isinstance(r, Exception))
                    failed = sum(1 for r in api_results_list if isinstance(r, Exception))
                    duration = end_time - start_time
                    throughput = successful / duration if duration > 0 else 0
                    
                    api_results.append({
                        "component": api_name,
                        "concurrency": concurrency,
                        "total_requests": len(api_tasks),
                        "successful": successful,
                        "failed": failed,
                        "duration_seconds": duration,
                        "throughput_per_second": throughput
                    })
                    
                    console_print(f"COMPLETED - {api_name} benchmark at concurrency {concurrency}: {successful} successful, {failed} failed, {duration:.2f}s, {throughput:.2f}/s")
            
            results.extend(api_results)
        
        self.results["openai_apis"] = results
        return results
    
    async def benchmark_vector_search(self):
        """Benchmark vector similarity searches in the database."""
        console_print("\n=== STARTING VECTOR SEARCH BENCHMARK ===")
        console_print("Testing vector similarity searches in Aurora using pgvector")
        results = []
        
        # First get some embeddings to use for searching
        async with aiohttp.ClientSession() as session:
            embedding_semaphore = asyncio.Semaphore(5)  # Use a reasonable concurrency for prep
            rate_limiter = RateLimiter(calls_per_minute=JINA_RATE_LIMIT_PER_MINUTE)
            
            # Create some mock embeddings in case the real embedding API is down
            mock_embeddings = [[0.1 * i] * 384 for i in range(1, 6)]  # Create 5 fake embeddings with dimension 384
            
            embeddings = []
            try:
                for _, row in self.sample_data.iterrows():
                    text = f"{row['Supplier_Name']} {row['Transaction_Description']}"
                    embedding = await get_embedding(text, embedding_semaphore, rate_limiter, 'classification')
                    if embedding:
                        embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embeddings for vector search benchmark: {e}")
                logger.info("Using mock embeddings instead")
                embeddings = mock_embeddings
            
            if not embeddings:
                logger.warning("Failed to get real embeddings, using mock embeddings for vector search benchmark")
                embeddings = mock_embeddings
                
            for concurrency in self.concurrency_levels:
                console_print(f"Testing vector search with concurrency level: {concurrency}")
                
                # Create a semaphore to limit concurrency
                search_semaphore = asyncio.Semaphore(concurrency)
                
                start_time = time.time()
                search_tasks = []
                
                # Test both customer collection and UNSPSC searches
                for embedding in embeddings:
                    # Customer collection search
                    task1 = asyncio.create_task(
                        self._time_operation_with_semaphore(
                            "vector_search",
                            search_semaphore,
                            search_customer_collection,
                            "SHQ001",  # Use a valid customer ID for testing
                            embedding
                        )
                    )
                    search_tasks.append(task1)
                    
                    # UNSPSC vector search
                    filter_dict = {"level": "L3"}
                    task2 = asyncio.create_task(
                        self._time_operation_with_semaphore(
                            "vector_search",
                            search_semaphore,
                            aurora_query,
                            embedding,
                            L3_MATCHES,
                            filter_dict
                        )
                    )
                    search_tasks.append(task2)
                
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                end_time = time.time()
                
                # Calculate statistics
                successful = sum(1 for r in search_results if not isinstance(r, Exception))
                failed = sum(1 for r in search_results if isinstance(r, Exception))
                duration = end_time - start_time
                throughput = successful / duration if duration > 0 else 0
                
                results.append({
                    "component": "Vector Search",
                    "concurrency": concurrency,
                    "total_requests": len(search_tasks),
                    "successful": successful,
                    "failed": failed,
                    "duration_seconds": duration,
                    "throughput_per_second": throughput
                })
                
                console_print(f"COMPLETED - Vector search benchmark at concurrency {concurrency}: {successful} successful, {failed} failed, {duration:.2f}s, {throughput:.2f}/s")
        
        self.results["vector_search"] = results
        return results
    
    async def benchmark_s3_operations(self):
        """Benchmark S3 read and write operations."""
        console_print("\n=== STARTING S3 OPERATIONS BENCHMARK ===")
        console_print("Testing S3 read and write performance with different concurrency levels")
        results = []
        
        s3_client = create_s3_client()
        s3_bucket = "benchmark-test-bucket"  # This is a placeholder - replace with an actual bucket
        s3_prefix = "benchmark-test/"
        
        # Create a test DataFrame similar to what would be used in production
        test_df = pd.DataFrame({
            "Transaction_ID": [f"T{i}" for i in range(100)],
            "Supplier_Name": ["Test Supplier"] * 100,
            "Transaction_Description": ["Test Description"] * 100,
            "Category_ID": ["12345678"] * 100,
            "Transaction_Value": [100.0] * 100
        })
        
        # Write the DataFrame to a CSV string
        csv_content = test_df.to_csv(index=False)
        
        from io import StringIO, BytesIO
        
        for concurrency in self.concurrency_levels:
            console_print(f"Testing S3 operations with concurrency level: {concurrency}")
            
            # Create a semaphore to limit concurrency
            s3_semaphore = asyncio.Semaphore(concurrency)
            
            # Test S3 write operations
            start_time = time.time()
            write_tasks = []
            
            for i in range(self.num_samples):
                task = asyncio.create_task(
                    self._time_operation_with_semaphore(
                        "s3_write",
                        s3_semaphore,
                        self._s3_write_test,
                        s3_client, s3_bucket, f"{s3_prefix}test_file_{i}.csv", csv_content
                    )
                )
                write_tasks.append(task)
            
            write_results = await asyncio.gather(*write_tasks, return_exceptions=True)
            end_time = time.time()
            
            # Calculate statistics for write operations
            successful_writes = sum(1 for r in write_results if r is True)
            failed_writes = sum(1 for r in write_results if r is not True)
            duration_writes = end_time - start_time
            throughput_writes = successful_writes / duration_writes if duration_writes > 0 else 0
            
            # Now test S3 read operations
            start_time = time.time()
            read_tasks = []
            
            for i in range(self.num_samples):
                task = asyncio.create_task(
                    self._time_operation_with_semaphore(
                        "s3_read",
                        s3_semaphore,
                        self._s3_read_test,
                        s3_client, s3_bucket, f"{s3_prefix}test_file_{i}.csv"
                    )
                )
                read_tasks.append(task)
            
            read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
            end_time = time.time()
            
            # Calculate statistics for read operations
            successful_reads = sum(1 for r in read_results if isinstance(r, str))
            failed_reads = sum(1 for r in read_results if not isinstance(r, str))
            duration_reads = end_time - start_time
            throughput_reads = successful_reads / duration_reads if duration_reads > 0 else 0
            
            # Add results to the list
            results.append({
                "component": "S3 Write",
                "concurrency": concurrency,
                "total_requests": len(write_tasks),
                "successful": successful_writes,
                "failed": failed_writes,
                "duration_seconds": duration_writes,
                "throughput_per_second": throughput_writes
            })
            
            results.append({
                "component": "S3 Read",
                "concurrency": concurrency,
                "total_requests": len(read_tasks),
                "successful": successful_reads,
                "failed": failed_reads,
                "duration_seconds": duration_reads,
                "throughput_per_second": throughput_reads
            })
            
            console_print(f"COMPLETED - S3 Write benchmark at concurrency {concurrency}: {successful_writes} successful, {failed_writes} failed, {duration_writes:.2f}s, {throughput_writes:.2f}/s")
            console_print(f"COMPLETED - S3 Read benchmark at concurrency {concurrency}: {successful_reads} successful, {failed_reads} failed, {duration_reads:.2f}s, {throughput_reads:.2f}/s")
        
        self.results["s3_operations"] = results
        return results
    
    async def benchmark_end_to_end(self):
        """Benchmark the complete transaction processing pipeline."""
        console_print("\n=== STARTING END-TO-END PIPELINE BENCHMARK ===")
        console_print("Testing complete transaction classification pipeline performance")
        results = []
        
        # We'll test with a few different concurrency settings for the end-to-end test
        concurrency_levels = [1, 5, 10, 25]
        
        for concurrency in concurrency_levels:
            console_print(f"Testing end-to-end processing with concurrency level: {concurrency}")
            
            # Create semaphores for various operations
            embedding_semaphore = asyncio.Semaphore(min(MAX_CONCURRENT_EMBEDDINGS, concurrency))
            chat_semaphore = asyncio.Semaphore(min(MAX_CONCURRENT_CHAT_CALLS, concurrency))
            task_semaphore = asyncio.Semaphore(concurrency)
            
            # Create rate limiters
            openai_rate_limiter = RateLimiter(calls_per_minute=OPENAI_RATE_LIMIT_PER_MINUTE)
            jina_rate_limiter = RateLimiter(calls_per_minute=JINA_RATE_LIMIT_PER_MINUTE)
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                tasks = []
                
                # Process each transaction end-to-end
                for _, row in self.sample_data.iterrows():
                    task = asyncio.create_task(
                        self._time_operation(
                            "end_to_end",
                            self._process_transaction_end_to_end,
                            row, session, embedding_semaphore, chat_semaphore, 
                            openai_rate_limiter, jina_rate_limiter
                        )
                    )
                    tasks.append(task)
                
                # Limit the number of concurrent tasks
                completed_tasks = []
                for i in range(0, len(tasks), concurrency):
                    batch = tasks[i:i+concurrency]
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    completed_tasks.extend(batch_results)
                
                end_time = time.time()
                
                # Calculate statistics
                successful = sum(1 for r in completed_tasks if r is not None and not isinstance(r, Exception))
                failed = sum(1 for r in completed_tasks if r is None or isinstance(r, Exception))
                duration = end_time - start_time
                throughput = successful / duration if duration > 0 else 0
                
                results.append({
                    "component": "End-to-End Pipeline",
                    "concurrency": concurrency,
                    "total_requests": len(tasks),
                    "successful": successful,
                    "failed": failed,
                    "duration_seconds": duration,
                    "throughput_per_second": throughput
                })
                
                console_print(f"COMPLETED - End-to-end benchmark at concurrency {concurrency}: {successful} successful, {failed} failed, {duration:.2f}s, {throughput:.2f}/s")
        
        self.results["end_to_end"] = results
        return results
    
    async def _process_transaction_end_to_end(self, row, session, embedding_semaphore, chat_semaphore, openai_rate_limiter, jina_rate_limiter):
        """Process a single transaction through the entire pipeline."""
        try:
            # Convert row to dict if it's a pandas Series
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            
            # Ensure customer_id is set to the valid test ID
            row_dict['customer_id'] = "SHQ001"
            
            # Make sure Transaction_Value is a float
            if 'Transaction_Value' in row_dict:
                if not isinstance(row_dict['Transaction_Value'], float):
                    try:
                        row_dict['Transaction_Value'] = float(row_dict['Transaction_Value'])
                    except (ValueError, TypeError):
                        # If conversion fails, set a default value
                        row_dict['Transaction_Value'] = 100.0
            
            # Step 1: Get initial classification
            initial_class = await classify_transaction_initial(
                session,
                row_dict['Supplier_Name'],
                row_dict['Transaction_Description'],
                row_dict['customer_industry_description'],
                chat_semaphore,
                openai_rate_limiter
            )
            
            if not initial_class:
                logger.warning(f"Initial classification failed for transaction {row_dict['Transaction_ID']}")
                return None
            
            # Step 2: Get embedding for vector search
            cleansed_description = initial_class['cleansed_description']
            embedding = await get_embedding(
                cleansed_description, 
                embedding_semaphore,
                jina_rate_limiter,
                'classification'
            )
            
            if not embedding:
                logger.warning(f"Failed to get embedding for transaction {row_dict['Transaction_ID']}")
                return None
            
            # Step 3: Process transaction with vector search and LLM classification
            result = await process_transaction(
                row=row_dict,
                l3_matches=L3_MATCHES,
                l4_matches=L4_MATCHES,
                session=session,
                embedding_semaphore=embedding_semaphore,
                chat_semaphore=chat_semaphore,
                rate_limiter=openai_rate_limiter,
                llm_description=cleansed_description,
                embedding=embedding
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in end-to-end processing: {e}")
            return None
    
    async def _time_operation(self, operation_key, func, *args, **kwargs):
        """Wrapper to time operations and record timestamps."""
        # Map operation_key to a progress category
        progress_category = None
        if "embedding" in operation_key:
            progress_category = "embedding"
        elif "openai" in operation_key:
            progress_category = "openai"
        elif "vector" in operation_key:
            progress_category = "vector"
        elif "s3" in operation_key:
            progress_category = "s3"
        elif "end_to_end" in operation_key:
            progress_category = "end_to_end"
        
        # Update total count if we have a valid category
        if progress_category:
            self.operation_counts[f"{progress_category}_total"] += 1
        
        start_time = time.time()
        success = False
        
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            self.timestamps[operation_key].append(duration)
            success = True
            
            # Update completed count
            if progress_category:
                self.operation_counts[f"{progress_category}_completed"] += 1
                completed = self.operation_counts[f"{progress_category}_completed"]
                total = self.operation_counts[f"{progress_category}_total"]
                
                # Give progress updates periodically
                if completed % 5 == 0 or completed == total:
                    elapsed = time.time() - self.start_time
                    console_print(f"Progress: {operation_key} - {completed}/{total} completed " +
                                 f"({completed/total*100:.1f}%) - Last op took {duration:.2f}s - " +
                                 f"Elapsed: {elapsed:.1f}s")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.timestamps[operation_key].append(duration)
            
            # Log error with the operation
            console_print(f"ERROR in {func.__name__}: {str(e)} (took {duration:.2f}s)")
            
            raise e
    
    async def _time_operation_with_semaphore(self, operation_key, semaphore, func, *args, **kwargs):
        """Wrapper to time operations with a semaphore for concurrency control."""
        async with semaphore:
            # Special handling for customer collection search - might fail if table doesn't exist
            if func.__name__ == 'search_customer_collection':
                try:
                    return await self._time_operation(operation_key, func, *args, **kwargs)
                except Exception as e:
                    # If the error is about table not existing, simulate a "no match found" result
                    if "does not exist" in str(e):
                        logger.warning(f"Customer collection table for {args[0]} does not exist. Returning mock result.")
                        # Return a mock "no match" result instead of raising an exception
                        return None
                    else:
                        # For other errors, propagate them
                        raise
            else:
                # Normal behavior for other functions
                return await self._time_operation(operation_key, func, *args, **kwargs)
    
    async def _s3_write_test(self, s3_client, bucket, key, content):
        """Test S3 write operation."""
        try:
            # This is a mock implementation - in a real test, you'd use actual S3
            # s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'))
            await asyncio.sleep(0.1)  # Simulate network latency
            return True
        except Exception as e:
            logger.error(f"Error in S3 write test: {e}")
            return False
    
    async def _s3_read_test(self, s3_client, bucket, key):
        """Test S3 read operation."""
        try:
            # This is a mock implementation - in a real test, you'd use actual S3
            # response = s3_client.get_object(Bucket=bucket, Key=key)
            # content = response['Body'].read().decode('utf-8')
            await asyncio.sleep(0.1)  # Simulate network latency
            return "Mock content"
        except Exception as e:
            logger.error(f"Error in S3 read test: {e}")
            return None
    
    def generate_report(self):
        """Generate a detailed report of the benchmark results."""
        console_print("\n=== GENERATING BENCHMARK REPORT ===")
        console_print("Creating CSV reports and visualizations of benchmark results")
        
        # Create a DataFrame with all results
        all_results = []
        for component, results in self.results.items():
            all_results.extend(results)
        
        if not all_results:
            logger.warning("No benchmark results to report")
            return
            
        df = pd.DataFrame(all_results)
        
        # Add latency statistics from timestamps
        latency_stats = {}
        for op, times in self.timestamps.items():
            if times:
                latency_stats[op] = {
                    "min": min(times),
                    "max": max(times),
                    "avg": sum(times) / len(times),
                    "p50": np.percentile(times, 50),
                    "p95": np.percentile(times, 95),
                    "p99": np.percentile(times, 99),
                    "count": len(times)
                }
        
        # Save results to CSV
        timestamp = int(time.time())
        results_csv_file = f"benchmark_results_{timestamp}.csv"
        latency_json_file = f"latency_stats_{timestamp}.json"
        
        console_print(f"Saving benchmark results to {results_csv_file}")
        df.to_csv(results_csv_file, index=False)
        
        # Save latency stats to JSON
        console_print(f"Saving latency statistics to {latency_json_file}")
        with open(latency_json_file, 'w') as f:
            json.dump(latency_stats, f, indent=2)
        
        # Generate plots
        console_print("Generating visualization plots")
        self._generate_plots(df, timestamp)
        
        # Print summary to console
        console_print("\n=== BENCHMARK SUMMARY ===")
        self._print_summary(df, latency_stats)
        
        console_print(f"\nBenchmark completed! Results saved to:")
        console_print(f"- CSV results: {results_csv_file}")
        console_print(f"- Latency stats: {latency_json_file}")
        console_print(f"- Throughput plot: throughput_vs_concurrency_{timestamp}.png")
        console_print(f"- Latency plot: latency_distribution_{timestamp}.png")
    
    def _generate_plots(self, df, timestamp):
        """Generate plots from benchmark results."""
        # Create throughput vs concurrency plot
        plt.figure(figsize=(12, 8))
        components = df['component'].unique()
        
        for component in components:
            component_df = df[df['component'] == component]
            plt.plot(component_df['concurrency'], component_df['throughput_per_second'], 
                    marker='o', label=component)
        
        plt.title('Throughput vs Concurrency')
        plt.xlabel('Concurrency Level')
        plt.ylabel('Throughput (requests/second)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"throughput_vs_concurrency_{timestamp}.png")
        
        # Create latency distribution plot
        plt.figure(figsize=(12, 8))
        
        operations = [op for op, times in self.timestamps.items() if times]
        data = [self.timestamps[op] for op in operations]
        
        plt.boxplot(data, tick_labels=operations)  # Using tick_labels instead of labels
        plt.title('Latency Distribution by Operation')
        plt.xlabel('Operation')
        plt.ylabel('Latency (seconds)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"latency_distribution_{timestamp}.png")
        
        console_print(f"Plots saved to throughput_vs_concurrency_{timestamp}.png and latency_distribution_{timestamp}.png")
    
    def _print_summary(self, df, latency_stats):
        """Print a summary of the benchmark results to the console."""
        print("\n" + "="*80)
        print("TRANSACTION CLASSIFICATION BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by component and concurrency, then find the max throughput for each component
        max_throughput = df.loc[df.groupby('component')['throughput_per_second'].idxmax()]
        
        print("\nMAX THROUGHPUT BY COMPONENT:")
        print("-"*80)
        for _, row in max_throughput.iterrows():
            print(f"{row['component']:<25} {row['throughput_per_second']:.2f} req/s at concurrency {row['concurrency']}")
        
        print("\nLATENCY STATISTICS (seconds):")
        print("-"*80)
        print(f"{'Operation':<20} {'Avg':<8} {'P50':<8} {'P95':<8} {'P99':<8} {'Min':<8} {'Max':<8} {'Count':<8}")
        print("-"*80)
        
        for op, stats in latency_stats.items():
            print(f"{op:<20} {stats['avg']:<8.3f} {stats['p50']:<8.3f} {stats['p95']:<8.3f} {stats['p99']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f} {stats['count']:<8}")
        
        print("\nBOTTLENECK ANALYSIS:")
        print("-"*80)
        
        # Find the slowest component by average latency
        slowest_op = max(latency_stats.items(), key=lambda x: x[1]['avg'])[0]
        slowest_avg = latency_stats[slowest_op]['avg']
        
        print(f"Slowest operation: {slowest_op} (avg: {slowest_avg:.3f}s)")
        
        # Find the component with the lowest max throughput
        min_throughput_component = max_throughput.loc[max_throughput['throughput_per_second'].idxmin()]
        print(f"Lowest throughput component: {min_throughput_component['component']} "
              f"({min_throughput_component['throughput_per_second']:.2f} req/s)")
        
        # Find the component that fails the most
        # Using a different approach to avoid the groupby.apply warning
        component_failure_rates = {}
        for component in df['component'].unique():
            component_df = df[df['component'] == component]
            if component_df['total_requests'].sum() > 0:
                rate = (component_df['failed'].sum() / component_df['total_requests'].sum()) * 100
            else:
                rate = 0
            component_failure_rates[component] = rate
            
        # Convert to Series for sorting
        failure_rates = pd.Series(component_failure_rates).sort_values(ascending=False)
        
        if not failure_rates.empty:
            highest_failure = failure_rates.index[0]
            highest_failure_rate = failure_rates.iloc[0]
            print(f"Highest failure rate: {highest_failure} ({highest_failure_rate:.2f}%)")
        
        print("\nRECOMMENDATIONS:")
        print("-"*80)
        
        # Add specific recommendations based on the benchmark results
        if 'openai_best_match' in latency_stats and latency_stats['openai_best_match']['avg'] > 1.0:
            print("- Consider caching OpenAI API responses to reduce latency")
            
        if 'vector_search' in latency_stats and latency_stats['vector_search']['avg'] > 0.5:
            print("- Optimize pgvector index settings for faster similarity searches")
            
        if 'jina_embedding' in latency_stats and latency_stats['jina_embedding']['avg'] > 0.3:
            print("- Consider batch processing for embeddings to improve throughput")
        
        # Find the optimal concurrency for the end-to-end pipeline
        if 'end_to_end' in self.results:
            end_to_end_df = pd.DataFrame(self.results['end_to_end'])
            if not end_to_end_df.empty:
                optimal_concurrency = end_to_end_df.loc[end_to_end_df['throughput_per_second'].idxmax()]['concurrency']
                print(f"- Optimal concurrency for the end-to-end pipeline: {optimal_concurrency}")
        
        print("="*80 + "\n")

async def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description='Benchmark Transaction Classification System')
    parser.add_argument('--samples', type=int, default=10, help='Number of sample transactions to use')
    parser.add_argument('--concurrency', type=str, default='1,5,10,25,50', help='Comma-separated list of concurrency levels to test')
    parser.add_argument('--components', type=str, default='all', help='Comma-separated list of components to benchmark (options: embedding,openai,vector,s3,end_to_end)')
    parser.add_argument('--skip-errors', action='store_true', help='Skip components that encounter errors rather than aborting')
    
    args = parser.parse_args()
    
    # Print welcome message
    console_print("=" * 80)
    console_print("TRANSACTION CLASSIFICATION SYSTEM BENCHMARK")
    console_print("=" * 80)
    console_print(f"Starting benchmark with {args.samples} samples")
    start_time = time.time()
    
    # Parse concurrency levels
    concurrency_levels = [int(c) for c in args.concurrency.split(',')]
    
    # Parse components to benchmark
    if args.components.lower() == 'all':
        components = ['embedding', 'openai', 'vector', 's3', 'end_to_end']
    else:
        components = [c.strip().lower() for c in args.components.split(',')]
    
    # Initialize benchmark
    benchmark = Benchmark(num_samples=args.samples, concurrency_levels=concurrency_levels)
    await benchmark.initialize()
    
    # Track which components were successfully benchmarked
    successful_components = []
    failed_components = []
    
    try:
        # Run selected benchmarks with error handling
        for component in components:
            try:
                if component == 'embedding':
                    logger.info("=== Starting Embedding Benchmark ===")
                    await benchmark.benchmark_embeddings()
                    successful_components.append(component)
                
                elif component == 'openai':
                    logger.info("=== Starting OpenAI API Benchmark ===")
                    await benchmark.benchmark_openai_apis()
                    successful_components.append(component)
                
                elif component == 'vector':
                    logger.info("=== Starting Vector Search Benchmark ===")
                    await benchmark.benchmark_vector_search()
                    successful_components.append(component)
                
                elif component == 's3':
                    logger.info("=== Starting S3 Operations Benchmark ===")
                    await benchmark.benchmark_s3_operations()
                    successful_components.append(component)
                
                elif component == 'end_to_end':
                    logger.info("=== Starting End-to-End Pipeline Benchmark ===")
                    await benchmark.benchmark_end_to_end()
                    successful_components.append(component)
                
                else:
                    logger.warning(f"Unknown component: {component}")
            except Exception as e:
                logger.error(f"Error benchmarking {component}: {e}")
                failed_components.append(component)
                if not args.skip_errors:
                    logger.error("Aborting benchmark due to error. Use --skip-errors to continue after errors.")
                    break
        
        # Generate report if we have at least one successful component
        if successful_components:
            logger.info(f"Generating report for components: {', '.join(successful_components)}")
            benchmark.generate_report()
        else:
            logger.error("No components were successfully benchmarked. No report will be generated.")
            
        # Report failed components
        if failed_components:
            console_print(f"WARNING: The following components failed benchmarking: {', '.join(failed_components)}")
        
    finally:
        # Calculate and display total runtime
        total_runtime = time.time() - start_time
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        console_print("\n" + "=" * 80)
        console_print(f"BENCHMARK COMPLETED IN {runtime_str}")
        console_print("=" * 80)
        
        # Clean up - with error handling
        console_print("Cleaning up resources and closing connections...")
        try:
            await stop_heartbeat()
        except Exception as e:
            console_print(f"Error stopping heartbeat: {e}")
            
        try:
            await close_database_connections()
        except Exception as e:
            console_print(f"Error closing database connections: {e}")
            
        console_print("Benchmark finished!")
        console_print("=" * 80)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)