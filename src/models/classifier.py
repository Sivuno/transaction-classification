import asyncio
import json
import logging
import aiohttp
from typing import List, Dict, Any, Optional, Tuple

from ..config.config import (
    L3_MATCHES, L4_MATCHES, MAX_CONCURRENT_EMBEDDINGS, MAX_CONCURRENT_CHAT_CALLS,
    OPENAI_RATE_LIMIT_PER_MINUTE, JINA_RATE_LIMIT_PER_MINUTE
)
from ..utils.rate_limiter import RateLimiter
from ..services.api import (
    get_embedding, classify_transaction_initial, select_best_match, get_level1_categories
)
from ..services.database import search_customer_collection, aurora_query, get_customer_collection_name, execute_with_retry

logger = logging.getLogger(__name__)

async def process_transaction(
    row: Dict[str, Any], 
    l3_matches: int, 
    l4_matches: int, 
    session: aiohttp.ClientSession, 
    embedding_semaphore: asyncio.Semaphore, 
    chat_semaphore: asyncio.Semaphore, 
    rate_limiter: RateLimiter, 
    llm_description: str, 
    embedding: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Process a transaction to find the best category match.
    
    Args:
        row: Transaction data row
        l3_matches: Number of L3 category matches to retrieve
        l4_matches: Number of L4 category matches to retrieve
        session: aiohttp session
        embedding_semaphore: Semaphore for controlling concurrent embedding requests
        chat_semaphore: Semaphore for controlling concurrent chat requests
        rate_limiter: Rate limiter for API calls
        llm_description: Cleansed description from initial classification
        embedding: Pre-computed embedding vector
        
    Returns:
        Optional[Dict[str, Any]]: Classification result or None if failed
    """
    logger.debug(f"process_transaction called for transaction {row['Transaction_ID']}.")
    max_retries = 5
    customer_industry_description = row.get('customer_industry_description', '')

    for attempt in range(max_retries):
        try:
            # Get level 1 categories for filtering
            logger.debug("Getting level 1 categories...")
            level1_categories = await get_level1_categories(
                session,
                row['Supplier_Name'],
                row['Transaction_Description'],
                llm_description,
                customer_industry_description,
                row['Transaction_Value'],
                row.get('industry_filter', 0),
                chat_semaphore,
                rate_limiter
            )

            if not level1_categories:
                logger.warning(f"No Level 1 categories found for transaction {row['Transaction_ID']}")
                level1_categories = None

            if not embedding:
                raise Exception("Failed to get embedding")

            # Process L3 and L4 matches separately
            unique_matches = {"L3": [], "L4": []}
            all_matches_formatted = []  # Store all matches for the Embedding_Matches column
            logger.debug("Querying Aurora for L3 and L4 matches using aurora_query replacement.")

            for level in ["L3", "L4"]:
                filter_dict = {"level": level}
                if level1_categories:
                    filter_dict["level_1_lookup"] = level1_categories

                matches_count = l3_matches if level == "L3" else l4_matches

                query_result = await aurora_query(
                    embedding=embedding,
                    top_k=matches_count,
                    query_filter=filter_dict
                )

                if not query_result or 'matches' not in query_result or not query_result['matches']:
                    logger.warning(f"No matches found for {level} in transaction {row['Transaction_ID']}")
                    continue

                for match in query_result['matches']:
                    match_metadata = match.get('metadata', {})
                    if match_metadata:
                        match_info = {
                            'Description': match_metadata.get('Description', 'No Description Available'),
                            'UNSPSC_ID': match_metadata.get('UNSPSC_ID', 'No ID Available'),
                            'level': match_metadata.get('level', level),
                            'Score': match.get('score', 0)
                        }
                        unique_matches[level].append(match_info)
                        
                        formatted_match = f"{match_metadata.get('UNSPSC_ID', 'No ID')} - {match_metadata.get('Description', 'No Description Available')}"
                        all_matches_formatted.append(formatted_match)

            logger.debug(f"Found {len(unique_matches['L3'])} L3 and {len(unique_matches['L4'])} L4 matches.")

            best_match = await select_best_match(
                session,
                row['Transaction_ID'],
                row['Transaction_Description'],
                row['Supplier_Name'],
                llm_description,
                row['Transaction_Value'],
                unique_matches["L3"],
                unique_matches["L4"],
                customer_industry_description,
                chat_semaphore,
                rate_limiter
            )

            if best_match:
                # Add the formatted embedding matches and level 1 categories to the result
                best_match['Embedding_Matches'] = '; '.join(all_matches_formatted)
                best_match['Level_1_Categories'] = level1_categories if level1_categories else []
                return best_match

        except Exception as e:
            logger.warning(f"Error in process_transaction (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"Failed process_transaction after {max_retries} attempts: {e}")
                return None

    return None

async def process_transaction_with_csv(
    row: Dict[str, Any],
    l3_matches: int,
    l4_matches: int,
    session: aiohttp.ClientSession,
    embedding_semaphore: asyncio.Semaphore,
    chat_semaphore: asyncio.Semaphore,
    openai_rate_limiter: RateLimiter,
    jina_rate_limiter: RateLimiter,
    csv_writer
) -> Optional[Dict[str, Any]]:
    """
    Process a single transaction row and write results to CSV.
    Checks for previous matches in the customer's Aurora table.
    
    Args:
        row: Transaction data row
        l3_matches: Number of L3 category matches to retrieve
        l4_matches: Number of L4 category matches to retrieve
        session: aiohttp session
        embedding_semaphore: Semaphore for controlling concurrent embedding requests
        chat_semaphore: Semaphore for controlling concurrent chat requests
        openai_rate_limiter: Rate limiter for OpenAI API calls
        jina_rate_limiter: Rate limiter for Jina API calls
        csv_writer: IncrementalCSVWriter instance for writing results
        
    Returns:
        Optional[Dict[str, Any]]: Classification result or None if failed
    """
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"process_transaction_with_csv attempt {attempt + 1}/{max_retries} for transaction {row.get('Transaction_ID', 'UNKNOWN')}")
            
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            transaction_id = row_dict.get('Transaction_ID', 'UNKNOWN')
            
            customer_id = str(row_dict.get('customer_id', '')).strip()
            if not customer_id:
                logger.error(f"Missing customer_id in row for transaction {transaction_id}")
                return None
                
            # Create embedding for customer collection search
            customer_search_text = f"{row_dict.get('Supplier_Name', '')} {row_dict.get('Transaction_Description', '')}"
            logger.debug(f"Creating customer search embedding for transaction {transaction_id}")
            
            customer_search_embedding = await get_embedding(
                customer_search_text, 
                embedding_semaphore,
                jina_rate_limiter,
                'classification'
            )
            if not customer_search_embedding:
                logger.error(f"Failed to get customer search embedding for transaction {transaction_id}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return None

            # Check for previous matches
            previous_match = await search_customer_collection(customer_id, customer_search_embedding)

            if previous_match and previous_match['score'] >= 0.95:
                logger.debug(f"Found previous match with score >= 0.95 for transaction {transaction_id}")
                meta = previous_match['metadata']
                result = {
                    "Transaction_ID": transaction_id,
                    "Matched_Transaction_ID": meta.get('matched_transaction_id', transaction_id),
                    "Supplier_Name": row_dict.get('Supplier_Name'),
                    "Transaction_Description": row_dict.get('Transaction_Description'),
                    "LLM_Description": meta.get('cleansed_description', ''),
                    "Transaction_Value": row_dict.get('Transaction_Value'),
                    "Category_ID": meta.get('unspsc_id', ''),
                    "Category_Description": meta.get('category_description', ''),
                    "level": meta.get("level", ""),
                    "Reason": meta.get('llm_reasoning', ''),
                    "Confidence_Score": 10,
                    "Embedding_Matches": "Matched from customer collection",
                    "Level_1_Categories": meta.get('level_1_categories', []),
                    "Single_Word": meta.get('single_word', 'misc'),
                    "Match_Source": "previous_transaction",
                    "transaction_type": meta.get('transaction_type', 'Indirect Services'),
                    "sourceable_flag": meta.get('sourceable_flag', False),
                    "Match_Score": previous_match['score'],
                    "searchable": meta.get('searchable', False),
                    "rfp_description": meta.get('rfp_description', "")
                }
                
                await csv_writer.write_row(result)
                return result

            # If no previous match, proceed with full classification
            customer_industry_description = row_dict.get('customer_industry_description', '')
            
            # Initial classification
            gpt_initial = await classify_transaction_initial(
                session,
                row_dict.get('Supplier_Name'),
                row_dict.get('primary_descriptor', row_dict.get('Transaction_Description')),
                customer_industry_description,
                chat_semaphore,
                openai_rate_limiter,
                row_dict.get('alternate_descriptor_1'),
                row_dict.get('alternate_descriptor_2'),
                row_dict.get('Transaction_Value', 0.0)
            )

            if not gpt_initial:
                cleansed_description = row_dict.get('Transaction_Description')
                sourceable_flag = False
                transaction_type = "Indirect Services"  # fallback new type
                single_word = "misc"
                searchable = False
                rfp_description = row_dict.get('Transaction_Description')
            else:
                cleansed_description = gpt_initial['cleansed_description']
                sourceable_flag = gpt_initial['sourceable_flag']
                transaction_type = gpt_initial['transaction_type']
                single_word = gpt_initial['single_word']
                searchable = gpt_initial.get('searchable', False)
                rfp_description = gpt_initial.get('rfp_description', '')

            # Create UNSPSC matching embedding
            unspsc_search_embedding = await get_embedding(
                cleansed_description, 
                embedding_semaphore,
                jina_rate_limiter,
                'classification'
            )
            if not unspsc_search_embedding:
                logger.error(f"Failed to get UNSPSC search embedding for transaction {transaction_id}")
                return None

            final_classification = await process_transaction(
                row=row_dict,
                l3_matches=l3_matches,
                l4_matches=l4_matches,
                session=session,
                embedding_semaphore=embedding_semaphore,
                chat_semaphore=chat_semaphore,
                rate_limiter=openai_rate_limiter,
                llm_description=cleansed_description,
                embedding=unspsc_search_embedding
            )

            if final_classification:
                # Add additional fields
                final_classification.update({
                    "Match_Source": "new_classification",
                    "transaction_type": transaction_type,
                    "sourceable_flag": sourceable_flag,
                    "Single_Word": single_word,
                    "Matched_Transaction_ID": transaction_id,
                    "searchable": searchable,
                    "rfp_description": rfp_description
                })

                # Convert and validate level_1_categories for PostgreSQL
                level_1_categories = final_classification.get('Level_1_Categories', [])
                if isinstance(level_1_categories, str):
                    try:
                        level_1_categories = json.loads(level_1_categories)
                    except json.JSONDecodeError:
                        level_1_categories = []
                
                if not isinstance(level_1_categories, list):
                    level_1_categories = []
                level_1_categories = [int(x) for x in level_1_categories if str(x).isdigit()]

                # Insert into customer table with UPSERT functionality
                table_name = get_customer_collection_name(customer_id)
                
                insert_query = f"""
                    INSERT INTO {table_name}
                    (transaction_id, embedding, unspsc_id, cleansed_description,
                     sourceable_flag, category_description, transaction_type,
                     llm_reasoning, single_word, level_1_categories, level, searchable, rfp_description)
                    VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (transaction_id) 
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        unspsc_id = EXCLUDED.unspsc_id,
                        cleansed_description = EXCLUDED.cleansed_description,
                        sourceable_flag = EXCLUDED.sourceable_flag,
                        category_description = EXCLUDED.category_description,
                        transaction_type = EXCLUDED.transaction_type,
                        llm_reasoning = EXCLUDED.llm_reasoning,
                        single_word = EXCLUDED.single_word,
                        level_1_categories = EXCLUDED.level_1_categories,
                        level = EXCLUDED.level,
                        searchable = EXCLUDED.searchable,
                        rfp_description = EXCLUDED.rfp_description;
                """

                embedding_str = f"[{','.join(str(x) for x in customer_search_embedding)}]"

                # Insert into database using execute_with_retry for robust connection handling
                try:
                    # Define the query function to be executed with retries
                    async def execute_db_query(conn):
                        await conn.execute(
                            insert_query,
                            transaction_id,
                            embedding_str,
                            final_classification['Category_ID'],
                            final_classification.get('LLM_Description', ''),
                            sourceable_flag,
                            final_classification['Category_Description'],
                            transaction_type,
                            final_classification['Reason'],
                            single_word,
                            level_1_categories,
                            final_classification.get('level', ''),
                            searchable,
                            rfp_description
                        )
                        return True  # indicate success
                    
                    # Execute the query with our robust retry mechanism
                    result = await execute_with_retry("customer_vectors", execute_db_query, max_retries=5)
                    
                    # Check if the operation failed after all retries
                    if result is None:
                        logger.error(f"Failed to insert data for transaction {transaction_id} after all retry attempts")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                        return None
                        
                except Exception as e:
                    logger.error(f"Database error in process_transaction_with_csv for transaction {transaction_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {base_delay * (2 ** attempt)} seconds...")
                        await asyncio.sleep(base_delay * (2 ** attempt))
                        continue
                    return None

                # Write to CSV
                try:
                    await csv_writer.write_row(final_classification)
                except Exception as e:
                    logger.error(f"Error writing to CSV for transaction {transaction_id}: {e}")
                    # Don't retry on CSV write errors as they're likely to be persistent
                    return None
                
                return final_classification

            logger.warning(f"No final classification obtained for transaction {transaction_id}")
            return None

        except Exception as e:
            logger.error(f"Error in process_transaction_with_csv for transaction {row.get('Transaction_ID', 'UNKNOWN')} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                return None

    return None

async def process_transactions(
    transactions_df, 
    l3_matches: int, 
    l4_matches: int, 
    fieldnames: List[str], 
    storage,
    global_progress=None, 
    specific_temp_key=None,
    max_concurrent_tasks: int = 50  # New parameter to control concurrency
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Process a batch of transactions.
    
    Args:
        transactions_df: DataFrame of transactions
        l3_matches: Number of L3 category matches to retrieve
        l4_matches: Number of L4 category matches to retrieve
        fieldnames: CSV column names
        storage: Storage instance (S3TempStorage or DynamoDBStorage)
        global_progress: GlobalProgress instance for tracking overall progress
        specific_temp_key: Optional specific key for temporary files
        max_concurrent_tasks: Maximum number of concurrent processing tasks
        
    Returns:
        Tuple[List[Dict[str, Any]], str]: List of failed rows and temporary key
    """
    logger.debug("process_transactions called.")
    async with aiohttp.ClientSession() as session:
        embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)
        chat_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHAT_CALLS)
        task_semaphore = asyncio.Semaphore(max_concurrent_tasks)  # New semaphore for task concurrency
        openai_rate_limiter = RateLimiter(calls_per_minute=OPENAI_RATE_LIMIT_PER_MINUTE)
        jina_rate_limiter = RateLimiter(calls_per_minute=JINA_RATE_LIMIT_PER_MINUTE)

        logger.info(f"Starting processing of {len(transactions_df)} transactions with max {max_concurrent_tasks} concurrent tasks")
        
        # Use specific_temp_key if provided, otherwise generate a new key
        import time
        from ..config.config import USE_DYNAMODB
        
        temp_key = specific_temp_key if specific_temp_key else f"temp_results_{int(time.time())}_{id(transactions_df)}"
        logger.debug(f"Using temporary key: {temp_key}")
        
        # Create appropriate writer based on storage type
        if USE_DYNAMODB:
            from ..data.dynamodb_storage import DynamoDBWriter
            csv_writer = DynamoDBWriter(storage)
        else:
            from ..data.s3_storage import IncrementalCSVWriter
            csv_writer = IncrementalCSVWriter(storage, temp_key, fieldnames)
            
        failed_rows = []
        active_tasks = set()
        completed_count = 0
        total_rows = len(transactions_df)
        
        async def process_with_semaphore(row):
            async with task_semaphore:
                return await process_transaction_with_csv(
                    row, l3_matches, l4_matches,
                    session, embedding_semaphore, chat_semaphore,
                    openai_rate_limiter, jina_rate_limiter, csv_writer
                )
        
        try:
            # Process rows in chunks to control memory usage
            chunk_size = max(1, min(max_concurrent_tasks * 2, 100))  # Process in reasonable chunks
            for start_idx in range(0, len(transactions_df), chunk_size):
                chunk = transactions_df.iloc[start_idx:start_idx + chunk_size]
                chunk_tasks = []
                
                # Create tasks for the chunk
                for _, row in chunk.iterrows():
                    task = asyncio.create_task(process_with_semaphore(row))
                    chunk_tasks.append((row, task))
                    active_tasks.add(task)
                
                # Wait for chunk tasks to complete
                for row, task in chunk_tasks:
                    try:
                        result = await task
                        active_tasks.remove(task)
                        completed_count += 1
                        
                        if result:
                            if global_progress:
                                await global_progress.increment_processed()
                        else:
                            if global_progress:
                                await global_progress.increment_failed()
                            failed_rows.append(row)
                            
                        # Log progress periodically
                        if completed_count % 100 == 0:
                            logger.info(f"Processed {completed_count}/{total_rows} transactions ({(completed_count/total_rows)*100:.1f}%)")
                            
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        if global_progress:
                            await global_progress.increment_failed()
                        failed_rows.append(row)
                        if task in active_tasks:
                            active_tasks.remove(task)
                        completed_count += 1
                
        except Exception as e:
            logger.error(f"Error in process_transactions: {e}")
        finally:
            # Cancel any remaining tasks
            for task in active_tasks:
                task.cancel()
            
            # Wait for tasks to be cancelled
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            # Close the writer
            await csv_writer.close()
            
        logger.debug(f"process_transactions completed. Results stored with key {temp_key}.")
        return failed_rows, temp_key
