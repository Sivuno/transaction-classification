import asyncpg
import logging
import sys
from typing import List, Dict, Any, Optional
from ..config.config import (
    RDS_USERNAME, RDS_PASSWORD, RDS_PORT, RDS_HOST,
    RDS_MASTER_TABLES_CLUSTER, RDS_MASTER_TABLES_DATABASE
)
import asyncio

logger = logging.getLogger(__name__)

# Global connection pools
conn_unspsc_pool = None
conn_customer_pool = None
conn_master_tables_pool = None

async def init_database_connections() -> bool:
    """
    Initialize all database connections.
    
    Returns:
        bool: True if all connections were successful, False otherwise
    """
    global conn_unspsc_pool
    global conn_customer_pool
    global conn_master_tables_pool
    
    success = True
    
    try:
        logger.info("Connecting to Aurora for UNSPSC vectors using asyncpg...")
        conn_unspsc_pool = await asyncpg.create_pool(
            host=RDS_HOST,
            database="unspsc_vectors",
            user=RDS_USERNAME,
            password=RDS_PASSWORD,
            port=RDS_PORT,
            min_size=1,
            max_size=50
        )
        logger.info("UNSPSC vectors database connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to UNSPSC vectors database: {e}")
        success = False
    
    try:
        logger.info("Connecting to Aurora for customer vectors using asyncpg...")
        if RDS_HOST is None or RDS_USERNAME is None or RDS_PASSWORD is None:
            logger.error("Missing required database credentials for customer vectors database")
            success = False
        else:
            conn_customer_pool = await asyncpg.create_pool(
                host=RDS_HOST,
                database="customer_vectors",
                user=RDS_USERNAME,
                password=RDS_PASSWORD,
                port=RDS_PORT,
                min_size=1,
                max_size=50
            )
            if conn_customer_pool is None:
                logger.error("Customer vectors connection pool is None after creation")
                success = False
            else:
                logger.info("Customer vectors database connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to customer vectors database: {e}")
        success = False
    
    try:
        logger.info("Connecting to master_tables DB...")
        conn_master_tables_pool = await asyncpg.create_pool(
            host=RDS_MASTER_TABLES_CLUSTER,
            database=RDS_MASTER_TABLES_DATABASE,
            user=RDS_USERNAME,
            password=RDS_PASSWORD,
            port=RDS_PORT,
            min_size=1,
            max_size=10
        )
        logger.info("Master tables database connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to master_tables DB: {e}")
        success = False
    
    if not success:
        logger.error("One or more database connections failed to initialize")
    else:
        logger.info("All database connections initialized successfully")
    
    return success

async def create_connection_pool(pool_name: str) -> Optional[asyncpg.Pool]:
    """
    Create a new connection pool with the appropriate settings.
    
    Args:
        pool_name: Name of the pool to create
        
    Returns:
        Optional[asyncpg.Pool]: The created connection pool or None if failed
    """
    try:
        if pool_name == "customer_vectors":
            return await asyncpg.create_pool(
                host=RDS_HOST,
                database="customer_vectors",
                user=RDS_USERNAME,
                password=RDS_PASSWORD,
                port=RDS_PORT,
                min_size=1,
                max_size=20
            )
        elif pool_name == "unspsc_vectors":
            return await asyncpg.create_pool(
                host=RDS_HOST,
                database="unspsc_vectors",
                user=RDS_USERNAME,
                password=RDS_PASSWORD,
                port=RDS_PORT,
                min_size=1,
                max_size=20
            )
        elif pool_name == "master_tables":
            return await asyncpg.create_pool(
                host=RDS_MASTER_TABLES_CLUSTER,
                database=RDS_MASTER_TABLES_DATABASE,
                user=RDS_USERNAME,
                password=RDS_PASSWORD,
                port=RDS_PORT,
                min_size=1,
                max_size=10
            )
        else:
            logger.error(f"Unknown pool name: {pool_name}")
            return None
    except Exception as e:
        logger.error(f"Failed to create connection pool for {pool_name}: {e}")
        return None

async def check_connection_pool_health(pool, pool_name: str) -> bool:
    """
    Check if a connection pool is healthy.
    
    Args:
        pool: The connection pool to check
        pool_name: Name of the pool for logging
        
    Returns:
        bool: True if pool is healthy, False otherwise
    """
    if pool is None:
        return False
        
    try:
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Health check failed for {pool_name}: {e}")
        return False

async def ensure_connection_pool(pool_name: str) -> bool:
    """
    Ensure a connection pool exists and is healthy, creating or recreating it if necessary.
    
    Args:
        pool_name: Name of the pool to ensure
        
    Returns:
        bool: True if pool is healthy after checks/recreation, False otherwise
    """
    global conn_customer_pool, conn_unspsc_pool, conn_master_tables_pool
    max_retries = 5  # Increased from 3 to 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Get the current pool reference
            current_pool = None
            if pool_name == "customer_vectors":
                current_pool = conn_customer_pool
            elif pool_name == "unspsc_vectors":
                current_pool = conn_unspsc_pool
            elif pool_name == "master_tables":
                current_pool = conn_master_tables_pool
            else:
                logger.error(f"Unknown pool name: {pool_name}")
                return False
            
            # Check if current pool is healthy
            if current_pool is not None:
                is_healthy = await check_connection_pool_health(current_pool, pool_name)
                if is_healthy:
                    return True
                    
                # Log more details about the unhealthy pool
                logger.warning(f"Pool {pool_name} is not healthy, will close and recreate")
                
                # If not healthy, close it before recreating
                try:
                    await current_pool.close()
                    logger.info(f"Successfully closed unhealthy pool {pool_name}")
                except Exception as e:
                    logger.warning(f"Error closing unhealthy pool {pool_name}: {e}")
            else:
                logger.warning(f"Pool {pool_name} is None, will create a new pool")
            
            # Create new pool
            logger.info(f"Creating new connection pool for {pool_name} (attempt {attempt + 1}/{max_retries})")
            new_pool = await create_connection_pool(pool_name)
            
            if new_pool is None:
                logger.error(f"Failed to create new pool for {pool_name}")
                if attempt < max_retries - 1:
                    retry_delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                return False
                
            # Verify new pool is healthy
            is_healthy = await check_connection_pool_health(new_pool, pool_name)
            if not is_healthy:
                logger.error(f"New pool for {pool_name} is not healthy after creation")
                try:
                    await new_pool.close()
                    logger.info(f"Closed unhealthy new pool for {pool_name}")
                except Exception as e:
                    logger.warning(f"Error closing unhealthy new pool for {pool_name}: {e}")
                
                if attempt < max_retries - 1:
                    retry_delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                return False
            
            # Assign new pool to global variable
            if pool_name == "customer_vectors":
                conn_customer_pool = new_pool
            elif pool_name == "unspsc_vectors":
                conn_unspsc_pool = new_pool
            elif pool_name == "master_tables":
                conn_master_tables_pool = new_pool
                
            logger.info(f"Successfully created and verified new pool for {pool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring connection pool for {pool_name} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                retry_delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                return False
    
    return False

async def wake_up_databases():
    """
    Ping each Aurora database pool to ensure they are awake.
    
    Returns:
        bool: True if all databases are awake, False otherwise
    """
    logger.info("Waking up Aurora databases...")
    try:
        # Ensure each pool is healthy
        unspsc_healthy = await ensure_connection_pool("unspsc_vectors")
        customer_healthy = await ensure_connection_pool("customer_vectors")
        master_healthy = await ensure_connection_pool("master_tables")
        
        if not all([unspsc_healthy, customer_healthy, master_healthy]):
            logger.error("One or more database pools are unhealthy")
            return False
            
        logger.info("All databases are awake and ready for processing.")
        return True
    except Exception as e:
        logger.error(f"Error waking up databases: {e}")
        return False
        
# Keep-alive task for database connections
heartbeat_task = None

async def _heartbeat_queries():
    """
    Background task to periodically ping database connections to keep them alive.
    """
    global conn_customer_pool, conn_unspsc_pool, conn_master_tables_pool
    
    logger.info("Starting database heartbeat task")
    heartbeat_interval = 20  # Check every 20 seconds (increased frequency)
    
    while True:
        try:
            # Check health of all pools
            pools_to_check = ["unspsc_vectors", "customer_vectors", "master_tables"]
            for pool_name in pools_to_check:
                # First check if the pool exists and is already in a healthy state
                pool = None
                if pool_name == "customer_vectors":
                    pool = conn_customer_pool
                elif pool_name == "unspsc_vectors":
                    pool = conn_unspsc_pool
                elif pool_name == "master_tables":
                    pool = conn_master_tables_pool
                
                # If pool is None, proactively recreate it
                if pool is None:
                    logger.warning(f"Heartbeat detected {pool_name} pool is None - recreating")
                    success = await ensure_connection_pool(pool_name)
                    if success:
                        logger.info(f"Heartbeat successfully recreated {pool_name} connection pool")
                        # Update our reference to the pool
                        if pool_name == "customer_vectors":
                            pool = conn_customer_pool
                        elif pool_name == "unspsc_vectors":
                            pool = conn_unspsc_pool
                        elif pool_name == "master_tables":
                            pool = conn_master_tables_pool
                    else:
                        logger.error(f"Heartbeat failed to recreate {pool_name} connection pool")
                    continue
                
                # Check pool health directly
                is_healthy = await check_connection_pool_health(pool, pool_name)
                if not is_healthy:
                    logger.warning(f"Heartbeat detected unhealthy {pool_name} pool - recreating")
                    
                    # Explicitly close the unhealthy pool
                    try:
                        await pool.close()
                        logger.info(f"Heartbeat closed unhealthy {pool_name} pool")
                    except Exception as e:
                        logger.warning(f"Heartbeat error closing unhealthy {pool_name} pool: {e}")
                    
                    # Clear the global reference
                    if pool_name == "customer_vectors":
                        conn_customer_pool = None
                    elif pool_name == "unspsc_vectors":
                        conn_unspsc_pool = None
                    elif pool_name == "master_tables":
                        conn_master_tables_pool = None
                    
                    # Create a new pool
                    success = await ensure_connection_pool(pool_name)
                    if success:
                        logger.info(f"Heartbeat successfully recreated {pool_name} connection pool")
                        # Update our reference to the pool
                        if pool_name == "customer_vectors":
                            pool = conn_customer_pool
                        elif pool_name == "unspsc_vectors":
                            pool = conn_unspsc_pool
                        elif pool_name == "master_tables":
                            pool = conn_master_tables_pool
                    else:
                        logger.error(f"Heartbeat failed to recreate {pool_name} connection pool")
                    continue
                
                # Use the execute_with_retry to verify query execution works
                async def simple_query(conn):
                    return await conn.fetchval("SELECT 1")
                    
                try:
                    result = await execute_with_retry(pool_name, simple_query)
                    if result == 1:
                        logger.debug(f"Heartbeat for {pool_name} successful")
                    else:
                        logger.warning(f"Heartbeat for {pool_name} returned unexpected result: {result}")
                except Exception as e:
                    logger.error(f"Heartbeat query failed for {pool_name}: {e}")
                    
            # Wait before next heartbeat
            await asyncio.sleep(heartbeat_interval)
        except asyncio.CancelledError:
            logger.info("Database heartbeat task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in database heartbeat task: {e}")
            await asyncio.sleep(heartbeat_interval)  # Still wait before retry even after error

async def start_heartbeat():
    """
    Start the database heartbeat task.
    """
    global heartbeat_task
    if heartbeat_task is None or heartbeat_task.done():
        heartbeat_task = asyncio.create_task(_heartbeat_queries())
        logger.info("Database heartbeat task started")
    else:
        logger.warning("Heartbeat task already running")

async def stop_heartbeat():
    """
    Stop the database heartbeat task.
    """
    global heartbeat_task
    if heartbeat_task and not heartbeat_task.done():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("Database heartbeat task stopped")
    else:
        logger.info("No running heartbeat task to stop")

async def close_database_connections():
    """Close all database connections."""
    logger.info("Closing database connections...")
    
    if conn_unspsc_pool:
        await conn_unspsc_pool.close()
        logger.info("UNSPSC vectors database connection closed")
    
    if conn_customer_pool:
        await conn_customer_pool.close()
        logger.info("Customer vectors database connection closed")
    
    if conn_master_tables_pool:
        await conn_master_tables_pool.close()
        logger.info("Master tables database connection closed")

def get_customer_collection_name(customer_id: str) -> str:
    """
    Generate a normalized table name from a customer ID.
    
    Args:
        customer_id (str): Customer ID
        
    Returns:
        str: Normalized table name
    """
    logger.debug(f"Generating customer collection name (table name) for customer_id: {customer_id}")
    # The table name in Aurora is the same as the customer_id, e.g., gra001
    return customer_id.lower()

async def fetch_relevant_unspsc_sectors(industry_filter_value: int) -> List[Dict[str, Any]]:
    """
    For the given 'industry_filter_value', query the unspsc_industry_mapping table 
    in the master_tables DB and return all rows where this value is in the 'industries' array.
    
    Args:
        industry_filter_value (int): Industry filter value
        
    Returns:
        List[Dict[str, Any]]: List of matching UNSPSC sectors
    """
    # Define the query function to be executed with retries
    async def query_function(conn):
        query = """
            SELECT unspsc_sector::int AS unspsc_sector, unspsc_sector_description
            FROM unspsc_industry_mapping
            WHERE $1 = ANY(industries);
        """
        rows = await conn.fetch(query, industry_filter_value)
        return [dict(r) for r in rows]
    
    # Execute the query with retry logic
    try:
        result = await execute_with_retry("master_tables", query_function)
        return result if result is not None else []
    except Exception as e:
        logger.error(f"Error querying unspsc_industry_mapping for filter={industry_filter_value}: {e}")
        return []

async def aurora_query(embedding: List[float], top_k: int, query_filter: Dict[str, Any]):
    """
    Execute a vector similarity query against the unspsc_categories table in Aurora (pgvector).

    Args:
        embedding: Vector embedding for similarity search.
        top_k: Number of results to return.
        query_filter: A dict with filter conditions (e.g., { "level": "L3", "level_1_lookup": [1, 2, 3] }).
        
    Returns:
        dict: Dictionary with matches
    """
    logger.debug(f"aurora_query (Aurora) called with top_k={top_k} and filter={query_filter}")
    
    # Convert the Python embedding list to a pgvector literal: '[x1, x2, ...]'
    vector_str = f"[{','.join(str(x) for x in embedding)}]"
    
    # Define the query function that will execute with retry logic
    async def query_function(conn):
        # Build WHERE clause from query_filter
        where_clauses = []
        if query_filter:
            if "level" in query_filter:
                where_clauses.append(f"level = '{query_filter['level']}'")
            if "level_1_lookup" in query_filter and query_filter["level_1_lookup"]:
                categories_str = ", ".join(str(x) for x in query_filter["level_1_lookup"])
                where_clauses.append(f"level_1_lookup::int IN ({categories_str})")

        base_query = """
            SELECT
                unspsc_id,
                description,
                level,
                type,
                level_1_lookup,
                (embedding <-> $1::vector) as distance
            FROM unspsc_categories
        """
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        base_query += f" ORDER BY embedding <-> $1::vector LIMIT {top_k};"

        logger.debug(f"Executing UNSPSC vector search query:\n{base_query}")

        # Use parameter $1 for vector_str
        rows = await conn.fetch(base_query, vector_str)

        if not rows:
            logger.debug("No results found from Aurora UNSPSC vector query.")
            return {'matches': []}

        matches = []
        for row in rows:
            dist = float(row['distance'])
            matches.append({
                'metadata': {
                    'UNSPSC_ID': row['unspsc_id'],
                    'Description': row['description'],
                    'level': row['level'],
                    'Type': row['type'],
                    'Level_1_Lookup': row['level_1_lookup']
                },
                'score': 1.0 - dist
            })
        logger.debug(f"Aurora UNSPSC query returned {len(matches)} matches.")
        return {'matches': matches}
    
    # Execute the query with retry logic
    try:
        result = await execute_with_retry("unspsc_vectors", query_function, max_retries=5)
        return result if result is not None else {'matches': []}
    except Exception as e:
        logger.error(f"Failed to execute UNSPSC query after multiple attempts: {e}")
        return {'matches': []}

async def execute_with_retry(pool_name: str, query_func, max_retries=5):
    """
    Execute a database query with automatic retry if the connection fails.
    
    Args:
        pool_name (str): The name of the connection pool to use
        query_func: An async function that takes a connection and performs the query
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        The result of the query function, or None if all attempts fail
    """
    global conn_customer_pool, conn_unspsc_pool, conn_master_tables_pool
    
    # Get the appropriate pool object
    pool = None
    if pool_name == "customer_vectors":
        pool = conn_customer_pool
    elif pool_name == "unspsc_vectors":
        pool = conn_unspsc_pool
    elif pool_name == "master_tables":
        pool = conn_master_tables_pool
    else:
        logger.error(f"Unknown pool name: {pool_name}")
        return None
    
    for attempt in range(max_retries):
        try:
            # Check if pool is None or unhealthy before trying
            if pool is None:
                logger.error(f"{pool_name} connection pool is None - attempting to reconnect (attempt {attempt+1}/{max_retries})")
                # Always recreate the pool on None
                success = await ensure_connection_pool(pool_name)
                if not success:
                    logger.error(f"Failed to recreate {pool_name} pool (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        retry_delay = 1 * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return None
                
                # Update our pool reference after reconnection
                if pool_name == "customer_vectors":
                    pool = conn_customer_pool
                elif pool_name == "unspsc_vectors":
                    pool = conn_unspsc_pool
                elif pool_name == "master_tables":
                    pool = conn_master_tables_pool
                
                logger.info(f"Successfully recreated {pool_name} pool")
            
            # Double-check that the pool is not None after trying to recreate it
            if pool is None:
                logger.error(f"{pool_name} pool is still None after recreation attempt")
                if attempt < max_retries - 1:
                    retry_delay = 1 * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    return None
            
            # Check pool health before executing query
            is_healthy = await check_connection_pool_health(pool, pool_name)
            if not is_healthy:
                logger.error(f"Connection pool is unhealthy before query execution")
                # Force recreate the pool - don't reuse an unhealthy pool
                try:
                    # Try to close the unhealthy pool first
                    await pool.close()
                    logger.info(f"Closed unhealthy {pool_name} pool")
                except Exception as e:
                    logger.warning(f"Error closing unhealthy {pool_name} pool: {e}")
                
                # Set to None to force recreation
                if pool_name == "customer_vectors":
                    conn_customer_pool = None
                elif pool_name == "unspsc_vectors":
                    conn_unspsc_pool = None
                elif pool_name == "master_tables":
                    conn_master_tables_pool = None
                
                # Recreate the pool
                success = await ensure_connection_pool(pool_name)
                if not success:
                    logger.error(f"Failed to recreate {pool_name} pool after health check")
                    if attempt < max_retries - 1:
                        retry_delay = 1 * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return None
                
                # Update our pool reference after reconnection
                if pool_name == "customer_vectors":
                    pool = conn_customer_pool
                elif pool_name == "unspsc_vectors":
                    pool = conn_unspsc_pool
                elif pool_name == "master_tables":
                    pool = conn_master_tables_pool
                
                # One more sanity check
                if pool is None:
                    logger.error(f"{pool_name} pool is still None after recreating in health check")
                    if attempt < max_retries - 1:
                        retry_delay = 1 * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return None
            
            # Execute the query function
            try:
                async with pool.acquire() as conn:
                    result = await query_func(conn)
                    return result
            except Exception as e:
                # Check if this is a pool acquire timeout error
                if "TimeoutError" in str(e) or "timeout" in str(e).lower() or "acquire" in str(e).lower():
                    logger.error(f"Timeout acquiring connection from pool (attempt {attempt+1}/{max_retries}): {e}")
                    # Force recreate the pool on acquire timeout
                    try:
                        await pool.close()
                    except Exception:
                        pass
                    
                    if pool_name == "customer_vectors":
                        conn_customer_pool = None
                    elif pool_name == "unspsc_vectors":
                        conn_unspsc_pool = None
                    elif pool_name == "master_tables":
                        conn_master_tables_pool = None
                    
                    if attempt < max_retries - 1:
                        retry_delay = 1 * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return None
                else:
                    # Re-raise the exception to be caught by the outer exception handlers
                    raise
                
        except (asyncpg.exceptions.PostgresConnectionError, 
                asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.ConnectionFailureError) as e:
            logger.error(f"Connection error during query execution (attempt {attempt+1}/{max_retries}): {e}")
            
            # Force close and recreate the pool on any connection error
            try:
                if pool is not None:
                    await pool.close()
            except Exception as close_error:
                logger.warning(f"Error closing pool after connection error: {close_error}")
            
            # Set to None to force recreation
            if pool_name == "customer_vectors":
                conn_customer_pool = None
            elif pool_name == "unspsc_vectors":
                conn_unspsc_pool = None
            elif pool_name == "master_tables":
                conn_master_tables_pool = None
            
            # Try to ensure connection pool is healthy
            success = await ensure_connection_pool(pool_name)
            if not success:
                logger.error(f"Failed to reconnect the {pool_name} pool after connection error")
            
            # Update our pool reference after reconnection
            if pool_name == "customer_vectors":
                pool = conn_customer_pool
            elif pool_name == "unspsc_vectors":
                pool = conn_unspsc_pool
            elif pool_name == "master_tables":
                pool = conn_master_tables_pool
                
            if attempt < max_retries - 1:
                retry_delay = 1 * (2 ** attempt)
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to execute query after {max_retries} attempts")
                return None
                
        except Exception as e:
            logger.error(f"Error executing query (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                retry_delay = 1 * (2 ** attempt)
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to execute query after {max_retries} attempts")
                return None
    
    return None

async def search_customer_collection(customer_id: str, embedding: List[float], top_k: int = 5) -> Optional[Dict[str, Any]]:
    """
    Search for similar transactions in the customer's collection using HNSW index with cosine similarity.
    Returns matches with similarity >= 0.95 (very close matches).
    
    Args:
        customer_id (str): Customer ID
        embedding (List[float]): Vector embedding for similarity search
        top_k (int): Number of results to return
        
    Returns:
        Optional[Dict[str, Any]]: Best matching transaction or None
    """
    logger.debug(f"search_customer_collection called for customer_id {customer_id}")
    
    if not customer_id or not embedding:
        logger.error("Missing required parameters: customer_id or embedding")
        return None

    table_name = get_customer_collection_name(customer_id)
    vector_str = f"[{','.join(str(x) for x in embedding)}]"
    
    # Define the query function to be executed with retries
    async def query_function(conn):
        # Set the ef search parameter for better recall
        await conn.execute("SET LOCAL hnsw.ef_search = 100;")
        
        query = f"""
            WITH similarity_matches AS (
                SELECT
                    transaction_id,
                    unspsc_id,
                    cleansed_description,
                    sourceable_flag,
                    category_description,
                    transaction_type,
                    llm_reasoning,
                    single_word,
                    level_1_categories,
                    level,
                    searchable,
                    rfp_description,
                    1 - (embedding <=> $1::vector) as similarity
                FROM {table_name}
                WHERE 1 - (embedding <=> $1::vector) >= 0.95
                ORDER BY embedding <=> $1::vector
                LIMIT {top_k}
            )
            SELECT * FROM similarity_matches
            WHERE similarity >= 0.95
            ORDER BY similarity DESC
            LIMIT 1;
        """

        match = await conn.fetchrow(query, vector_str)

        if match:
            similarity_score = float(match['similarity'])
            logger.debug(f"Found match with similarity score: {similarity_score}")
            
            return {
                'metadata': {
                    'unspsc_id': match['unspsc_id'],
                    'cleansed_description': match['cleansed_description'],
                    'matched_transaction_id': match['transaction_id'],
                    'sourceable_flag': match['sourceable_flag'],
                    'category_description': match['category_description'],
                    'transaction_type': match['transaction_type'],
                    'llm_reasoning': match['llm_reasoning'],
                    'single_word': match['single_word'],
                    'level_1_categories': match['level_1_categories'],
                    'level': match['level'],
                    'searchable': match['searchable'],
                    'rfp_description': match['rfp_description']
                },
                'score': similarity_score
            }
        
        logger.debug("No matches found above similarity threshold 0.95")
        return None

    # Execute the query with retry logic
    try:
        result = await execute_with_retry("customer_vectors", query_function)
        return result
    except Exception as e:
        logger.error(f"Error searching customer collection: {e}")
        return None