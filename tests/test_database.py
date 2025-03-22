"""
Tests for database functionality.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.database import (
    init_database_connections, close_database_connections, wake_up_databases,
    search_customer_collection, aurora_query, get_customer_collection_name,
    fetch_relevant_unspsc_sectors
)

class TestDatabase:
    """Tests for database functionality."""
    
    @pytest.mark.asyncio
    async def test_init_database_connections(self):
        """Test initializing database connections."""
        with patch('src.services.database.asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.return_value = MagicMock()
            
            # Execute
            result = await init_database_connections()
            
            # Verify
            assert result is True
            assert mock_create_pool.call_count == 2  # Should create two pools
    
    @pytest.mark.asyncio
    async def test_init_database_connections_error(self):
        """Test handling errors during database initialization."""
        with patch('src.services.database.asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection error")
            
            # Execute
            result = await init_database_connections()
            
            # Verify
            assert result is False
    
    @pytest.mark.asyncio
    async def test_close_database_connections(self):
        """Test closing database connections."""
        # Mock the connection pools
        with patch('src.services.database.conn_pool', new=MagicMock()) as mock_conn_pool, \
             patch('src.services.database.conn_customer_pool', new=MagicMock()) as mock_cust_pool:
            
            # Execute
            await close_database_connections()
            
            # Verify
            mock_conn_pool.close.assert_called_once()
            mock_cust_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_wake_up_databases(self):
        """Test waking up databases with simple queries."""
        # Mock connection pools and connections
        with patch('src.services.database.conn_pool') as mock_conn_pool, \
             patch('src.services.database.conn_customer_pool') as mock_cust_pool:
            
            # Setup mock connections
            mock_conn1 = AsyncMock()
            mock_conn2 = AsyncMock()
            
            # Configure mock pools to return mock connections
            mock_conn_pool.acquire.return_value.__aenter__.return_value = mock_conn1
            mock_cust_pool.acquire.return_value.__aenter__.return_value = mock_conn2
            
            # Mock query results
            mock_conn1.fetchval.return_value = 1
            mock_conn2.fetchval.return_value = 1
            
            # Execute
            result = await wake_up_databases()
            
            # Verify
            assert result is True
            mock_conn1.fetchval.assert_called_with("SELECT 1")
            mock_conn2.fetchval.assert_called_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_wake_up_databases_error(self):
        """Test handling errors during database wake-up."""
        # Mock connection pools and connections
        with patch('src.services.database.conn_pool') as mock_conn_pool:
            
            # Setup mock to raise exception
            mock_conn_pool.acquire.return_value.__aenter__.side_effect = Exception("Connection error")
            
            # Execute
            result = await wake_up_databases()
            
            # Verify
            assert result is False
    
    @pytest.mark.asyncio
    async def test_search_customer_collection(self):
        """Test searching customer collection for matching transactions."""
        customer_id = "cust123"
        embedding = [0.1] * 1024
        
        # Mock the connection pool and query
        with patch('src.services.database.conn_customer_pool') as mock_pool, \
             patch('src.services.database.get_customer_collection_name') as mock_get_name:
            
            # Configure mocks
            mock_get_name.return_value = "customer_cust123_collection"
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock query result
            mock_conn.fetchrow.return_value = {
                "transaction_id": "tx123",
                "score": 0.95,
                "unspsc_id": "43211500",
                "cleansed_description": "Computer",
                "category_description": "Computers",
                "transaction_type": "Indirect Goods",
                "llm_reasoning": "Best match",
                "single_word": "computer",
                "level_1_categories": [43, 44, 45],
                "level": "L3",
                "searchable": True,
                "rfp_description": "Desktop computer"
            }
            
            # Execute
            result = await search_customer_collection(customer_id, embedding)
        
        # Verify
        assert result is not None
        assert result['score'] == 0.95
        assert result['metadata']['unspsc_id'] == "43211500"
        assert result['metadata']['category_description'] == "Computers"
        
        # Verify query execution
        mock_conn.fetchrow.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_aurora_query(self):
        """Test querying Aurora database for vector search."""
        embedding = [0.1] * 1024
        query_filter = {"level": "L3"}
        
        # Mock the connection pool and query
        with patch('src.services.database.conn_pool') as mock_pool:
            # Configure mocks
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock query result
            mock_conn.fetch.return_value = [
                {
                    "id": 1,
                    "unspsc_id": "43211500",
                    "description": "Computers",
                    "level": "L3",
                    "level_1_lookup": [43, 44, 45],
                    "score": 0.95
                }
            ]
            
            # Execute
            result = await aurora_query(embedding, top_k=10, query_filter=query_filter)
        
        # Verify
        assert result is not None
        assert len(result['matches']) == 1
        assert result['matches'][0]['metadata']['UNSPSC_ID'] == "43211500"
        assert result['matches'][0]['metadata']['Description'] == "Computers"
        assert result['matches'][0]['score'] == 0.95
        
        # Verify query execution
        mock_conn.fetch.assert_called_once()
    
    def test_get_customer_collection_name(self):
        """Test generating customer collection table name."""
        customer_id = "cust-123"
        
        # Execute
        table_name = get_customer_collection_name(customer_id)
        
        # Verify
        assert table_name == "customer_cust_123_collection"
    
    @pytest.mark.asyncio
    async def test_fetch_relevant_unspsc_sectors(self):
        """Test fetching relevant UNSPSC sectors based on industry filter."""
        industry_filter = 1
        
        # Mock the connection pool and query
        with patch('src.services.database.conn_pool') as mock_pool:
            # Configure mocks
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock query result
            mock_conn.fetch.return_value = [
                {"unspsc_sector": 43, "unspsc_sector_description": "IT Equipment"},
                {"unspsc_sector": 44, "unspsc_sector_description": "Office Equipment"}
            ]
            
            # Execute
            result = await fetch_relevant_unspsc_sectors(industry_filter)
        
        # Verify
        assert result is not None
        assert len(result) == 2
        assert result[0]["unspsc_sector"] == 43
        assert result[0]["unspsc_sector_description"] == "IT Equipment"
        
        # Verify query execution
        mock_conn.fetch.assert_called_once()