"""
Tests for the classifier module.
"""
import pytest
import pandas as pd
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.models.classifier import process_transaction, process_transaction_with_csv, process_transactions
from src.utils.rate_limiter import RateLimiter

class TestClassifier:
    """Tests for the classifier functionality."""
    
    @pytest.mark.asyncio
    async def test_process_transaction(self):
        """Test processing a single transaction."""
        # Setup mocks
        row = {
            'Transaction_ID': '123',
            'Supplier_Name': 'Test Supplier',
            'Transaction_Description': 'Test Purchase',
            'Transaction_Value': 1000,
            'customer_industry_description': 'IT'
        }
        
        session = MagicMock()
        embedding_semaphore = asyncio.Semaphore(1)
        chat_semaphore = asyncio.Semaphore(1)
        rate_limiter = MagicMock(spec=RateLimiter)
        
        # Mock aurora_query
        with patch('src.models.classifier.aurora_query') as mock_aurora:
            # First call returns L3 matches
            mock_aurora.return_value = {
                'matches': [
                    {
                        'score': 0.9,
                        'metadata': {
                            'UNSPSC_ID': '43211500',
                            'Description': 'Computers',
                            'level': 'L3'
                        }
                    }
                ]
            }
            
            # Mock select_best_match
            with patch('src.models.classifier.select_best_match') as mock_select:
                mock_select.return_value = {
                    'Transaction_ID': '123',
                    'Supplier_Name': 'Test Supplier',
                    'Transaction_Description': 'Test Purchase',
                    'LLM_Description': 'Computer',
                    'Transaction_Value': 1000,
                    'Category_ID': '43211500',
                    'Category_Description': 'Computers',
                    'level': 'L3',
                    'Reason': 'Best match',
                    'Confidence_Score': 8
                }
                
                # Execute
                result = await process_transaction(
                    row,
                    l3_matches=25,
                    l4_matches=25,
                    session=session,
                    embedding_semaphore=embedding_semaphore,
                    chat_semaphore=chat_semaphore,
                    rate_limiter=rate_limiter,
                    llm_description='Computer',
                    embedding=[0.1] * 1024
                )
        
        # Verify
        assert result is not None
        assert result['Transaction_ID'] == '123'
        assert result['Category_ID'] == '43211500'
        assert result['Category_Description'] == 'Computers'
        assert result['level'] == 'L3'
        
        # Verify aurora_query was called
        mock_aurora.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_transaction_with_csv(self):
        """Test processing a transaction and writing to CSV."""
        # Setup mocks
        row = pd.Series({
            'Transaction_ID': '123',
            'Supplier_Name': 'Test Supplier',
            'Transaction_Description': 'Test Purchase',
            'primary_descriptor': 'Office Computer Equipment',
            'alternate_descriptor_1': 'Desktop PC',
            'alternate_descriptor_2': 'Intel i7 Processor',
            'Transaction_Value': 1000,
            'customer_id': 'cust123',
            'customer_industry_description': 'IT'
        })
        
        session = MagicMock()
        embedding_semaphore = asyncio.Semaphore(1)
        chat_semaphore = asyncio.Semaphore(1)
        openai_rate_limiter = MagicMock(spec=RateLimiter)
        jina_rate_limiter = MagicMock(spec=RateLimiter)
        csv_writer = MagicMock()
        # Make write_row a coroutine
        csv_writer.write_row = AsyncMock()
        
        # Mock search_customer_collection to return no previous match
        with patch('src.models.classifier.search_customer_collection') as mock_search:
            mock_search.return_value = None
            
            # Mock get_embedding
            with patch('src.models.classifier.get_embedding') as mock_get_embedding:
                mock_get_embedding.return_value = [0.1] * 1024
                
                # Mock classify_transaction_initial
                with patch('src.models.classifier.classify_transaction_initial') as mock_classify:
                    mock_classify.return_value = {
                        'cleansed_description': 'Computer',
                        'sourceable_flag': True,
                        'transaction_type': 'Indirect Goods',
                        'single_word': 'computer',
                        'searchable': True,
                        'rfp_description': 'Desktop computer'
                    }
                    
                    # Mock process_transaction
                    with patch('src.models.classifier.process_transaction') as mock_process:
                        mock_process.return_value = {
                            'Transaction_ID': '123',
                            'Supplier_Name': 'Test Supplier',
                            'Transaction_Description': 'Test Purchase',
                            'LLM_Description': 'Computer',
                            'Transaction_Value': 1000,
                            'Category_ID': '43211500',
                            'Category_Description': 'Computers',
                            'level': 'L3',
                            'Reason': 'Best match',
                            'Confidence_Score': 8,
                            'Embedding_Matches': 'Test matches',
                            'Level_1_Categories': [43, 44, 45]
                        }
                        
                        # Mock execute_with_retry instead of directly patching the connection pool
                        with patch('src.models.classifier.execute_with_retry') as mock_execute:
                            mock_execute.return_value = True
                            # No need for mock_conn anymore as we're mocking execute_with_retry directly
                            
                            # Execute
                            result = await process_transaction_with_csv(
                                row,
                                l3_matches=25,
                                l4_matches=25,
                                session=session,
                                embedding_semaphore=embedding_semaphore,
                                chat_semaphore=chat_semaphore,
                                openai_rate_limiter=openai_rate_limiter,
                                jina_rate_limiter=jina_rate_limiter,
                                csv_writer=csv_writer
                            )
        
        # Verify
        assert result is not None
        assert result['Transaction_ID'] == '123'
        assert result['Category_ID'] == '43211500'
        assert result['Category_Description'] == 'Computers'
        assert result['level'] == 'L3'
        assert result['transaction_type'] == 'Indirect Goods'
        assert result['sourceable_flag'] is True
        
        # Verify CSV writer was called
        csv_writer.write_row.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_transactions(self):
        """Test processing a batch of transactions."""
        # Simple test data
        df_data = {
            'Transaction_ID': ['123', '456'],
            'Supplier_Name': ['Supplier1', 'Supplier2'],
            'Transaction_Description': ['Purchase1', 'Purchase2'],
            'Transaction_Value': [1000, 2000],
            'customer_id': ['cust1', 'cust2'],
            'primary_descriptor': ['Computer', 'Office'],
            'alternate_descriptor_1': ['Desktop', 'Supplies'],
            'alternate_descriptor_2': ['PC', 'Paper'],
            'customer_industry_description': ['IT', 'Finance']
        }
        
        test_df = pd.DataFrame(df_data)
        field_names = ['Transaction_ID', 'Category_ID', 'Category_Description']
        
        # Create test resources
        temp_storage = MagicMock()
        global_progress = MagicMock()
        global_progress.increment_processed = AsyncMock()
        global_progress.increment_failed = AsyncMock()
        
        # Mock CSV writer
        csv_writer_mock = MagicMock()
        csv_writer_mock.write_row = AsyncMock()
        csv_writer_mock.close = AsyncMock()
        
        # Mock transaction processing
        with patch('src.models.classifier.process_transaction_with_csv') as mock_process:
            # Side effect to simulate first success, second failure
            async def mock_process_side_effect(row, *args, **kwargs):
                if row['Transaction_ID'] == '123':
                    return {'Transaction_ID': '123', 'Category_ID': '43211500'}
                return None
            
            mock_process.side_effect = mock_process_side_effect
            
            # Mock S3 storage writer
            with patch('src.data.s3_storage.IncrementalCSVWriter', return_value=csv_writer_mock):
                # Mock config
                with patch('src.config.config.USE_DYNAMODB', False):
                    # Execute the function
                    failed_rows, temp_key = await process_transactions(
                        test_df,
                        l3_matches=25,
                        l4_matches=25,
                        fieldnames=field_names,
                        storage=temp_storage,
                        global_progress=global_progress
                    )
        
        # Verify
        assert len(failed_rows) == 1  # One failed row (the second one)
        assert csv_writer_mock.close.called
        assert global_progress.increment_processed.call_count == 1
        assert global_progress.increment_failed.call_count == 1
