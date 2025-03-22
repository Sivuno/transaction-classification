"""
Tests for API service functionality.
"""
import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.api import get_embedding, classify_transaction_initial, select_best_match, get_level1_categories
from src.utils.rate_limiter import RateLimiter

class TestAPIService:
    """Tests for the API service functions."""
    
    @pytest.mark.asyncio
    async def test_get_embedding(self):
        """Test getting embeddings from Jina API."""
        # Setup mocks
        semaphore = asyncio.Semaphore(1)
        rate_limiter = MagicMock(spec=RateLimiter)
        rate_limiter.acquire = AsyncMock()
        
        # Mock response data
        response_data = {'data': [{'embedding': [0.1, 0.2, 0.3, 0.4] * 256}]}  # 1024 dimensions
        
        # Mock the ClientSession and response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=response_data)
        
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)
        
        # Mock ClientSession constructor
        with patch('aiohttp.ClientSession', MagicMock(return_value=mock_session)):
            # Execute
            embedding = await get_embedding("test text", semaphore, rate_limiter)
        
        # Verify result
        assert embedding is not None
        assert len(embedding) == 1024
        assert embedding[0] == 0.1
        assert embedding[1] == 0.2
        
        # Verify rate limiter was used
        rate_limiter.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embedding_error(self):
        """Test error handling in get_embedding."""
        # Setup mocks
        semaphore = asyncio.Semaphore(1)
        rate_limiter = MagicMock(spec=RateLimiter)
        
        # Mock aiohttp.ClientSession
        session_mock = MagicMock()
        response_mock = MagicMock()
        
        # Setup error response
        response_mock.status = 500
        response_mock.text.return_value = "Internal Server Error"
        
        session_mock.__aenter__.return_value = session_mock
        session_mock.post.return_value.__aenter__.return_value = response_mock
        
        # Execute with patched session
        with patch('aiohttp.ClientSession', return_value=session_mock):
            embedding = await get_embedding("test text", semaphore, rate_limiter)
        
        # Verify result
        assert embedding is None
    
    @pytest.mark.asyncio
    async def test_classify_transaction_initial(self):
        """Test initial transaction classification."""
        # Setup
        session = MagicMock()
        semaphore = MagicMock()
        rate_limiter = MagicMock()
        
        # Mock response from OpenAI
        response_mock = MagicMock()
        response_mock.status = 200
        response_mock.json.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'cleansed_description': 'Test product',
                        'sourceable_flag': True,
                        'single_word': 'test',
                        'transaction_type': 'Indirect Goods',
                        'searchable': False,
                        'rfp_description': 'Test product description'
                    })
                }
            }]
        }
        
        session.post.return_value.__aenter__.return_value = response_mock
        
        # Execute
        result = await classify_transaction_initial(
            session, 
            'Test Supplier',
            'Test Description',
            'Test Industry',
            semaphore,
            rate_limiter
        )
        
        # Verify
        assert result is not None
        assert result['cleansed_description'] == 'Test product'
        assert result['sourceable_flag'] is True
        assert result['single_word'] == 'test'
        assert result['transaction_type'] == 'Indirect Goods'
        
        # Verify request
        session.post.assert_called_once()
        rate_limiter.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_select_best_match(self):
        """Test selecting the best match from candidates."""
        # Setup
        session = MagicMock()
        semaphore = MagicMock()
        rate_limiter = MagicMock()
        
        # Candidate matches
        l3_matches = [
            {'UNSPSC_ID': '43211500', 'Description': 'Computers', 'level': 'L3', 'Score': 0.85}
        ]
        l4_matches = [
            {'UNSPSC_ID': '43211501', 'Description': 'Desktop computers', 'level': 'L4', 'Score': 0.75}
        ]
        
        # Mock response from OpenAI
        response_mock = MagicMock()
        response_mock.status = 200
        response_mock.json.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'transaction_id': '123',
                        'supplier_name': 'Acme',
                        'category_classification': 'Computers',
                        'classification_code': '43211500',
                        'level': 'L3',
                        'reason': 'Best match based on context',
                        'confidence_score': 8
                    })
                }
            }]
        }
        
        session.post.return_value.__aenter__.return_value = response_mock
        
        # Execute
        result = await select_best_match(
            session,
            '123',
            'Computer purchase',
            'Acme',
            'Dell desktop computer',
            1000.0,
            l3_matches,
            l4_matches,
            'IT department',
            semaphore,
            rate_limiter
        )
        
        # Verify
        assert result is not None
        assert result['Category_ID'] == '43211500'
        assert result['Category_Description'] == 'Computers'
        assert result['level'] == 'L3'
        assert result['Confidence_Score'] == 8
        
        # Verify request
        session.post.assert_called_once()
        rate_limiter.acquire.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_get_level1_categories(self):
        """Test getting level 1 categories."""
        # Setup
        session = MagicMock()
        semaphore = MagicMock()
        rate_limiter = MagicMock()
        
        # Mock response from OpenAI
        response_mock = MagicMock()
        response_mock.status = 200
        response_mock.json.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'category_numbers': [43, 44, 45]
                    })
                }
            }]
        }
        
        session.post.return_value.__aenter__.return_value = response_mock
        
        # Mock database function to return sectors
        with patch('src.services.database.fetch_relevant_unspsc_sectors') as mock_fetch:
            mock_fetch.return_value = [
                {'unspsc_sector': 43, 'unspsc_sector_description': 'IT Equipment'},
                {'unspsc_sector': 44, 'unspsc_sector_description': 'Office Equipment'},
                {'unspsc_sector': 45, 'unspsc_sector_description': 'Printing Equipment'}
            ]
            
            # Execute
            result = await get_level1_categories(
                session,
                'Test Supplier',
                'Test Description',
                'Cleansed Description',
                'IT Industry',
                1000.0,
                1,  # industry_filter
                semaphore,
                rate_limiter
            )
        
        # Verify
        assert result == [43, 44, 45]
        
        # Verify request
        session.post.assert_called_once()
        rate_limiter.acquire.assert_called_once()