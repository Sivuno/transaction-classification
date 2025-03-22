"""
Tests for rate limiter functionality.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from src.utils.rate_limiter import RateLimiter

class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_init(self):
        """Test initialization."""
        limiter = RateLimiter(calls_per_minute=60)
        assert limiter.calls_per_minute == 60
        assert limiter.call_times == []
        assert isinstance(limiter.lock, asyncio.Lock)
    
    @pytest.mark.asyncio
    async def test_acquire_no_waiting(self):
        """Test acquiring a token when not at the limit."""
        # Set a fixed time for testing
        with patch('time.time', return_value=1000.0):
            limiter = RateLimiter(calls_per_minute=60)
            
            # Acquire a token
            await limiter.acquire()
            
            # Verify a timestamp was added
            assert len(limiter.call_times) == 1
            assert limiter.call_times[0] == 1000.0  
    
    @pytest.mark.asyncio
    async def test_acquire_with_cleanup(self):
        """Test that old timestamps are cleaned up."""
        limiter = RateLimiter(calls_per_minute=60)
        
        # Add some old timestamps (more than 60 seconds old)
        # If current time is 1000, then times before 940 should be cleaned up
        with patch('time.time', return_value=1000.0):
            limiter.call_times = [930.0, 935.0, 938.0]  # All older than 60 seconds
            
            # Acquire should clean up old timestamps and add new one
            await limiter.acquire()
            
            # The implementation actually adds the new timestamp after cleanup
            # So we expect only the new timestamp
            assert len(limiter.call_times) == 1
            assert limiter.call_times[0] == 1000.0
    
    @pytest.mark.asyncio
    async def test_acquire_with_waiting(self):
        """Test waiting when rate limit is reached."""
        limiter = RateLimiter(calls_per_minute=2)  # Only 2 calls per minute
        
        # Mock sleep to avoid actual waiting
        mock_sleep = AsyncMock()
        
        # Set up timestamps to simulate being at the limit
        with patch('time.time', return_value=1000.0), \
             patch('asyncio.sleep', mock_sleep):
            
            # Add existing timestamps to simulate being at the limit
            limiter.call_times = [950.0, 980.0]  # 2 calls in the last minute
            
            # Acquire should wait
            await limiter.acquire()
            
            # Verify sleep was called with appropriate wait time
            # Should wait until 950.0 + 60 = 1010.0, so 10 seconds
            mock_sleep.assert_called_once_with(10.0)
            
            # Oldest timestamp should be removed and a new one added
            assert len(limiter.call_times) == 2
            assert limiter.call_times[0] == 980.0
            assert limiter.call_times[1] == 1000.0
    
    @pytest.mark.asyncio
    async def test_multiple_acquires(self):
        """Test acquiring multiple tokens in sequence."""
        limiter = RateLimiter(calls_per_minute=60)
        
        # Use a fixed timestamp for each call to avoid side_effect exhaustion
        with patch('time.time') as mock_time:
            # Set return values for each call
            mock_time.side_effect = [1000.0, 1000.0, 1001.0, 1001.0, 1002.0, 1002.0]
            
            # Acquire 3 tokens in sequence
            await limiter.acquire()  # 2 calls to time.time() per acquire
            await limiter.acquire()
            await limiter.acquire()
            
            # Should have 3 timestamps
            assert len(limiter.call_times) == 3
            assert 1000.0 in limiter.call_times
            assert 1001.0 in limiter.call_times
            assert 1002.0 in limiter.call_times