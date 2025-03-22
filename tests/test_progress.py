"""
Tests for progress tracking functionality.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.utils.progress import GlobalProgress

class TestGlobalProgress:
    """Tests for the GlobalProgress class."""
    
    def test_init(self):
        """Test initialization."""
        progress = GlobalProgress(100)
        
        assert progress.total_count == 100
        assert progress.processed_count == 0
        assert progress.failed_count == 0
        assert progress.start_time is not None
    
    def test_initialize_progress_bar(self):
        """Test initializing progress bar."""
        with patch('src.utils.progress.tqdm') as mock_tqdm:
            # Setup mock
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar
            
            # Execute
            progress = GlobalProgress(100)
            progress.initialize_progress_bar()
            
            # Verify
            assert progress.progress_bar == mock_bar
            mock_tqdm.assert_called_once()
            assert mock_tqdm.call_args[1]['total'] == 100
    
    @pytest.mark.asyncio
    async def test_increment_processed(self):
        """Test incrementing processed count."""
        # Setup
        progress = GlobalProgress(100)
        progress.progress_bar = MagicMock()
        
        # Execute
        await progress.increment_processed()
        
        # Verify
        assert progress.processed_count == 1
        progress.progress_bar.update.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_increment_failed(self):
        """Test incrementing failed count."""
        # Setup
        progress = GlobalProgress(100)
        progress.progress_bar = MagicMock()
        
        # Execute
        await progress.increment_failed()
        
        # Verify
        assert progress.failed_count == 1
        progress.progress_bar.update.assert_called_once_with(1)
    
    def test_close(self):
        """Test closing progress bar."""
        # Setup
        progress = GlobalProgress(100)
        progress.progress_bar = MagicMock()
        
        # Execute
        progress.close()
        
        # Verify
        progress.progress_bar.close.assert_called_once()
    
    def test_get_summary(self):
        """Test getting progress summary."""
        # Setup with known values
        with patch('time.time', return_value=3600):  # 1 hour later
            with patch('src.utils.progress.time.time', return_value=0):  # Start time
                progress = GlobalProgress(100)
                progress.processed_count = 60
                progress.failed_count = 10
                
                # Override start time for consistent testing
                progress.start_time = 0
                
                # Execute
                summary = progress.get_summary()
                
        # Verify
        assert summary['processed_count'] == 60
        assert summary['failed_count'] == 10
        assert summary['elapsed_time_seconds'] == 3600
        assert summary['elapsed_time_hours'] == 1.0
        assert summary['processing_rate'] == 60 / 3600  # 60 transactions in 1 hour