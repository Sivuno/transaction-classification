import time
import asyncio
import logging
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

class GlobalProgress:
    """
    Tracks global progress across multiple processing tasks.
    Provides a progress bar and statistics.
    """
    def __init__(self, total_transactions):
        """
        Initialize progress tracker.
        
        Args:
            total_transactions (int): Total number of transactions to process
        """
        logger.debug(f"Initializing GlobalProgress with {total_transactions} total transactions.")
        self.total_transactions = total_transactions
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()
        self.progress_bar = None

    def initialize_progress_bar(self):
        """Initialize the tqdm progress bar."""
        logger.debug("Initializing global progress bar.")
        self.progress_bar = tqdm(
            total=self.total_transactions,
            desc="Processing Transactions",
            unit="tx",
            ncols=100,
            position=0,
            leave=True
        )

    async def increment_processed(self):
        """Increment the count of successfully processed transactions."""
        logger.debug("Incrementing processed count.")
        async with self._lock:
            self.processed_count += 1
            if self.progress_bar:
                self.progress_bar.update(1)
                if self.processed_count % 100 == 0:
                    self._update_progress_stats()

    async def increment_failed(self):
        """Increment the count of failed transactions."""
        logger.debug("Incrementing failed count.")
        async with self._lock:
            self.failed_count += 1
            self._update_progress_stats()

    def _update_progress_stats(self):
        """Update progress bar statistics."""
        if not self.progress_bar:
            return
            
        elapsed = time.time() - self.start_time
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        remaining = (self.total_transactions - self.processed_count) / rate if rate > 0 else 0

        self.progress_bar.set_postfix({
            'Failed': self.failed_count,
            'Rate': f'{rate:.1f} tx/s',
            'ETA': f'{remaining/3600:.1f}h'
        }, refresh=True)

    def close(self):
        """Close the progress bar."""
        logger.debug("Closing global progress bar.")
        if self.progress_bar:
            self.progress_bar.close()
            
    def get_summary(self):
        """
        Generate a summary of the processing results.
        
        Returns:
            dict: Summary statistics
        """
        elapsed = time.time() - self.start_time
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        
        return {
            "total_transactions": self.total_transactions,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "elapsed_time_seconds": elapsed,
            "elapsed_time_hours": elapsed / 3600,
            "processing_rate": rate
        }