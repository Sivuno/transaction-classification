import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Asynchronous rate limiter for API calls.
    Helps avoid rate limiting errors by spacing out requests.
    """
    def __init__(self, calls_per_minute):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_minute (int): Maximum number of calls allowed per minute
        """
        logger.debug(f"Initializing RateLimiter with {calls_per_minute} calls/minute")
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire a token to make an API call.
        This method will block until it's safe to make a call without exceeding the rate limit.
        """
        logger.debug("Acquiring rate limiter token...")
        async with self.lock:
            current_time = time.time()
            # Remove timestamps older than 1 minute
            self.call_times = [t for t in self.call_times if current_time - t < 60]

            # If we've reached the limit, wait until we're allowed to make another call
            if len(self.call_times) >= self.calls_per_minute:
                wait_time = 60 - (current_time - self.call_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached. Waiting {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                self.call_times.pop(0)

            # Record this call
            self.call_times.append(time.time())
        logger.debug("Rate limiter token acquired.")