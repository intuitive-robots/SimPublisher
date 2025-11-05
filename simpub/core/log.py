import logging
import time
from functools import wraps
from typing import Any, Callable

from colorama import Fore, init

init(autoreset=True)

# Define a new log level
REMOTE_LOG_LEVEL_NUM = 25
logging.addLevelName(REMOTE_LOG_LEVEL_NUM, "REMOTELOG")


class CustomLogger(logging.Logger):
    def remote_log(self, message, *args, **kws):
        if self.isEnabledFor(REMOTE_LOG_LEVEL_NUM):
            self._log(REMOTE_LOG_LEVEL_NUM, message, args, **kws)


class CustomFormatter(logging.Formatter):
    """Custom log formatter that adds colors"""

    FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.YELLOW + FORMAT + Fore.RESET,
        logging.INFO: Fore.BLUE + FORMAT + Fore.RESET,
        logging.WARNING: Fore.RED + FORMAT + Fore.RESET,
        logging.ERROR: Fore.MAGENTA + FORMAT + Fore.RESET,
        logging.CRITICAL: Fore.CYAN + FORMAT + Fore.RESET,
        # Add custom level format
        REMOTE_LOG_LEVEL_NUM: Fore.GREEN + FORMAT + Fore.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger():
    """Create and return a custom logger"""
    # logger = logging.getLogger("SimPublisher")
    logger = CustomLogger("SimPublisher")
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create custom formatter and add to handler
    formatter = CustomFormatter()
    ch.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(ch)

    return logger


# Get the logger
logger = get_logger()


def func_timing(func: Callable) -> Callable:
    """Simple timing decorator that just logs total execution time"""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            total_time = time.time() - start_time
            logger.info(f"{func.__name__} took {total_time*1000:.2f}ms")
            return result
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {total_time*1000:.2f}ms: {e}"
            )
            raise

    return wrapper
