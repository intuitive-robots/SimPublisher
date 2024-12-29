import logging
from colorama import init, Fore

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
