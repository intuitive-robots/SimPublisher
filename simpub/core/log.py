import logging
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)


class CustomFormatter(logging.Formatter):
    """Custom log formatter that adds colors"""
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.YELLOW + FORMAT + Fore.RESET,
        logging.INFO: Fore.BLUE + FORMAT + Fore.RESET,
        logging.WARNING: Fore.RED + FORMAT + Fore.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger():
    """Create and return a custom logger"""
    logger = logging.getLogger()
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


logger = get_logger()
