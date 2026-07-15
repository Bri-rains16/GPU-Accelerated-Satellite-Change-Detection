import logging
import sys


def setup_logger(name="HPC_Logger", log_level=logging.INFO):
    """
    Sets up a centralized logger that outputs to both console and a log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if logger already exists
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)
    return logger
