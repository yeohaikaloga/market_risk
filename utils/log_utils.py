import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger name (usually __name__ from the calling module)
        level: Logging level (default: INFO)

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid adding multiple handlers if called multiple times
        logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # StreamHandler to print to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger