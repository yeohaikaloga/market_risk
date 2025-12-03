import logging
import sys
import pickle


def get_logger(name: str, level=logging.INFO):
    """Standard logger setup."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def save_to_pickle(data, filename: str):
    """Saves data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(filename: str):
    """Loads data from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data