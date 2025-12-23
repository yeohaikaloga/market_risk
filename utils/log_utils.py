import logging
import sys
import os
import io


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


def setup_logging(app_name="default_app_name", log_filename="run_summary.txt"):
    """
    Sets up a logger that captures everything in memory for a final export
    while still printing to the console.
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)

    # 1. Clear existing handlers if re-running in a notebook/session
    if logger.hasHandlers():
        logger.handlers.clear()

    # 2. Create a String Buffer to hold logs in memory
    log_buffer = io.StringIO()

    # 3. Define Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. Console Handler (so you see it live)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 5. Buffer Handler (for the final export)
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setFormatter(formatter)
    logger.addHandler(buffer_handler)

    def export_logs():
        """Writes the captured memory buffer to a physical file."""
        try:
            with open(log_filename, "w") as f:
                f.write(log_buffer.getvalue())
            print(f"\n[Success] Logs exported to {os.path.abspath(log_filename)}")
        except Exception as e:
            print(f"Failed to export logs: {e}")
        finally:
            log_buffer.close()

    return logger, export_logs

# def save_to_pickle(data, filename: str):
#     """Saves data to a pickle file."""
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)
#
# def load_from_pickle(filename: str):
#     """Loads data from a pickle file."""
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)
#     return data