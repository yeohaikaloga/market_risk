import logging
import sys
import os
import io
import colorlog

_APP_NAME_STORE = None

def setup_logging(app_name="default_app_name", log_filename="run_summary.txt"):
    """
    Called ONCE in main.py to set the 'Identity' of the app
    and configure where logs go.
    """
    global _APP_NAME_STORE
    _APP_NAME_STORE = app_name

    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)

    # 1. Clear existing handlers if re-running in a notebook/session
    if logger.hasHandlers():
        logger.handlers.clear()

    # 2. Create a String Buffer to hold logs in memory
    log_buffer = io.StringIO()

    # 3. Define Formatter
    standard_formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. Console Handler (so you see it live)
    console_handler = logging.StreamHandler()
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        style='%'
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # 5. Buffer Handler (No colors here, so the exported file is readable)
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setFormatter(standard_formatter)
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


def get_logger(name: str, level=logging.INFO):
    """
    Used in every other file.
    It checks what 'app_name' was set in setup_logging.
    """
    # If setup_logging hasn't been called yet, it defaults to the module name.
    # If it HAS, it returns "YourAppName.module_name"
    prefix = f"{_APP_NAME_STORE}." if _APP_NAME_STORE else ""
    return logging.getLogger(f"{prefix}{name}")


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