import os
import pandas as pd
import pickle
from typing import Any


def get_output_directory(cob_date: str) -> str:
    """
    Constructs the path for the output directory based on the COB date.
    """
    # Format the directory name using the date, e.g., 'var_reports_2024-01-30'
    dir_name = f'var_reports_{cob_date}'
    return dir_name


def create_output_directory(cob_date: str):
    """
    Creates the required output directory if it doesn't already exist.
    """
    output_dir = get_output_directory(cob_date)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"INFO: Created output directory: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"ERROR: Could not create directory {output_dir}: {e}")
        # Re-raise the exception to halt the workflow if essential step fails
        raise


def get_full_path(cob_date: str, filename: str) -> str:
    """
    Returns the full file path inside the COB date-specific output directory.
    """
    output_dir = get_output_directory(cob_date)
    return os.path.join(output_dir, filename)


def save_to_pickle_in_dir(data: Any, cob_date: str, filename: str):
    """
    Saves data to a pickle file inside the COB date-specific output directory.
    """
    full_path = get_full_path(cob_date, filename)
    try:
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"INFO: Data saved to pickle: {full_path}")
    except Exception as e:
        print(f"ERROR: Failed to save pickle file {full_path}: {e}")


def load_from_pickle_in_dir(cob_date: str, filename: str) -> Any:
    """
    Loads data from a pickle file inside the COB date-specific output directory.
    """
    full_path = get_full_path(cob_date, filename)
    try:
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
        print(f"INFO: Data loaded from pickle: {full_path}")
        return data
    except Exception as e:
        print(f"ERROR: Failed to load pickle file {full_path}: {e}")
        raise


def save_to_feather_in_dir(data: pd.DataFrame, cob_date: str, filename: str):
    """
    Saves a Pandas DataFrame to a Feather file inside the COB date-specific
    output directory.

    NOTE: Feather format works best with Pandas DataFrames.
    """
    # Assuming get_full_path constructs the full path based on cob_date and filename
    # Example: full_path = os.path.join(f'var_reports_{cob_date}', filename)
    full_path = get_full_path(cob_date, filename)

    # Feather files typically use the .feather or .arrow extension.
    if not filename.lower().endswith(('.feather', '.arrow')):
        print("WARNING: Feather filename does not end with .feather or .arrow. Appending .feather.")
        full_path += ".feather"

    try:
        # Check if the data is a DataFrame, as Feather is optimized for it
        if not isinstance(data, pd.DataFrame):
            print("WARNING: Data being saved to Feather is not a Pandas DataFrame. Conversion attempts may fail.")

        data.to_feather(full_path)
        print(f"INFO: Data saved to feather: {full_path}")
    except Exception as e:
        print(f"ERROR: Failed to save feather file {full_path}: {e}")
        raise


def load_from_feather_in_dir(cob_date: str, filename: str) -> pd.DataFrame:
    """
    Loads data (as a Pandas DataFrame) from a Feather file inside the
    COB date-specific output directory.
    """
    full_path = get_full_path(cob_date, filename)

    # Ensure file path has correct extension if convention is followed
    if not filename.lower().endswith(('.feather', '.arrow')):
        full_path += ".feather"

    try:
        data = pd.read_feather(full_path)
        print(f"INFO: Data loaded from feather: {full_path}")
        return data
    except Exception as e:
        print(f"ERROR: Failed to load feather file {full_path}: {e}")
        raise
