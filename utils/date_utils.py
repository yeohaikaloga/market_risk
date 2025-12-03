import pandas as pd
from typing import List, Union
from sqlalchemy import text
from db.db_connection import get_engine


def get_prev_biz_days_list(date: str, no_of_days: int) -> List[pd.Timestamp]:
    """
        Generate a list of business days before and including the COB date.

        Parameters:
            date (pd.Timestamp): date (typically close-of-business date).
            no_of_days (int): Number of business days to go back (including cob_date).

        Returns:
            List[pd.Timestamp]: List of business day timestamps in descending order.
    """
    dt_date = pd.to_datetime(date)
    return sorted([dt_date - pd.tseries.offsets.BDay(i) for i in range(no_of_days)])


def get_biz_days_between_list(start_date: str, end_date: str) -> List[pd.Timestamp]:
    """
    Generate a list of business days between start_date and end_date inclusive.

    Parameters:
        start_date (str): Start date (e.g. '2025-07-01')
        end_date (str): End date (e.g. '2025-07-10')

    Returns:
        List[pd.Timestamp]: List of business day timestamps in ascending order.
    """
    dt_start_date = pd.to_datetime(start_date)
    dt_end_date = pd.to_datetime(end_date)

    # Generate business days between start and end inclusive
    biz_days = pd.bdate_range(start=dt_start_date, end=dt_end_date)

    return list(biz_days)


def get_prev_weekdays_list(end_date: Union[str, pd.Timestamp], no_of_days: int) -> List[pd.Timestamp]:
    """
    Generate a list of the most recent weekdays (Mon–Fri) ending at `end_date`.

    Parameters:
        end_date (str or pd.Timestamp): End date (inclusive)
        no_of_days (int): Number of weekdays to include (Mon–Fri only)

    Returns:
        List[pd.Timestamp]: List of weekday timestamps in ascending order
    """
    end_date = pd.to_datetime(end_date)

    # Initialize a list to collect weekdays
    weekdays = []
    current_date = end_date

    while len(weekdays) < no_of_days:
        if current_date.weekday() < 5:  # 0=Mon, 4=Fri
            weekdays.append(current_date)
        current_date -= pd.Timedelta(days=1)

    return sorted(weekdays)  # Ascending order


def get_weekdays_between_list(start_date: Union[str, pd.Timestamp],
                              end_date: Union[str, pd.Timestamp]) -> List[pd.Timestamp]:
    """
    Generate a list of weekdays (Mon–Fri) between start_date and end_date inclusive.

    Parameters:
        start_date (str or pd.Timestamp): Start date (inclusive)
        end_date (str or pd.Timestamp): End date (inclusive)

    Returns:
        List[pd.Timestamp]: List of weekday timestamps in ascending order
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_dates = pd.date_range(start, end, freq='D')
    weekdays = all_dates[all_dates.weekday < 5]  # 0–4 = Mon–Fri
    return list(weekdays)

def load_opera_market_calendar(instrument: Union[str, list]) -> pd.DataFrame:
    """
    Load the Opera market calendar for one or more instruments.
    Returns a DataFrame indexed by value_date with a single column 'is_holiday'.

    Parameters:
        instrument (str or list): Product code(s).

    Returns:
        pd.DataFrame:
            index: value_date (datetime)
            column: is_holiday
    """

    # Convert instrument into SQL filter
    if isinstance(instrument, str):
        instr_filter = f"'{instrument.strip()}'"
    elif isinstance(instrument, list) and instrument:
        instr_filter = ", ".join(f"'{i}'" for i in instrument)
    else:
        raise ValueError("Instrument must be a non-empty string or list of strings.")

    query = f"""
        SELECT product_code, value_date, is_holiday
        FROM staging.opera_holiday_calendar
        WHERE product_code IN ({instr_filter})
    """

    print(query)

    uat_engine = get_engine("uat")
    with uat_engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    # Convert value_date to datetime
    df["value_date"] = pd.to_datetime(df["value_date"])

    # --- VALIDATION: ensure only one product_code present ---
    unique_products = df["product_code"].unique()

    if len(unique_products) == 0:
        raise ValueError(f"No holiday calendar data found for given instrument(s): {unique_products}.")

    if len(unique_products) > 1:
        raise ValueError(
            f"Holiday calendar contains multiple product codes: {unique_products}. "
            f"Expected exactly 1. Please load one product at a time."
        )

    # Drop product_code now that we know it is unique
    df = df.drop(columns=["product_code"])

    # Set value_date as index
    df = df.set_index("value_date").sort_index()

    return df