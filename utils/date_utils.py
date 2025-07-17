import pandas as pd
from typing import List


def no_of_days_list(cob_date: str, no_of_days: int) -> List[pd.Timestamp]:
    """
        Generate a list of business days before and including the COB date.

        Parameters:
            cob_date (pd.Timestamp): The current date (e.g. close-of-business date).
            no_of_days (int): Number of business days to go back (including cob_date).

        Returns:
            List[pd.Timestamp]: List of business day timestamps in descending order.
    """
    dt_cob_date = pd.to_datetime(cob_date)
    return sorted([dt_cob_date - pd.tseries.offsets.BDay(i) for i in range(no_of_days)])
