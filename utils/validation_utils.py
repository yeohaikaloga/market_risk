import pandas as pd
from typing import Literal


def check_cartesian_product(left_df, right_df, join_keys: list,
                            join_type: Literal['left', 'right', 'inner', 'outer', 'cross'] = 'left') \
        -> tuple[bool, pd.DataFrame]:

    """
    Check whether a merge between left_df and right_df on join_keys would result
    in a Cartesian product or row inflation.

    Args:
        left_df (pd.DataFrame): The left DataFrame in the merge.
        right_df (pd.DataFrame): The right DataFrame in the merge.
        join_keys (list): List of column names to merge on.
        join_type (str): Type of join to simulate ('left', 'inner', etc.)

    Returns:
        bool: True if a Cartesian product is detected, False otherwise.
        pd.DataFrame: Duplicated rows if any.
    """
    pre_merge_rows = len(left_df)
    merged_df = left_df.merge(right_df, on=join_keys, how=join_type)
    post_merge_rows = len(merged_df)

    if post_merge_rows > pre_merge_rows:
        # Show duplicates that contributed to row explosion
        duplicates = right_df[right_df.duplicated(subset=join_keys, keep=False)]
        return True, duplicates
    else:
        return False, pd.DataFrame()

