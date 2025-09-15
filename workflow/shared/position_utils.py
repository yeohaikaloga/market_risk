"""
POSITION PROCESSING UTILITIES
=============================

Specialist functions for calculating position aggregates, basis adjustments, and exposure tagging.

These are pure, reusable functions — they know nothing about workflows, loaders, or databases.
They operate ONLY on the data given to them.

Used by: data_preparation_workflow.py
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any, Union


def calculate_aggregates(group: pd.DataFrame) -> pd.Series:
    net_fixed_phys = group[group['position_type'] == 'FIXED PHYS']['quantity'].sum()
    net_diff_phys = group[group['position_type'] == 'DIFF PHYS']['quantity'].sum()
    derivs = group[group['position_type'] == 'DERIVS']['quantity'].sum()

    return pd.Series({
        'net_fixed_phys': float(net_fixed_phys),
        'net_diff_phys': float(net_diff_phys),
        'derivs': float(derivs)
    })


def calculate_basis_adj_and_basis_pos(row: pd.Series) -> Tuple[float, float]:
    """
    Calculate basis adjustment and basis position from an aggregated row.

    Business Rule (Cotton):
    - basis_adj = net_diff_phys
    - basis_pos = net_fixed_phys + net_diff_phys

    Used in data preparation to tag and quantify "BASIS (NET PHYS)" exposure.

    Args:
        row (pd.Series): A row containing 'net_fixed_phys' and 'net_diff_phys' (from calculate_aggregates).

    Returns:
        Tuple[float, float]: (basis_adj, basis_pos)
    """
    basis_adj = float(row.get('net_diff_phys', 0.0))
    basis_pos = float(row.get('net_fixed_phys', 0.0)) + basis_adj

    return basis_adj, basis_pos


def tag_exposure_type(row: pd.Series) -> str:
    """
    Assign exposure type based on position type.

    Business Rules:
    - 'FIXED PHYS', 'FUTURES', 'OPTIONS' → 'OUTRIGHT'
    - 'DIFF PHYS' → 'BASIS'
    - Others → 'UNKNOWN'

    Can be applied with: df['exposure'] = df.apply(tag_exposure_type, axis=1)

    Args:
        row (pd.Series): A row with 'position_type' column.

    Returns:
        str: Exposure type ('OUTRIGHT', 'BASIS', or 'UNKNOWN')
    """
    position_type = row.get('position_type', '')

    if position_type in ['FIXED PHYS', 'FUTURES']:
        return 'OUTRIGHT'
    elif 'CALLS' in position_type or 'PUTS' in position_type:
        return 'OUTRIGHT'
    elif position_type == 'DIFF PHYS':
        return 'BASIS'
    else:
        return 'UNKNOWN'


def calculate_net_position(row: pd.Series) -> float:
    """
    Calculate net position (delta equivalent) for a single position row.

    For physicals: quantity = delta
    For derivatives: total_active_lots * settle_delta_1 * lots_to_MT_conversion

    Assumes row has:
    - 'position_type'
    - 'quantity' (for physicals)
    - 'total_active_lots', 'settle_delta_1', 'lots_to_MT_conversion' (for derivatives)

    Args:
        row (pd.Series): Single position row.

    Returns:
        float: Net position in MT (or equivalent base unit)
    """
    if row['position_type'] == 'DERIVS':
        return float(
            row.get('total_active_lots', 0.0) *
            row.get('settle_delta_1', 0.0) *
            row.get('lots_to_MT_conversion', 1.0)
        )
    else:
        return float(row.get('quantity', 0.0))


def validate_position_row(row: pd.Series, required_columns: list) -> bool:
    """
    Validate that a position row has all required columns and no nulls in critical fields.

    Args:
        row (pd.Series): Position row to validate
        required_columns (list): List of column names that must be present and non-null

    Returns:
        bool: True if valid, False otherwise
    """
    for col in required_columns:
        if col not in row or pd.isna(row[col]):
            return False
    return True
