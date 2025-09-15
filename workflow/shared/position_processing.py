import pandas as pd
from typing import Tuple


def calculate_phys_derivs_aggs(group: pd.DataFrame) -> pd.Series:
    """
    Calculate aggregated position metrics for a group (e.g., by region, contract).

    Assumes columns: 'position_type', 'quantity'
    """
    net_fixed_phys = group[group['position_type'] == 'FIXED PHYS']['quantity'].sum()
    net_diff_phys = group[group['position_type'] == 'DIFF PHYS']['quantity'].sum()
    derivs = group[group['position_type'] == 'DERIVS']['quantity'].sum()
    return pd.Series({
        'net_fixed_phys': net_fixed_phys,
        'net_diff_phys': net_diff_phys,
        'derivs': derivs
    })


def calculate_basis_adj_and_basis_pos(row: pd.Series) -> Tuple[float, float]:
    """
    Calculate basis adjustment and basis position from aggregated row.

    Business rule: basis_pos = net_fixed_phys + net_diff_phys
                   basis_adj = net_diff_phys
    """
    basis_adj = row.get('net_diff_phys', 0.0)
    basis_pos = row.get('net_fixed_phys', 0.0) + basis_adj
    return basis_adj, basis_pos
