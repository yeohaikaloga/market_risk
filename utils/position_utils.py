import pandas as pd
import numpy as np


def calculate_aggregates(x):
    net_fixed_physicals = x[x['typ'].isin(['FIXED PURCHASE', 'FIXED SALES'])]['quantity'].sum()
    net_diff_physicals = x[x['typ'].isin(['DIFF PURCHASE', 'DIFF SALES'])]['quantity'].sum()
    net_physicals = x[x['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'DIFF PURCHASE', 'DIFF SALES'])]['quantity'].sum()
    derivs = x[x['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])]['quantity'].sum()

    return pd.Series({'net_fixed_phys': net_fixed_physicals, 'net_diff_phys': net_diff_physicals,
                      'net_phys': net_physicals, 'derivs': derivs})


def calculate_basis_adj_and_basis_pos(row):
    sign_fixed = np.sign(row['net_fixed_phys'])
    sign_deriv = np.sign(row['derivs'])
    abs_fixed = abs(row['net_fixed_phys'])
    abs_deriv = abs(row['derivs'])

    if sign_fixed == sign_deriv:
        return 0, row['net_diff_phys']
    else:
        min_abs = min(abs_fixed, abs_deriv)
        return min_abs * sign_fixed, min_abs * sign_fixed + row['net_diff_phys']
