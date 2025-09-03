import pandas as pd
from utils.contract_utils import custom_monthly_contract_sort_key

def check_positions(df, date, units):
    """
    Generate and save a pivot table by 'unit' and 'derivative_type'
    for either 'delta' or 'total_active_lots'.

    Parameters:
        df (pd.DataFrame): Position data.
        date (str): Date string for CSV filename.
        units (str): 'delta' or 'total_active_lots'.

    Output:
        Saves a CSV file with the pivot.
    """
    check_pos_df = df.copy()
    check_pos_df = check_pos_df[~check_pos_df['portfolio'].str.contains('MILL OPTNS', na=False) &
                                ~check_pos_df['books'].isin(['ADMIN', 'POOL', 'NON OIL'])]
    check_pos_df.to_csv('check_pos' + str(date) + '.csv', index=False)
    if units not in ['delta', 'total_active_lots']:
        raise ValueError("Parameter 'units' must be either 'delta' or 'total_active_lots'.")

    value_col = units

    pivot_df = pd.pivot_table(check_pos_df, index=['unit', 'derivative_type'], columns='underlying_bbg_ticker',
                              values=value_col, aggfunc='sum', fill_value=0).reset_index()

    # Sort contract columns by your custom sort key
    index_cols = ['unit', 'derivative_type']
    contract_cols = [col for col in pivot_df.columns if col not in index_cols]
    sorted_contracts = sorted(contract_cols, key=custom_monthly_contract_sort_key)
    pivot_df = pivot_df[index_cols + sorted_contracts]

    csv_filename = f"pivot_check_{value_col}_{date}.csv"
    pivot_df.to_csv(csv_filename, index=False)
    print(f"Pivot saved to {csv_filename}")
