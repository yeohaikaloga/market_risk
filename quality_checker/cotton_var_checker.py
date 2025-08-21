import pandas as pd
from utils.contract_utils import custom_monthly_contract_sort_key

def check_derivatives_positions(df):
    check_deriv_pos_df = df.copy()
    check_deriv_pos_df = check_deriv_pos_df[
        (~check_deriv_pos_df['portfolio'].str.contains('MILL OPTNS'))  # filter out MILL OPTNS from UNIT_SD
        & (~check_deriv_pos_df['books'].isin(
            ['ADMIN', 'POOL', 'NON OIL']))]  # filter out ADMIN, POOL, NON OIL from BOOKS
    grouped_df = (check_deriv_pos_df.groupby(['region', 'derivative_type']).sum(numeric_only=True).
                  reset_index().sort_values(by=['region', 'derivative_type']))
    print(grouped_df)

    pivot_df = pd.pivot_table(check_deriv_pos_df,
                              index=['unit', 'derivative_type'],
                              columns=['underlying_bbg_ticker'],
                              values='total_active_lots',
                              aggfunc='sum',
                              fill_value=0).reset_index()  # fill missing combinations with 0
    index_cols = ['unit', 'derivative_type']
    cols_to_sort = [col for col in pivot_df.columns if col not in index_cols]
    sorted_cols = sorted(cols_to_sort, key=custom_monthly_contract_sort_key)
    sorted_columns = index_cols + sorted_cols
    pivot_df = pivot_df[sorted_columns]
    print(pivot_df)
    pivot_df.to_csv('pivot_check.csv')
