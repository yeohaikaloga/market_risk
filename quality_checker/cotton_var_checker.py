import pandas as pd
from utils.contract_utils import custom_monthly_contract_sort_key


def check_pivot_positions(df, date, units):
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


def validate_combined_position_df(combined_pos_df: pd.DataFrame):
    """
    Validates the combined cotton position DataFrame for common issues like missing mappings,
    overwritten values, and inconsistent fields across physical and derivatives positions.
    Prints warnings and summaries.

    Args:
        combined_pos_df (pd.DataFrame): The merged physical + derivatives position table.
    """

    print("\n===== VALIDATION: combined_pos_df =====")

    # 1. Check missing critical fields
    critical_fields = ['region', 'instrument_name', 'bbg_ticker', 'generic_curve', 'delta', 'to_USD_conversion']
    for col in critical_fields:
        missing_count = combined_pos_df[col].isna().sum()
        if missing_count > 0:
            print(f"[WARNING] Missing values in '{col}': {missing_count} rows")

    # 2. Check optional fields that are often empty but useful
    optional_fields = ['underlying_bbg_ticker', 'trader_id', 'counterparty_id', 'books']
    for col in optional_fields:
        if col in combined_pos_df.columns:
            na_count = combined_pos_df[col].isna().sum()
            total = len(combined_pos_df)
            print(f"[NOTE] Column '{col}' has {na_count}/{total} NaN values")

    # 3. Check if 'books' column has unexpected values
    if 'books' in combined_pos_df.columns:
        books_unique = combined_pos_df['books'].dropna().unique()
        expected_books = {'PHYSICALS', 'DERIVATIVES', 'HEDGE', 'SPR NEW NZ', 'PRICE', 'SPR NEW HK', 'EQUITIES',
                          'LIBERTY', 'SPR NEW KN', 'SPR NEW ZH'}
        unexpected = set(books_unique) - expected_books
        if unexpected:
            print(f"[WARNING] Unexpected 'books' values found: {unexpected}")

    # 4. Check if all positions have valid delta
    if (combined_pos_df['delta'] == 0).all():
        print("[WARNING] All delta values are zero — check if delta was calculated correctly.")
    elif (combined_pos_df['delta'].isna()).any():
        print("[WARNING] Some delta values are missing — likely due to conversion mapping failure.")

    # 5. Check region mapping integrity
    missing_regions = combined_pos_df[combined_pos_df['region'].isna()]
    if not missing_regions.empty:
        print(f"[WARNING] {len(missing_regions)} rows missing region — possible mapping issue.")
        print(missing_regions[['unit', 'instrument_name']].drop_duplicates())

    # 6. Print summary stats
    print("\n===== ✅ Summary =====")
    print("Non-zero position rows:", len(combined_pos_df[combined_pos_df['delta'] != 0]))
    print("Unique instruments:", combined_pos_df['instrument_name'].nunique())
    print("Unique regions:", combined_pos_df['region'].nunique())

    print("===== END VALIDATION =====\n")
