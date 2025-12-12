import pandas as pd
from db.db_connection import get_engine
from position_loader.position_loader import PositionLoader
from workflow.shared.positions.cotton_positions import generate_cotton_combined_position
from workflow.shared.positions.rubber_positions import generate_rubber_combined_position
from workflow.shared.positions.rms_positions import generate_rms_combined_position
from workflow.shared.positions.biocane_positions import generate_biocane_combined_position
from workflow.shared.positions.wood_positions import generate_wood_combined_position
from utils.log_utils import get_logger


def build_combined_position(cob_date, product, instrument_dict, prices_df, fx_spot_df):
    """
    Build combined position DataFrame for one or multiple product groups.

    Parameters
    ----------
    cob_date : str
    product : str
        'cotton', 'rubber', 'rms', 'biocane', 'wood' or 'all'
    instrument_dict : dict
    prices_df : pd.DataFrame
    fx_spot_df : pd.DataFrame


    Returns
    -------
    pd.DataFrame
        Combined position dataframe.
    """

    if product == 'cotton':
        combined_pos_df = generate_cotton_combined_position(cob_date, instrument_dict, prices_df, fx_spot_df)
    elif product == 'rubber':
        combined_pos_df = generate_rubber_combined_position(cob_date, instrument_dict, prices_df, fx_spot_df)
    elif product == 'rms':
        combined_pos_df = generate_rms_combined_position(cob_date, instrument_dict, prices_df, fx_spot_df)
    elif product == 'wood':
        combined_pos_df = generate_wood_combined_position(cob_date, instrument_dict, prices_df, fx_spot_df)
    elif product == 'biocane':
        combined_pos_df = generate_biocane_combined_position(cob_date, instrument_dict, prices_df, fx_spot_df)
    else:
        raise NotImplementedError(f"Product '{product}' not yet supported in position workflow.")
    return combined_pos_df


def prepare_positions_data_for_var(product: str, combined_pos_df: pd.DataFrame, price_df: pd.DataFrame, cob_date: str,
                                   simulation_method: str, calculation_method: str, trader: bool, counterparty: bool) \
        -> pd.DataFrame:

    base_cols = [
        'cob_date', 'product', 'unit', 'region', 'position_type', 'return_type', 'total_active_lots', 'settle_delta_1',
        'exposure', 'product_code', 'instrument_name', 'bbg_ticker', 'underlying_bbg_ticker', 'generic_curve',
        'cob_date_price', 'delta', 'delta_exposure', 'to_USD_conversion', 'currency', 'cob_date_fx'
    ]

    uat_engine = get_engine('uat')
    pos_loader = PositionLoader(source=uat_engine)

    if trader:
        if 'trader_id' not in base_cols:
            base_cols.append('trader_id')
            base_cols.append('trader_name')

    if counterparty:
        if 'counterparty_id' not in base_cols:
            base_cols.append('counterparty_id')
            base_cols.append('counterparty_parent')

    if simulation_method == 'hist_sim':
        combined_pos_df = pos_loader.assign_linear_risk_factor(combined_pos_df)
        combined_pos_df = pos_loader.assign_cob_date_price(combined_pos_df, price_df, cob_date)
    elif simulation_method == 'mc_sim':
        combined_pos_df = pos_loader.assign_risk_factor(combined_pos_df)
        na_risk_factor_pos_df = combined_pos_df[~combined_pos_df['risk_factor'].notna()]
        if len(na_risk_factor_pos_df) > 0:
            print(f"WARNING: {len(na_risk_factor_pos_df)} positions were filtered out due to a missing or None risk_factor.")

        if product == 'cotton':
            combined_pos_df = pos_loader.duplicate_basis_and_assign_ct1(combined_pos_df)
            combined_pos_df = pos_loader.assign_cob_date_price(combined_pos_df, price_df, cob_date)
            combined_pos_df['return_type'] = 'relative'
    base_cols.append('risk_factor')

    # Add cotton/rubber/rms-specific columns
    if (combined_pos_df['product'].str.lower() == 'cotton').any():
        if 'portfolio' not in base_cols:
            base_cols.append('portfolio')
        if 'book' not in base_cols:
            base_cols.append('book')
        if 'basis_series' not in base_cols:
            base_cols.append('basis_series')
    elif (combined_pos_df['product'].str.lower() == 'rubber').any():
        if 'portfolio' not in base_cols:
            base_cols.append('portfolio')
        if 'product_type' not in base_cols:
            base_cols.append('product_type')
        if 'source' not in base_cols:
            base_cols.append('source')
    elif (combined_pos_df['product'].str.lower() == 'rms').any() and calculation_method == 'taylor_series':
        if 'settle_gamma_11' not in base_cols:
            base_cols.append('settle_gamma_11')
        if 'settle_vega_1' not in base_cols:
            base_cols.append('settle_vega_1')
        if 'settle_theta' not in base_cols:
            base_cols.append('settle_theta')
        if 'gamma' not in base_cols:
            base_cols.append('gamma')
        if 'vega' not in base_cols:
            base_cols.append('vega')
        if 'theta' not in base_cols:
            base_cols.append('theta')

    # No derivatives position
    if 'DERIVS' in combined_pos_df['position_type'].unique():
        if 'product_code' not in base_cols:
            base_cols.append('product_code')
        if 'to_USD_conversion' not in base_cols:
            base_cols.append('to_USD_conversion')
        if 'lots_to_MT_conversion' not in base_cols:
            base_cols.append('lots_to_MT_conversion')

    combined_pos_df = combined_pos_df[base_cols]
    # combined_pos_df = aggregate_positions_for_pnl(combined_pos_df, product) #TODO Revisit this again in future to streamline positions.
    combined_pos_df['position_index'] = (combined_pos_df.index.map(lambda i: str(i).zfill(4)))
    # Position index is placed only after positions are prepared for either simulations (split into legs for mc_sim)
    return combined_pos_df


def aggregate_positions_for_pnl(df: pd.DataFrame, product) -> pd.DataFrame:
    """
    Performs targeted aggregation required for PnL calculation based on core risk keys.

    This function specifically handles type coercion to prevent pandas' groupby
    from failing due to mixed types, which often results in an empty DataFrame.

    Args:
        df: The raw positions DataFrame (which should already have 'risk_factor' assigned).

    Returns:
        A DataFrame grouped by risk keys with aggregated delta and delta_exposure.
    """

    # 1. Define the required columns
    if product == 'cotton':
        GROUPING_KEYS = ['instrument_name', 'book', 'region', 'portfolio', 'position_type', 'exposure', 'bbg_ticker',
                         'underlying_bbg_ticker', 'currency', 'return_type', 'risk_factor']
    elif product == 'rubber':
        GROUPING_KEYS = ['instrument_name', 'region', 'portfolio', 'position_type', 'exposure', 'bbg_ticker',
                         'underlying_bbg_ticker', 'currency', 'return_type', 'risk_factor']
    AGG_COLUMNS = ['delta', 'delta_exposure', 'cob_date_price', 'to_USD_conversion', 'cob_date_fx']
    agg_dict = {
        'delta': 'sum',
        'delta_exposure': 'first',
        'cob_date_price': 'first',
        'to_USD_conversion': 'first',
        'cob_date_fx': 'first'
    }

    # Check if all required columns are present
    all_required_cols = GROUPING_KEYS + list(agg_dict.keys())
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: The following required columns are missing from the DataFrame: {missing_cols}")
        return pd.DataFrame()  # Return empty if essential columns are missing

    # 2. Aggressive Type Coercion for Summation Columns
    # Convert summation columns to numeric. 'errors=coerce' turns any non-numeric
    # value (like 'N/A' or empty strings) into NaN.
    initial_rows = len(df)
    for col in AGG_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where the primary values to be summed are NaN (these rows are unusable)
    df.dropna(subset=AGG_COLUMNS, inplace=True)

    if len(df) < initial_rows:
        print(
            f"WARNING: Filtered {initial_rows - len(df)} rows due to non-numeric/null values in summation columns ({', '.join(AGG_COLUMNS)}).")

    # 3. Defensive Cleaning for Grouping Keys
    for col in GROUPING_KEYS:
        # Convert grouping keys to string and fill NaNs with a placeholder.
        # This guarantees consistent, hashable types for pandas.groupby.
        df[col] = df[col].astype(str).fillna('UNKNOWN')

    print(f"INFO: Grouping by keys: {GROUPING_KEYS}")
    print(f"INFO: Aggregation: Delta/Exposure=sum, cob_date_price=first")

    aggregated_df = df.groupby(GROUPING_KEYS, dropna=False).agg(agg_dict).reset_index()

    # 5. Final Indexing
    aggregated_df['pnl_agg_index'] = aggregated_df.index.map(lambda i: f"PNL_{i:05d}")

    if aggregated_df.empty:
        print("FATAL ERROR: Targeted GroupBy resulted in 0 rows even after cleaning. Check data integrity.")

    return aggregated_df
