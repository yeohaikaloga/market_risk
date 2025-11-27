import pandas as pd
from db.db_connection import get_engine
from position_loader.position_loader import PositionLoader
from workflow.shared.positions.cotton_positions import generate_cotton_combined_position
from workflow.shared.positions.rubber_positions import generate_rubber_combined_position
from workflow.shared.positions.rms_positions import generate_rms_combined_position
from utils.log_utils import get_logger


def build_combined_position(cob_date, product, instrument_dict, prices_df):
    """
    Build combined position DataFrame for one or multiple product groups.

    Parameters
    ----------
    product : str
        'cotton', 'rubber', 'rms', or 'all'

    Returns
    -------
    pd.DataFrame
        Combined position dataframe.
    """

    if product == 'cotton':
        combined_pos_df = generate_cotton_combined_position(cob_date, instrument_dict, prices_df)
    elif product == 'rubber':
        combined_pos_df = generate_rubber_combined_position(cob_date, instrument_dict, prices_df)
    elif product == 'rms':
        combined_pos_df = generate_rms_combined_position(cob_date, instrument_dict, prices_df)
    elif product == 'all':
        combined_pos_df = generate_rms_combined_position(cob_date, instrument_dict, prices_df)
    else:
        NotImplementedError(f"Product '{product}' not yet supported in data preparation workflow.")
    return combined_pos_df

def prepare_positions_data_for_var(combined_pos_df: pd.DataFrame, method: str, trader: bool, counterparty: bool) -> pd.DataFrame:

    base_cols = [
        'cob_date', 'product', 'unit', 'position_type',
        'total_active_lots', 'settle_delta_1', 'exposure', 'product_code', 'instrument_name', 'bbg_ticker',
        'underlying_bbg_ticker', 'generic_curve', 'cob_date_price', 'delta', 'position_index',
        'to_USD_conversion'
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

    if method == 'linear':
        combined_pos_df = pos_loader.assign_linear_var_map(combined_pos_df)
        base_cols.append('linear_var_map')

    elif method == 'non-linear_monte_carlo':
        combined_pos_df = pos_loader.assign_monte_carlo_var_risk_factor(combined_pos_df)
        combined_pos_df = pos_loader.duplicate_basis_and_assign_ct1(combined_pos_df)
        base_cols.append('monte_carlo_var_risk_factor')

    # Add cotton/rubber/rms-specific columns
    if (combined_pos_df['product'].str.lower() == 'cotton').any():
        if 'portfolio' not in base_cols:
            base_cols.append('portfolio')
        if 'region' not in base_cols:
            base_cols.append('region')
        if 'book' not in base_cols:
            base_cols.append('book')
        if 'basis_series' not in base_cols:
            base_cols.append('basis_series')
    elif (combined_pos_df['product'].str.lower() == 'rubber').any():
        if 'portfolio' not in base_cols:
            base_cols.append('portfolio')
        if 'product_type' not in base_cols:
            base_cols.append('product_type')
        if 'region' not in base_cols:
            base_cols.append('region')
        if 'source' not in base_cols:
            base_cols.append('source')
    elif (combined_pos_df['product'].str.lower() == 'rms').any():
        if 'region' not in base_cols:
            base_cols.append('region')
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

    combined_pos_df['position_index'] = (
        combined_pos_df['product'].str[:3] + '_L_' +
        combined_pos_df['cob_date'].astype(str) + '_' +
        combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )
    return combined_pos_df