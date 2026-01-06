import pandas as pd
from typing import Dict, Any
from db.db_connection import get_engine
from position_loader.physical_position_loader import PhysicalPositionLoader
from utils.log_utils import get_logger

logger = get_logger(__name__)
def generate_wood_combined_position(cob_date: str, instrument_dict: Dict[str, Any], prices_df, fx_spot_df: pd.DataFrame) \
        -> pd.DataFrame:


    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'wood'

    physical_loader = PhysicalPositionLoader(date=cob_date, source=uat_engine)
    outright_phy_pos_df = physical_loader.load_wood_phy_position_from_staging(cob_date=cob_date)
    if len(outright_phy_pos_df) > 0:
        outright_phy_pos_df = outright_phy_pos_df.rename(columns={'product': 'product_type'})
        outright_phy_pos_df['product'] = 'wood'
        outright_phy_pos_df['instrument_name'] = 'WOOD AVG'
        outright_phy_pos_df['unit'] = None
        outright_phy_pos_df['region'] = outright_phy_pos_df['product_type']
        outright_phy_pos_df['total_active_lots'] = None
        outright_phy_pos_df['settle_delta_1'] = None
        outright_phy_pos_df['product_code'] = outright_phy_pos_df['instrument_name']
        outright_phy_pos_df['bbg_ticker'] = None
        outright_phy_pos_df['underlying_bbg_ticker'] = None
        outright_phy_pos_df['generic_curve'] = outright_phy_pos_df['instrument_name']
        outright_phy_pos_df = physical_loader.assign_cob_date_price(outright_phy_pos_df, prices_df, cob_date)
        outright_phy_pos_df['position_type'] = 'FIXED PHYSICALS'
        outright_phy_pos_df['cob_date'] = cob_date
        outright_phy_pos_df['lots_to_MT_conversion'] = 1
        outright_phy_pos_df['to_USD_conversion'] = 1
        outright_phy_pos_df['exposure'] = 'OUTRIGHT'
        outright_phy_pos_df['currency'] = 'EUR'
        outright_phy_pos_df = physical_loader.assign_cob_date_fx(outright_phy_pos_df, fx_spot_df, cob_date)
        outright_phy_pos_df['return_type'] = 'relative'
        outright_phy_pos_df['delta_exposure'] = outright_phy_pos_df['delta']
        return outright_phy_pos_df
    else:
        logger.warning("No wood positions found.")
        return pd.DataFrame()