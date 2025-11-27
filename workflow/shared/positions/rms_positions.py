import pandas as pd
import numpy as np
from typing import Dict, Any
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from utils.contract_utils import (load_instrument_ref_dict, extract_instrument_from_product_code)
from utils.log_utils import get_logger

def generate_rms_combined_position(cob_date: str, instrument_dict: Dict[str, Any]) -> pd.DataFrame:
    logger = get_logger(__name__)
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'rms'
    instrument_ref_dict = load_instrument_ref_dict('uat')

    # Step 2A: Load and process derivatives positions
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_position(
        date=cob_date,
        trader_id=None,
        counterparty_id=None,
        product=product,
        book=None
    )
    deriv_pos_df = deriv_pos_df[~deriv_pos_df['security_id'].astype(str).str.startswith('CR')]

    # Attach sensitivities
    all_sensitivity_types = [
        'settle_delta_1',
        'settle_delta_2',
        'settle_gamma_11',
        'settle_gamma_12',
        'settle_gamma_21',
        'settle_gamma_22',
        'settle_vega_1',
        'settle_vega_2',
        'settle_theta',
        'settle_chi'
    ]

    sensitivity_df = derivatives_loader.load_opera_sensitivities(
        deriv_pos_df,
        sensitivity_types=all_sensitivity_types,
        product=product
    )
    deriv_pos_df = derivatives_loader.assign_opera_sensitivities(
        deriv_pos_df,
        sensitivity_df,
        sensitivity_types=all_sensitivity_types
    )
    logger.info("STEP 2A completed")

    # Clean and map
    if len(deriv_pos_df) != 0:
        deriv_pos_df['product_code'] = deriv_pos_df['security_id'].str.split().str[:2].str.join(' ')
        deriv_pos_df = derivatives_loader.assign_bbg_tickers(deriv_pos_df)
        deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
        deriv_pos_df = derivatives_loader.assign_generic_curves(deriv_pos_df, instrument_dict)
        #deriv_pos_df['instrument_name'] = deriv_pos_df['product_code'].apply(extract_instrument_from_product_code)
        #print(deriv_pos_df['instrument_name'].unique())
        deriv_pos_df['position_type'] = 'DERIVS'
        deriv_pos_df['exposure'] = 'OUTRIGHT'
        deriv_pos_df['unit'] = deriv_pos_df['portfolio']
        deriv_pos_df['region'] = deriv_pos_df['portfolio']
        deriv_pos_df['instrument_name'] = (
            deriv_pos_df['product_code']
            .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
        )
        # Add conversions
        deriv_pos_df['to_USD_conversion'] = deriv_pos_df['product_code'].map(
            lambda x: instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
        )
        deriv_pos_df['lots_to_MT_conversion'] = deriv_pos_df['product_code'].map(
            lambda x: instrument_ref_dict.get(x, {}).get('lots_to_MT_conversion', np.nan)
        )
        deriv_pos_df['conversion_factor'] = (
                deriv_pos_df['to_USD_conversion'] * deriv_pos_df['lots_to_MT_conversion']
        )
        deriv_pos_df['delta'] = (
                deriv_pos_df['total_active_lots'] *
                (deriv_pos_df['settle_delta_1'] + deriv_pos_df['settle_delta_2']) *
                deriv_pos_df['lots_to_MT_conversion'] *
                deriv_pos_df['to_USD_conversion']
        )
        deriv_pos_df['gamma'] = (
                deriv_pos_df['total_active_lots'] *
                (deriv_pos_df['settle_gamma_11'] + deriv_pos_df['settle_gamma_12'] + deriv_pos_df['settle_gamma_21'] +
                 deriv_pos_df['settle_gamma_22']) *
                deriv_pos_df['to_USD_conversion'] ** 2
        )
        deriv_pos_df['vega'] = (
                deriv_pos_df['total_active_lots'] *
                (deriv_pos_df['settle_vega_1'] + deriv_pos_df['settle_vega_2'])
        )
        deriv_pos_df['theta'] = (
                deriv_pos_df['total_active_lots'] *
                deriv_pos_df['settle_theta'] * 5/7
        )
    logger.info("STEP 2B completed")

    # Step 2C: Combine all positions
    combined_pos_df = pd.concat(
        [deriv_pos_df],
        axis=0,
        ignore_index=True
    )
    combined_pos_df = combined_pos_df.reset_index(drop=True)
    combined_pos_df['product'] = product
    combined_pos_df['cob_date'] = cob_date
    combined_pos_df['position_index'] = (
            product[:3] + '_L_' + str(cob_date) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )

    logger.info(f"STEP 2C completed: Combined {product} position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df