import pandas as pd
import numpy as np
from typing import Dict, Any
from sqlalchemy import text
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from utils.contract_utils import (load_instrument_ref_dict, extract_instrument_from_product_code)
from utils.log_utils import get_logger
from workflow.shared.forex_workflow import load_forex
import re

logger = get_logger(__name__)
def generate_rms_combined_position(cob_date: str, instrument_dict: Dict[str, Any], prices_df: pd.DataFrame,
                                   fx_spot_df: pd.DataFrame) -> pd.DataFrame:

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
        deriv_pos_df['derivative_type'] = deriv_pos_df['derivative_type'].str.upper()
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
                deriv_pos_df['lots_to_MT_conversion'] *
                deriv_pos_df['to_USD_conversion']
        )
        deriv_pos_df['vega'] = (
                deriv_pos_df['total_active_lots'] *
                (deriv_pos_df['settle_vega_1'] + deriv_pos_df['settle_vega_2'])
        )
        deriv_pos_df['theta'] = (
                deriv_pos_df['total_active_lots'] *
                deriv_pos_df['settle_theta'] * 5/7
        )
        deriv_pos_df = derivatives_loader.assign_cob_date_price(deriv_pos_df, prices_df, cob_date)
        deriv_pos_df['cob_date_fx'] = 1
    logger.info("STEP 2B completed")

    # Step 2C: Combine all positions
    # combined_pos_df = pd.concat(
    #     [deriv_pos_df],
    #     axis=0,
    #     ignore_index=True
    # )

    deriv_pos_df['product'] = product
    deriv_pos_df['cob_date'] = cob_date
    deriv_pos_df['return_type'] = 'relative'
    deriv_pos_df['delta_exposure'] = deriv_pos_df['delta']

    logger.info(f"STEP 2C completed: Combined {product} position DataFrame generated. Shape: {deriv_pos_df.shape}")
    print(deriv_pos_df.head())

    return deriv_pos_df


def generate_rms_combined_position_from_screen(cob_date: str, instrument_dict: Dict[str, Any], prices_df: pd.DataFrame,
                                               fx_spot_df: pd.DataFrame) -> pd.DataFrame:
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    instrument_ref_dict = load_instrument_ref_dict('uat')
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_rms_screen()
    deriv_pos_df = derivatives_loader.assign_risk_factors_for_rms_screen(deriv_pos_df, prices_df)
    deriv_pos_df = deriv_pos_df.rename(columns={'invenio_product_code': 'product_code', 'delta_screen': 'delta',
                                                'gamma_screen': 'gamma', 'theta_screen': 'theta',
                                                'vega_screen': 'vega'})
    deriv_pos_df['instrument_name'] = (
        deriv_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )
    deriv_pos_df['region'] = deriv_pos_df['subportfolio']
    deriv_pos_df['generic_curve'] = deriv_pos_df['risk_factor']
    deriv_pos_df['product'] = 'rms'
    deriv_pos_df['cob_date'] = cob_date
    deriv_pos_df['cob_date_fx'] = 1
    deriv_pos_df['position_type'] = 'DERIVS'
    deriv_pos_df['exposure'] = 'OUTRIGHT'
    deriv_pos_df['return_type'] = 'relative'
    deriv_pos_df = derivatives_loader.assign_cob_date_price(deriv_pos_df, prices_df, cob_date)
    deriv_pos_df['to_USD_conversion'] = 1
    deriv_pos_df['currency'] = deriv_pos_df['product_code'].apply(
        lambda x: instrument_ref_dict.get(x, {}).get('currency').upper()
    )
    deriv_pos_df["delta"] = pd.to_numeric(deriv_pos_df["delta"], errors='coerce').fillna(0.0)
    deriv_pos_df["gamma"] = pd.to_numeric(deriv_pos_df["gamma"], errors='coerce').fillna(0.0)
    deriv_pos_df['delta_exposure'] = deriv_pos_df['delta']
    return deriv_pos_df