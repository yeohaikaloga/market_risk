import pandas as pd
import numpy as np
from typing import Dict, Any

from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from position_loader.physical_position_loader import PhysicalPositionLoader
from utils.contract_utils import (load_instrument_ref_dict, extract_instrument_from_product_code,
                                  get_USD_MT_conversion_from_product_code)
from position_loader.physical_position_loader import fy24_unit_to_cotlook_basis_origin_dict
from utils.position_utils import calculate_phys_derivs_aggs, calculate_basis_adj_and_basis_pos
from utils.log_utils import get_logger

def generate_cotton_combined_position(cob_date: str, instrument_dict: Dict[str, Any], prices_df: pd.DataFrame,
                                      fx_spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 2: Generate combined physical + derivatives position DataFrame for cotton.

    Business Logic:
    - Physicals: FIXED PHYS, DIFF PHYS → tagged as BASIS (NET PHYS) or OUTRIGHT
    - Derivatives: FUTURES, OPTIONS → tagged as OUTRIGHT
    - Region mapping from physicals applied to derivatives
    - Delta calculated in MT, scaled to USD

    Args:
        cob_date: Close-of-business date (YYYY-MM-DD)
        instrument_dict: Output from generate_product_generic_curves()

    Returns:
        pd.DataFrame with validated schema for risk workflows.
        Required columns: cob_date, product, unit, region, exposure, instrument_name, generic_curve, delta, ...
    """
    logger = get_logger(__name__)
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'cotton'
    instrument_ref_dict = load_instrument_ref_dict('uat')

    # Step 2A: Load physical positions from staging
    physical_loader = PhysicalPositionLoader(date=cob_date, source=uat_engine)
    conso_pos_df = physical_loader.load_cotton_phy_position_from_staging(cob_date=cob_date)
    conso_pos_df.columns = [col.lower() for col in conso_pos_df.columns]
    logger.info("STEP 2A completed")

    # Step 2B: Classify position types
    pos_type_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='UNKNOWN')
    logger.info("STEP 2B completed")

    # Step 2C: Temporary exposure for basis logic (to be replaced)
    exposure_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='UNKNOWN')
    logger.info("STEP 2C completed")

    # Step 2D: Assign instrument based on unit and position
    conso_pos_df['contract'] = '-'
    for idx in conso_pos_df.index:
        unit = conso_pos_df.loc[idx, 'unit']
        position_type = conso_pos_df.loc[idx, 'position_type']
        exposure = conso_pos_df.loc[idx, 'exposure_for_old_basis']
        # China adjustment: Fixed outright → VV; India adjustment: Fixed outright → S6
        if position_type == 'FIXED PHYS' and exposure == 'OUTRIGHT':
            if unit == 'CHINA':
                conso_pos_df.loc[idx, 'product_code'] = 'CM VV'
            elif unit == 'INDIA':
                conso_pos_df.loc[idx, 'product_code'] = 'EX GIN S6'
            else:
                conso_pos_df.loc[idx, 'product_code'] = 'CM CT'
        else:
            conso_pos_df.loc[idx, 'product_code'] = 'CM CT'
    logger.info("STEP 2D completed")

    # Step 2E: Calculate aggregates by region/product_code
    phys_derivs_aggs = conso_pos_df.groupby(['region', 'product_code']).apply(calculate_phys_derivs_aggs).reset_index()
    phys_derivs_aggs['outright_pos'] = phys_derivs_aggs['net_fixed_phys'] + phys_derivs_aggs['derivs']
    phys_derivs_aggs[['basis_adj', 'basis_pos']] = phys_derivs_aggs.apply(
        lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)),
        axis=1
    )
    logger.info("STEP 2E completed")

    # Step 2F: Prepare physical positions for combination
    phy_pos_df = conso_pos_df[conso_pos_df['position_type'] != 'DERIVS'].copy()
    phy_pos_df = phy_pos_df.rename(columns={'quantity': 'delta'})

    phy_grouped_pos_df = phy_pos_df.groupby([
        'region', 'typ', 'position_type', 'terminal_month', 'product_code'
    ], as_index=False)['delta'].sum()

    phy_grouped_pos_df = phy_grouped_pos_df[phy_grouped_pos_df['delta'] != 0]
    phy_grouped_pos_df['subportfolio'] = phy_grouped_pos_df['typ']
    phy_grouped_pos_df['strike'] = np.nan
    phy_grouped_pos_df['book'] = 'PHYSICALS'
    phy_grouped_pos_df['trader_id'] = 1
    phy_grouped_pos_df['trader_name'] = None
    phy_grouped_pos_df['counterparty_id'] = 0
    phy_grouped_pos_df['counterparty_parent'] = None
    phy_grouped_pos_df['currency'] = np.where(phy_grouped_pos_df['product_code'] == 'EX GIN S6', 'INR', 'USD')

    # Add conversion of physicals from c/lbs to USD/MT
    phy_grouped_pos_df['to_USD_conversion'] = phy_grouped_pos_df.apply(
        lambda row: get_USD_MT_conversion_from_product_code(row['product_code'], instrument_ref_dict),
        axis=1
    )

    # Split into basis and outright exposures: All physicals positions are basis; all fixed physical positions are outright
    basis_phy_pos_df = phy_grouped_pos_df.copy()
    basis_phy_pos_df['exposure'] = 'BASIS (NET PHYS)'

    outright_phy_pos_df = phy_grouped_pos_df[
        phy_grouped_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES'])
    ].copy()
    outright_phy_pos_df['exposure'] = 'OUTRIGHT'

    # India adjustment: If India DIFF PHYS =/= 0, split into respective legs and classify them as OUTRIGHT, remove the original DIFF PHYS position
    india_diff_phy_df = basis_phy_pos_df[(basis_phy_pos_df['region'] == 'INDIA') &
                                         (basis_phy_pos_df['position_type'] == 'DIFF PHYS')]
    india_pos_df = pd.DataFrame()
    if len(india_diff_phy_df) > 0:
        print(india_diff_phy_df)
        for index, row in india_diff_phy_df.iterrows():
            base_leg = row.to_dict()

            # PHYSICAL LEG
            phys_leg = base_leg.copy()
            phys_leg.update({
                'product_code': 'EX GIN S6',
                'exposure': 'OUTRIGHT',
                'position_type': 'DIFF PHYS',
                'delta': row['delta'],
                'to_USD_conversion': instrument_ref_dict['CM CCL']['to_USD_conversion'],
                'currency': 'INR'
            })

            # DERIVATIVES LEG
            derivs_leg = base_leg.copy()
            derivs_leg.update({
                'product_code': 'CM CT',
                'exposure': 'OUTRIGHT',
                'position_type': 'DERIVS',
                'delta': -row['delta'],  # negated
                'to_USD_conversion': instrument_ref_dict['CM CCL']['to_USD_conversion'],
                'currency': 'USD'
            })
            both_india_legs_df = pd.DataFrame([phys_leg, derivs_leg])
            india_pos_df = pd.concat([india_pos_df, both_india_legs_df], ignore_index=True)
        outright_phy_pos_df = pd.concat([outright_phy_pos_df, india_pos_df], ignore_index=True)
    else:
        print('India has no basis position')
    basis_phy_pos_df = basis_phy_pos_df[basis_phy_pos_df['region'] != 'INDIA']

    # Assign Bloomberg tickers and generic curves
    outright_phy_pos_df['instrument_name'] = (
        outright_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )

    outright_phy_pos_df['lots_to_MT_conversion'] = 1
    outright_phy_pos_df = physical_loader.assign_bbg_tickers(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df = physical_loader.assign_generic_curves(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df = physical_loader.assign_cob_date_price(outright_phy_pos_df, prices_df, cob_date)
    outright_phy_pos_df = physical_loader.assign_cob_date_fx(outright_phy_pos_df, fx_spot_df, cob_date)

    basis_phy_pos_df['instrument_name'] = (
        basis_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )
    basis_phy_pos_df['lots_to_MT_conversion'] = 1
    basis_phy_pos_df = physical_loader.assign_bbg_tickers(basis_phy_pos_df, instrument_dict) # for MC VaR
    basis_phy_pos_df = physical_loader.assign_generic_curves(basis_phy_pos_df, instrument_dict) # for MC VaR
    basis_phy_pos_df['cob_date_price'] = 1
    basis_phy_pos_df = physical_loader.assign_basis_series(basis_phy_pos_df, fy24_unit_to_cotlook_basis_origin_dict)
    basis_phy_pos_df['cob_date_fx'] = 1
    # Create unit-region mapping for derivatives
    cotton_unit_region_mapping = dict(zip(conso_pos_df['unit'], conso_pos_df['region']))
    logger.info("STEP 2F completed")

    # Step 2G: Load and process derivatives positions
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_position(
        date=cob_date,
        trader_id='all',
        counterparty_id='all',
        product=product,
        book=None
    )
    deriv_pos_df = deriv_pos_df[~deriv_pos_df['security_id'].astype(str).str.startswith('CR')]

    # Attach sensitivities
    sensitivity_df = derivatives_loader.load_opera_sensitivities(
        deriv_pos_df,
        sensitivity_types=['settle_delta_1'],
        product=product
    )
    deriv_pos_df = derivatives_loader.assign_opera_sensitivities(
        deriv_pos_df,
        sensitivity_df,
        sensitivity_types=['settle_delta_1']
    )

    # Clean and map
    if len(deriv_pos_df) != 0:
        deriv_pos_df = derivatives_loader.assign_bbg_tickers(deriv_pos_df)
        deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
        deriv_pos_df['instrument_name'] = (
            deriv_pos_df['product_code']
            .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
        )
        deriv_pos_df = derivatives_loader.assign_generic_curves(deriv_pos_df, instrument_dict)
        deriv_pos_df = derivatives_loader.assign_cob_date_price(deriv_pos_df, prices_df, cob_date)
        deriv_pos_df['product_code'] = deriv_pos_df['security_id'].str.split().str[:2].str.join(' ')
        deriv_pos_df = derivatives_loader.assign_cotton_unit(deriv_pos_df)
        deriv_pos_df['region'] = deriv_pos_df['unit'].map(cotton_unit_region_mapping)
        deriv_pos_df['position_type'] = 'DERIVS'
        deriv_pos_df['exposure'] = 'OUTRIGHT'
        deriv_pos_df = deriv_pos_df.rename(columns={'books': 'book'})

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
            deriv_pos_df['settle_delta_1'] *
            deriv_pos_df['lots_to_MT_conversion']
        )
        deriv_pos_df['cob_date_fx'] = 1
    logger.info("STEP 2G completed")

    # Step 2H: Combine all positions
    combined_pos_df = pd.concat(
        [deriv_pos_df, outright_phy_pos_df, basis_phy_pos_df],
        axis=0,
        ignore_index=True
    )

    combined_pos_df['product'] = product
    combined_pos_df['cob_date'] = cob_date
    combined_pos_df = combined_pos_df[
        combined_pos_df['region'].apply(lambda x: isinstance(x, str))
    ]

    combined_pos_df = combined_pos_df.reset_index(drop=True)
    combined_pos_df['position_index'] = (
            product[:3] + '_L_' + str(cob_date) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )

    logger.info(f"Step 2G: Combined {product} position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df