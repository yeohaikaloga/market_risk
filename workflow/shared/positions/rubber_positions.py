import pandas as pd
import numpy as np
from typing import Dict, Any
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from position_loader.physical_position_loader import PhysicalPositionLoader
from utils.contract_utils import (load_instrument_ref_dict, month_codes, extract_instrument_from_product_code)
from utils.position_utils import calculate_phys_derivs_aggs, calculate_basis_adj_and_basis_pos
from utils.log_utils import get_logger

def generate_rubber_combined_position(cob_date: str, instrument_dict: Dict[str, Any], prices_df: pd.DataFrame,
                                      fx_spot_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_logger(__name__)
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'rubber'
    instrument_ref_dict = load_instrument_ref_dict('uat')

    # Step 2A: Load physical positions from staging
    physical_loader = PhysicalPositionLoader(date=cob_date, source=uat_engine)
    conso_pos_df = physical_loader.load_rubber_phy_position_from_staging(cob_date=cob_date)
    conso_pos_df.columns = [col.lower() for col in conso_pos_df.columns]
    conso_pos_df = conso_pos_df[['product', 'unit', 'trade type', 'transaction type', 'purchases type',
                                 'terminal month', 'delta quantity', 'party', 'trader', 'source']]
    conso_pos_df = conso_pos_df.rename(columns={'unit': 'portfolio', 'delta quantity': 'quantity',
                                                'terminal month': 'terminal_month', 'party': 'counterparty_parent',
                                                'trader': 'trader_name', 'product': 'product_type'})
    conso_pos_df['portfolio'] = conso_pos_df['portfolio'].astype(str).str.upper()
    source_list = conso_pos_df['source'].unique()
    conso_pos_df['quantity'] = pd.to_numeric(conso_pos_df['quantity'], errors='coerce')
    conso_pos_df['transaction type'] = 'PHYS'
    conso_pos_df['purchases type'] = conso_pos_df['purchases type'].astype(str).str.upper()
    conso_pos_df['purchases type'] = conso_pos_df['purchases type'].str.replace('PURCHASES', 'PURCHASE')
    conso_pos_df['typ'] = conso_pos_df['trade type'] + ' ' + conso_pos_df['purchases type']

    #TODO Adjust for 2026 to include split for Midstream - Silvertree, Renata, Sourire
    rubber_unit_region_mapping = {
        'CAMBO TRAD': 'CAMBODIA',
        'CHINA TRAD': 'CHINA',
        'CHINATRAD2': 'CHINA',
        'CHINA PROP': 'CHINA',
        'CHINA TRAD 2': 'CHINA',
        'CHINA': 'CHINA',
        'INDO TRAD': 'INDO',
        'IVC EXAT': 'AFRICA',
        'IVC TRADE': 'AFRICA',
        'MALAY TRAD': 'MALAY',
        'RE. AFRICA': 'AFRICA',
        'RUBBER SINGAPORE PROP': 'SINGAPORE PROP 1',
        'SAIC PC': 'SAIC PC',
        'SG ALTERNATIVE-BOOK': 'SINGAPORE PROP 2',
        'SG-ALTER': 'SINGAPORE PROP 2',
        'SGT4': 'SINGAPORE PROP 1',
        'SOCIETE AG': 'IVC MANUFACTURING',
        'RENATA MID': 'IVC MANUFACTURING',
        'RENATA MIDSTREAM': 'IVC MANUFACTURING',
        'THAIL TRAD': 'THAILAND 1',
        'THAIL TRAD 2': 'THAILAND 2',
        'THAILTRAD2': 'THAILAND 2',
        'THAIL TRAD 3': 'THAILAND 3',
        'THAILTRAD3': 'THAILAND 3',
        'VIETN TRAD': 'VIETNAM',
    }
    conso_pos_df['unit'] = conso_pos_df['portfolio']
    conso_pos_df['terminal_month'] = pd.to_datetime(conso_pos_df['terminal_month'], errors='coerce')
    conso_pos_df['region'] = conso_pos_df['unit'].map(rubber_unit_region_mapping)
    def adjusted_delta(product_type, delta_quantity):
        if product_type == 'PRIMARY PR':
            return delta_quantity * 0.68
        else:
            return delta_quantity
    conso_pos_df['quantity'] = conso_pos_df.apply(lambda row: adjusted_delta(row['product_type'], row['quantity']), axis=1)
    # Apply check for delta only after adjusted_delta function to account for dry factor.
    for source in source_list:
        logger.info(f"{len(conso_pos_df[conso_pos_df['source'] == source])} rubber physical positions with total delta "
                    f"{conso_pos_df[conso_pos_df['source'] == source]['quantity'].sum()} from {source}.")

    def rubber_portfolio_product_code_mapping(portfolio):
        if portfolio == 'CHINA TRAD' or portfolio == 'CHINATRAD2' or portfolio == 'CHINA':
            return 'IM RT'
        else:
            return 'CM OR'
    conso_pos_df['product_code'] = conso_pos_df.apply(
        lambda row: rubber_portfolio_product_code_mapping(row['portfolio']), axis=1)

    logger.info("STEP 2A completed")

    # Step 2B: Classify position types
    pos_type_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'FIXED STOCK', 'FIXED NONE']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]

    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='UNKNOWN')
    logger.info("STEP 2B completed")

    # Step 2C: Temporary exposure for basis logic (to be replaced)
    exposure_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'FIXED STOCK']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='UNKNOWN')
    logger.info("STEP 2C completed")

    # Step 2D: Calculate aggregates by unit/contract
    phys_derivs_aggs = conso_pos_df.groupby(['region', 'terminal_month']).apply(calculate_phys_derivs_aggs).reset_index()
    phys_derivs_aggs['outright_pos'] = phys_derivs_aggs['net_fixed_phys'] + phys_derivs_aggs['derivs']
    phys_derivs_aggs[['basis_adj', 'basis_pos']] = phys_derivs_aggs.apply(
        lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)),
        axis=1
    )
    logger.info("STEP 2D completed")

    # Step 2E: Prepare physical positions for combination
    phy_pos_df = conso_pos_df[(conso_pos_df['position_type'] != 'DERIVS') &
                              (conso_pos_df['trade type'].isin(['FIXED', 'DIFF']))].copy()
    for source in source_list:
        logger.info(f"{len(phy_pos_df[phy_pos_df['source'] == source])} rubber physical positions with total delta "
                    f"{phy_pos_df[phy_pos_df['source'] == source]['quantity'].sum()} from {source}.")
    phy_grouped_pos_df = phy_pos_df.groupby(['unit', 'region', 'typ', 'position_type', 'terminal_month', 'product_code',
                                             'counterparty_parent', 'trader_name', 'product_type', 'source'],
                                            as_index=False, dropna=False)['quantity'].sum()
    phy_grouped_pos_df = phy_grouped_pos_df.rename(columns={'quantity': 'delta'})
    print(len(phy_grouped_pos_df))
    for source in source_list:
        logger.info(f"{len(phy_grouped_pos_df[phy_grouped_pos_df['source'] == source])} rubber physical positions with total delta "
                    f"{phy_grouped_pos_df[phy_grouped_pos_df['source'] == source]['delta'].sum()} from {source}.")

    phy_grouped_pos_df = phy_grouped_pos_df[phy_grouped_pos_df['delta'] != 0]
    phy_grouped_pos_df['subportfolio'] = phy_grouped_pos_df['typ']
    phy_grouped_pos_df['strike'] = np.nan

    # Split into basis and outright exposures: All physicals positions are basis; all fixed physical positions are outright
    outright_phy_pos_df = phy_grouped_pos_df[
        phy_grouped_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'FIXED STOCK', 'FIXED NONE'])].copy()
    outright_phy_pos_df['exposure'] = 'OUTRIGHT'

    # Assign Bloomberg tickers and generic curves
    outright_phy_pos_df['instrument_name'] = (
        outright_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )
    outright_phy_pos_df = physical_loader.assign_bbg_tickers(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df['underlying_bbg_ticker'] = outright_phy_pos_df['bbg_ticker']
    outright_phy_pos_df = physical_loader.assign_generic_curves(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df = physical_loader.assign_cob_date_price(outright_phy_pos_df, prices_df, cob_date)
    outright_phy_pos_df['to_USD_conversion'] = outright_phy_pos_df['product_code'].map(
        lambda x: instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
    )
    outright_phy_pos_df['currency'] = 'USD'
    outright_phy_pos_df['cob_date_fx'] = 1

    basis_phy_pos_df = phy_grouped_pos_df.copy()
    basis_phy_pos_df['exposure'] = 'BASIS (NET PHYS)'
    basis_phy_pos_df['instrument_name'] = (
        basis_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )

    basis_phy_pos_df['terminal_month'] = pd.to_datetime(basis_phy_pos_df['terminal_month'], errors='coerce')
    basis_phy_pos_df['bbg_ticker'] = basis_phy_pos_df.apply(
        lambda row: (
                row['product_code']
                + [k for k, v in month_codes.items() if v == row['terminal_month'].month][0]
                + str(row['terminal_month'].year)[-1]
                + " Comdty"
        ),
        axis=1
    )
    basis_phy_pos_df['to_USD_conversion'] = basis_phy_pos_df['product_code'].map(
        lambda x: instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
    )
    basis_phy_pos_df['underlying_bbg_ticker'] = basis_phy_pos_df['bbg_ticker']
    basis_phy_pos_df = physical_loader.assign_generic_curves(basis_phy_pos_df, instrument_dict)  # for MC VaR
    basis_phy_pos_df['cob_date_price'] = 1
    basis_phy_pos_df['currency'] = 'USD'
    basis_phy_pos_df['cob_date_fx'] = 1

    logger.info("STEP 2E completed")

    # Step 2F: Load and process derivatives positions
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
        deriv_pos_df['position_type'] = 'DERIVS'
        deriv_pos_df['derivative_type'] = deriv_pos_df['derivative_type'].str.upper()
        deriv_pos_df['exposure'] = 'OUTRIGHT'
        deriv_pos_df['unit'] = deriv_pos_df['portfolio']
        deriv_pos_df['region'] = deriv_pos_df['unit'].map(rubber_unit_region_mapping)

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
        deriv_pos_df['currency'] = 'USD'
        deriv_pos_df['cob_date_fx'] = 1
    logger.info(f"STEP 2F completed. Shape of deriv_pos_df: {deriv_pos_df.shape}")

    # Step 2G: Combine all positions
    combined_pos_df = pd.concat(
        [deriv_pos_df, outright_phy_pos_df, basis_phy_pos_df],
        axis=0,
        ignore_index=True
    )
    combined_pos_df = combined_pos_df.reset_index(drop=True)
    combined_pos_df['product'] = product
    combined_pos_df['cob_date'] = cob_date
    combined_pos_df = combined_pos_df[
        combined_pos_df['region'].apply(lambda x: isinstance(x, str))
    ]
    combined_pos_df['return_type'] = 'relative'
    combined_pos_df['delta_exposure'] = combined_pos_df['delta']

    logger.info(f"STEP 2G: Combined {product} position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df