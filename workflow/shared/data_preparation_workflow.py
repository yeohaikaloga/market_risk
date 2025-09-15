"""
DATA PREPARATION WORKFLOW
=========================

Orchestrates end-to-end preparation of position and market data for risk workflows.

Steps:
1. Generate generic price curves and returns for specified product
2. Load and combine physical + derivatives positions
3. Assign exposures, regions, generic curves, deltas
4. Validate output schema

Used by: VaR, Stress Testing, Attribution workflows.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Core loaders and generators
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from price_series_generator.generic_curve_generator import GenericCurveGenerator
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from position_loader.physical_position_loader import PhysicalPositionLoader
from utils.contract_utils import extract_instrument_name, instrument_ref_dict
from financial_calculations.returns import relative_returns
from position_loader.physical_position_loader import fy24_unit_to_cotlook_basis_origin_dict

# Specialist processors (moved from utils/)
from workflow.shared.position_processing import calculate_phys_derivs_aggs, calculate_basis_adj_and_basis_pos

# External workflow (only if reused — otherwise inline or move logic)
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow


def generate_product_generic_curves(product: str, cob_date: str, window: int) -> Dict[str, Any]:
    """
    STEP 1: Generate generic futures curves and returns for all relevant instruments.

    For each instrument (e.g., 'CT', 'VV'), generates:
    - generic_curves_df: rolled generic contract prices (CT1, CT2, ...)
    - relative_returns_df: percentage returns
    - relative_returns_$_df: dollar returns (scaled by COB price)
    - contract_to_curve_map: maps actual contract (e.g., CTZ4) to generic curve (e.g., CT1)

    Args:
        product: Product name (e.g., 'cotton')
        cob_date: Close-of-business date (YYYY-MM-DD)
        window: Lookback window for returns calculation

    Returns:
        Dict[instrument_name, Dict[str, Any]] — structured market data per instrument
    """
    prod_engine = get_engine('prod')
    days_list = get_prev_biz_days_list(cob_date, window + 1)

    # Define instruments based on product
    instruments_list = (
        ['CT', 'VV', 'CCL', 'S ', 'W ', 'C ', 'SB'] if product == 'cotton'
        else ['CT']  # Default fallback
    )

    instrument_dict = {}

    for instrument_name in instruments_list:
        # Step 1A: Load contract metadata
        derivatives_contract = DerivativesContractRefLoader(
            instrument_name=instrument_name,
            source=prod_engine
        )
        # TODO: Re-enable relevant_months when validated
        futures_contracts = derivatives_contract.load_contracts(
            mode='futures',
            relevant_months=None,
            relevant_years=None,
            relevant_options=None
        )

        # Step 1B: Load historical prices
        futures_price_loader = DerivativesPriceLoader(
            mode='futures',
            instrument_name=instrument_name,
            source=prod_engine
        )
        price_df = futures_price_loader.load_prices(
            start_date=days_list[0],
            end_date=cob_date,
            contracts=futures_contracts,
            reindex_dates=days_list,
            instrument_name=instrument_name
        )

        # Step 1C: Generate generic curves
        curve_generator = GenericCurveGenerator(
            df=price_df,
            futures_contract=derivatives_contract
        )
        max_position = 13 if instrument_name == 'CT' else 6  # CT goes up to CT13, others to xx6

        generic_curves_df, active_contracts_df = curve_generator.generate_generic_curves_df_up_to(
            max_position=max_position,
            roll_days=14,
            adjustment='ratio',
            label_prefix=instrument_name
        )

        # Clean and prepare returns
        generic_curves_df = (
            generic_curves_df
            .replace({pd.NA: np.nan})
            .astype(float)
            .round(3)
        )

        relative_returns_df = relative_returns(generic_curves_df)
        relative_returns_dollarised_df = relative_returns_df * generic_curves_df.loc[cob_date]

        # Create contract-to-curve mapping
        contract_to_curve_map = {
            v.replace(' COMB', ''): k
            for k, v in active_contracts_df.loc[cob_date].to_dict().items()
            if v is not None
        }

        # Store in instrument_dict
        instrument_dict[instrument_name] = {
            'generic_curves_df': generic_curves_df,
            'relative_returns_df': relative_returns_df,
            'relative_returns_$_df': relative_returns_dollarised_df,
            'contract_to_curve_map': contract_to_curve_map
        }

        print(f"[INFO] Generated curves for {instrument_name}. Sample:\n{generic_curves_df.head()}")

    return instrument_dict


def generate_cotton_combined_position(cob_date: str, instrument_dict: Dict[str, Any]) -> pd.DataFrame:
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
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'cotton'

    # Step 2A: Load physical positions from staging
    physical_loader = PhysicalPositionLoader(date=cob_date, source=uat_engine)
    conso_pos_df = physical_loader.load_cotton_phy_position_from_staging(cob_date=cob_date)
    conso_pos_df.columns = [col.lower() for col in conso_pos_df.columns]

    # Step 2B: Classify position types
    pos_type_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='Unknown')

    # Step 2C: Temporary exposure for basis logic (to be replaced)
    exposure_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='Unknown')

    # Step 2D: Assign instrument (CT or VV) based on unit and position
    conso_pos_df['contract'] = '-'
    for idx in conso_pos_df.index:
        unit = conso_pos_df.loc[idx, 'unit']
        position_type = conso_pos_df.loc[idx, 'position_type']
        exposure = conso_pos_df.loc[idx, 'exposure_for_old_basis']
        # China fixed outright → VV
        if unit == 'CHINA' and position_type == 'FIXED PHYS' and exposure == 'OUTRIGHT':
            conso_pos_df.loc[idx, 'contract'] = 'VV'
        else:
            conso_pos_df.loc[idx, 'contract'] = 'CT'
        # TODO: Add India logic when available

    # Step 2E: Calculate aggregates by region/contract
    phys_derivs_aggs = conso_pos_df.groupby(['region', 'contract']).apply(calculate_phys_derivs_aggs).reset_index()
    phys_derivs_aggs['outright_pos'] = phys_derivs_aggs['net_fixed_phys'] + phys_derivs_aggs['derivs']
    phys_derivs_aggs[['basis_adj', 'basis_pos']] = phys_derivs_aggs.apply(
        lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)),
        axis=1
    )

    # Step 2F: Prepare physical positions for combination
    phy_pos_df = conso_pos_df[conso_pos_df['position_type'] != 'DERIVS'].copy()
    phy_pos_df = phy_pos_df.rename(columns={'contract': 'instrument_name', 'quantity': 'delta'})

    phy_grouped_pos_df = phy_pos_df.groupby([
        'region', 'typ', 'position_type', 'terminal_month', 'instrument_name'
    ], as_index=False)['delta'].sum()

    phy_grouped_pos_df = phy_grouped_pos_df[phy_grouped_pos_df['delta'] != 0]
    phy_grouped_pos_df['subportfolio'] = phy_grouped_pos_df['typ']
    phy_grouped_pos_df['strike'] = np.nan
    phy_grouped_pos_df['books'] = 'PHYSICALS'
    phy_grouped_pos_df['trader_id'] = 1
    phy_grouped_pos_df['trader_name'] = None
    phy_grouped_pos_df['counterparty_id'] = 0
    phy_grouped_pos_df['counterparty_parent'] = None

    # Assign Bloomberg tickers and generic curves
    phy_grouped_pos_df = physical_loader.assign_bbg_tickers(phy_grouped_pos_df)
    phy_grouped_pos_df = physical_loader.assign_generic_curves(phy_grouped_pos_df, instrument_dict)

    # Add FX conversion
    phy_grouped_pos_df['to_USD_conversion'] = phy_grouped_pos_df['instrument_name'].map(
        lambda x: instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
    )

    # Split into basis and outright exposures
    basis_phy_pos_df = phy_grouped_pos_df.copy()
    basis_phy_pos_df['exposure'] = 'BASIS (NET PHYS)'
    basis_phy_pos_df = physical_loader.assign_basis_series(basis_phy_pos_df, fy24_unit_to_cotlook_basis_origin_dict)

    outright_phy_pos_df = phy_grouped_pos_df[
        phy_grouped_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES'])
    ].copy()
    outright_phy_pos_df['exposure'] = 'OUTRIGHT'

    # Create unit-region mapping for derivatives
    cotton_unit_region_mapping = dict(zip(conso_pos_df['unit'], conso_pos_df['region']))

    # Step 2G: Load and process derivatives positions
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_position(
        date=cob_date,
        trader_id='all',
        counterparty_id='all',
        product=product,
        books=None
    )

    # Attach sensitivities
    sensitivity_df = derivatives_loader.load_opera_sensitivities(
        deriv_pos_df,
        sensitivity_type='settle_delta_1',
        product=product
    )
    deriv_pos_df = derivatives_loader.assign_opera_sensitivities(
        deriv_pos_df,
        sensitivity_df,
        sensitivity_type='settle_delta_1'
    )

    # Clean and map
    deriv_pos_df = derivatives_loader.assign_bbg_tickers(deriv_pos_df)
    deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
    deriv_pos_df = derivatives_loader.assign_generic_curves(deriv_pos_df, instrument_dict)
    deriv_pos_df['instrument_name'] = deriv_pos_df['product_code'].apply(extract_instrument_name)
    deriv_pos_df = derivatives_loader.assign_cotton_unit(deriv_pos_df)
    deriv_pos_df['region'] = deriv_pos_df['unit'].map(cotton_unit_region_mapping)
    deriv_pos_df['position_type'] = 'DERIVS'
    deriv_pos_df['exposure'] = 'OUTRIGHT'

    # Add conversions
    deriv_pos_df['to_USD_conversion'] = deriv_pos_df['instrument_name'].map(
        lambda x: instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
    )
    deriv_pos_df['lots_to_MT_conversion'] = deriv_pos_df['instrument_name'].map(
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
    combined_pos_df['position_index'] = (
            product[:3] + '_L_' + str(cob_date) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )

    # Final column selection
    combined_pos_df = combined_pos_df[[
        'cob_date', 'product', 'unit', 'region', 'books', 'position_type',
        'total_active_lots', 'settle_delta_1', 'exposure', 'trader_id', 'trader_name', 'counterparty_id',
        'counterparty_parent', 'instrument_name', 'bbg_ticker', 'underlying_bbg_ticker', 'generic_curve',
        'basis_series', 'delta', 'to_USD_conversion', 'lots_to_MT_conversion', 'position_index'
    ]]

    print(f"[INFO] Combined position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df


def prepare_data_for_var(product: str, cob_date: str, window: int) \
        -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
    """
    MAIN ENTRY POINT: Prepare all data required for VaR or other risk workflows.

    Executes:
    1. Generate market data (generic curves, returns)
    2. Generate combined position data
    3. (If cotton) Generate basis absolute returns

    Args:
        product: Product name (e.g., 'cotton')
        cob_date: Close-of-business date (YYYY-MM-DD)
        window: Lookback window for returns

    Returns:
        Tuple:
        - instrument_dict: Market data per instrument
        - combined_pos_df: Validated position DataFrame
        - basis_abs_ret_df: (Optional) Basis absolute returns for cotton
    """
    print(f"[INFO] Starting data preparation for {product} on {cob_date} with {window}-day window.")

    # Step 1: Generate market data
    instrument_dict = generate_product_generic_curves(product, cob_date, window)
    print("[INFO] Step 1A: Market data generation completed.")

    # Step 2: Generate position data
    if product == 'cotton':
        combined_pos_df = generate_cotton_combined_position(cob_date, instrument_dict)
        print("[INFO] Step 1B: Position data generation completed.")

        # Step 3: Generate basis returns (cotton-specific)
        basis_abs_ret_df = fy24_cotton_basis_workflow(
            cob_date=cob_date,
            window=window,
            write_to_excel=True,
            apply_smoothing=True
        )
        basis_abs_ret_df.columns = basis_abs_ret_df.columns.str.replace(' final AR series', '', regex=False)
        basis_abs_ret_df.columns = basis_abs_ret_df.columns.str.replace(' final AR (sm) series', '', regex=False)
        print("[INFO] Step 1C: Basis returns generation completed.")
    else:
        raise NotImplementedError(f"Product '{product}' not yet supported in data preparation workflow.")

    print("[SUCCESS] Data preparation workflow completed.")
    return instrument_dict, combined_pos_df, basis_abs_ret_df
