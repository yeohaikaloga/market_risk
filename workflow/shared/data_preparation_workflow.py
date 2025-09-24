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
from contract_ref_loader.physical_contract_ref_loader import PhysicalContractRefLoader
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from price_series_loader.physical_price_loader import PhysicalPriceLoader
from price_series_generator.generic_curve_generator import GenericCurveGenerator
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from position_loader.physical_position_loader import PhysicalPositionLoader
from position_loader.position_loader import PositionLoader
from utils.contract_utils import extract_instrument_name, instrument_ref_dict
from financial_calculations.returns import relative_returns
from position_loader.physical_position_loader import fy24_unit_to_cotlook_basis_origin_dict

# Specialist processors (moved from utils/)
from workflow.shared.position_processing import calculate_phys_derivs_aggs, calculate_basis_adj_and_basis_pos

# External workflow (only if reused — otherwise inline or move logic)
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow
from contract_ref_loader.physical_contract_ref_loader import crop_dict


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
        # TODO: Re-enable relevant_months when validated for CT
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
        relative_returns_df = relative_returns_df.fillna(0)
        relative_returns_dollarised_df = relative_returns_df * generic_curves_df.loc[cob_date]

        # TODO: Temporary FOREX if-else clause. Need proper fix HERE.
        if instrument_ref_dict[instrument_name]['currency'] == 'CNY':
            relative_returns_dollarised_df = relative_returns_dollarised_df / 1

        # Create contract-to-curve mapping
        contract_to_curve_map = {}
        for k, v in active_contracts_df.loc[cob_date].to_dict().items():
            if v is not None:
                contract = v.replace(' COMB', '')
                if contract not in contract_to_curve_map:
                    contract_to_curve_map[contract] = k

        # Store in instrument_dict
        instrument_dict[instrument_name] = {
            'generic_curves_df': generic_curves_df,
            'relative_returns_df': relative_returns_df,
            'relative_returns_$_df': relative_returns_dollarised_df,
            'contract_to_curve_map': contract_to_curve_map
        }

        print(f"[INFO] Generated curves for {instrument_name}. Sample:\n{generic_curves_df.head()}")

    return instrument_dict

def generate_ex_gin_s6_returns_df(cob_date: str, window: int) -> pd.DataFrame:
    uat_engine = get_engine('uat')
    days_list = get_prev_biz_days_list(cob_date, window + 1)
    ex_gin_s6 = PhysicalPriceLoader(instrument_name='EX GIN S6', source=uat_engine)
    ex_gin_s6_df = ex_gin_s6.load_ex_gins6_prices_from_staging(start_date=days_list[0], end_date=cob_date, data_source='EX GIN S6')
    ex_gin_s6_df['date'] = pd.to_datetime(ex_gin_s6_df['date'])
    ex_gin_s6_df = ex_gin_s6_df.set_index('date')[['price']]
    ex_gin_s6_relative_returns_df = relative_returns(ex_gin_s6_df)
    print(ex_gin_s6_df.tail())
    print(ex_gin_s6_df.loc[cob_date])
    # Rs/Candy to Rs/MT conversion factor : 1000 / 355.56
    ex_gin_s6_relative_returns_df['relative_returns_INR'] = (ex_gin_s6_relative_returns_df['price'] * ex_gin_s6_df.loc[cob_date, 'price'])
    ex_gin_s6_relative_returns_df['relative_returns_$'] = ex_gin_s6_relative_returns_df['relative_returns_INR']
    # TODO: Insert FOREX conversion here (USDINR hardcoded as 87.5275)
    return ex_gin_s6_relative_returns_df

def generate_cotlook_returns_df(cob_date: str, window: int) -> dict:
    prod_engine = get_engine('prod')
    biz_days = pd.DatetimeIndex(get_prev_biz_days_list(date=cob_date, no_of_days=window))
    start_date = biz_days[0]

    cif_crops = ['Burkina Faso Bola/s', 'Brazilian', 'Ivory Coast Manbo/s', 'Mali Juli/s', 'Memphis/Orleans/Texas',
                 'A Index']
    physical_contracts_and_prices = []
    cotlook_dict = {}
    for crop_name in cif_crops:
        crop_params = crop_dict.get(crop_name, {})
        crop_params['data_source'] = 'cotlook'
        physical_contract = PhysicalContractRefLoader(instrument_name=crop_name, source=prod_engine, params=crop_params)
        physical_contract.load_ref_data()
        crop_price = PhysicalPriceLoader(instrument_name=crop_name, source=prod_engine, params=crop_params)
        crop_price_df = crop_price.load_prices(start_date=start_date, end_date=cob_date, data_source='cotlook')
        crop_price_df_by_year = crop_price_df.pivot(index='tdate', columns='crop_year', values='px_settle')
        physical_contracts_and_prices.append((physical_contract, crop_price_df_by_year))
        stitched_crop_series = pd.DataFrame(index=crop_price_df_by_year.index)
        for crop_year in crop_price_df_by_year.columns:
            if physical_contract.crop_year_type == 'cross':
                ref_year, ref_next_year = map(int, crop_year.split("/"))
            else:
                ref_year = int(crop_year)
                ref_next_year = ref_year + 1
            crop_year_start = pd.Timestamp(f"{ref_year}-08-01")
            crop_year_end = pd.Timestamp(f"{ref_next_year}-07-31")
            for i in crop_price_df_by_year.index:
                if crop_year_start <= i <= crop_year_end:
                    stitched_crop_series.loc[i, physical_contract.instrument_name] = crop_price_df_by_year.loc[i, crop_year]
            stitched_crop_series = stitched_crop_series.reindex(biz_days, method='ffill')
            stitched_crop_series_relative_returns_df = relative_returns(stitched_crop_series)
            stitched_crop_series_relative_returns_dollarised_df = (stitched_crop_series_relative_returns_df *
                                    stitched_crop_series.loc[cob_date, physical_contract.instrument_name])
        cotlook_dict[physical_contract.instrument_name] = stitched_crop_series_relative_returns_dollarised_df
    return cotlook_dict


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
    print('Step 2A completed')

    # Step 2B: Classify position types
    pos_type_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='Unknown')
    print('Step 2B completed')

    # Step 2C: Temporary exposure for basis logic (to be replaced)
    exposure_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='Unknown')
    print('Step 2C completed')

    # Step 2D: Assign instrument (CT or VV) based on unit and position
    conso_pos_df['contract'] = '-'
    for idx in conso_pos_df.index:
        unit = conso_pos_df.loc[idx, 'unit']
        position_type = conso_pos_df.loc[idx, 'position_type']
        exposure = conso_pos_df.loc[idx, 'exposure_for_old_basis']
        # China adjustment: Fixed outright → VV; India adjustment: Fixed outright → S6
        if position_type == 'FIXED PHYS' and exposure == 'OUTRIGHT':
            if unit == 'CHINA':
                conso_pos_df.loc[idx, 'contract'] = 'VV'
            elif unit == 'INDIA':
                conso_pos_df.loc[idx, 'contract'] = 'EX GIN S6'
            else:
                conso_pos_df.loc[idx, 'contract'] = 'CT'
        else:
            conso_pos_df.loc[idx, 'contract'] = 'CT'
    print('Step 2D completed')

    # Step 2E: Calculate aggregates by region/contract
    phys_derivs_aggs = conso_pos_df.groupby(['region', 'contract']).apply(calculate_phys_derivs_aggs).reset_index()
    phys_derivs_aggs['outright_pos'] = phys_derivs_aggs['net_fixed_phys'] + phys_derivs_aggs['derivs']
    phys_derivs_aggs[['basis_adj', 'basis_pos']] = phys_derivs_aggs.apply(
        lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)),
        axis=1
    )
    print('Step 2E completed')

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

    # Add FX conversion
    phy_grouped_pos_df['to_USD_conversion'] = phy_grouped_pos_df['instrument_name'].map(
        lambda x: instrument_ref_dict.get('CCL', {}).get('to_USD_conversion', np.nan)
        if x == 'EX GIN S6' else instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
    )

    # Split into basis and outright exposures: All physicals positions are basis; all fixed physical positions are outright
    basis_phy_pos_df = phy_grouped_pos_df.copy()
    basis_phy_pos_df['exposure'] = 'BASIS (NET PHYS)'

    outright_phy_pos_df = phy_grouped_pos_df[
        phy_grouped_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES'])
    ].copy()
    outright_phy_pos_df['exposure'] = 'OUTRIGHT'

    # India adjustment: If India DIFF PHYS =/= 0, split into respective legs and classify them as OUTRIGHT, remove the original DIFF PHYS position
    # TODO: Add India logic when available
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
                'instrument_name': 'EX GIN S6',
                'exposure': 'OUTRIGHT',
                'position_type': 'DIFF PHYS',
                'delta': row['delta'],
                'to_USD_conversion': instrument_ref_dict['CCL']['to_USD_conversion']
            })

            # DERIVATIVES LEG
            derivs_leg = base_leg.copy()
            derivs_leg.update({
                'instrument_name': 'CT',
                'exposure': 'OUTRIGHT',
                'position_type': 'DERIVS',
                'delta': -row['delta'],  # negated
                'to_USD_conversion': instrument_ref_dict['CT']['to_USD_conversion']
            })
            both_india_legs_df = pd.DataFrame([phys_leg, derivs_leg])
            india_pos_df = pd.concat([india_pos_df, both_india_legs_df], ignore_index=True)
        outright_phy_pos_df = pd.concat([outright_phy_pos_df, india_pos_df], ignore_index=True)
    else:
        print('India has no basis position')
    basis_phy_pos_df = basis_phy_pos_df[basis_phy_pos_df['region'] != 'INDIA']

    # Assign Bloomberg tickers and generic curves
    outright_phy_pos_df = physical_loader.assign_bbg_tickers(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df = physical_loader.assign_generic_curves(outright_phy_pos_df, instrument_dict)
    basis_phy_pos_df = physical_loader.assign_bbg_tickers(basis_phy_pos_df, instrument_dict) # for MC VaR
    basis_phy_pos_df = physical_loader.assign_generic_curves(basis_phy_pos_df, instrument_dict) # for MC VaR
    basis_phy_pos_df = physical_loader.assign_basis_series(basis_phy_pos_df, fy24_unit_to_cotlook_basis_origin_dict)

    # Create unit-region mapping for derivatives
    cotton_unit_region_mapping = dict(zip(conso_pos_df['unit'], conso_pos_df['region']))
    print('Step 2F completed')

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
    print('Step 2G completed')

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

    print(f"[INFO] Combined position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df


def prepare_returns_and_positions_data(product: str, cob_date: str, window: int) \
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

    if product == 'cotton':
        instrument_dict['PHYS'] = {}
        instrument_dict['PHYS']['EX GIN S6'] = generate_ex_gin_s6_returns_df(cob_date, window)
        instrument_dict['PHYS']['COTLOOK'] = generate_cotlook_returns_df(cob_date, window)
        print("[INFO] Step 1A: Market data generation completed.")

        # Step 3: Generate basis returns (cotton-specific)
        basis_abs_ret_df = fy24_cotton_basis_workflow(
            cob_date=cob_date,
            window=window,
            write_to_excel=True,
            apply_smoothing=True
        )
        basis_abs_ret_df.columns = basis_abs_ret_df.columns.str.replace(' final AR series', '', regex=False)
        basis_abs_ret_df.columns = basis_abs_ret_df.columns.str.replace(' final AR (sm) series', '', regex=False)
        instrument_dict['BASIS'] = {}
        instrument_dict['BASIS']['abs_returns_$_df'] = basis_abs_ret_df
        print("[INFO] Step 1C: Basis returns generation completed.")

    # Step 2: Generate position data
    if product == 'cotton':
        combined_pos_df = generate_cotton_combined_position(cob_date, instrument_dict)
        print("[INFO] Step 1B: Position data generation completed.")

    else:
        raise NotImplementedError(f"Product '{product}' not yet supported in data preparation workflow.")

    print("[SUCCESS] Data preparation workflow completed.")
    return instrument_dict, combined_pos_df

def prepare_pos_data_for_var(combined_pos_df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == 'linear':
        uat_engine = get_engine('uat')
        pos_loader = PositionLoader(source=uat_engine)
        combined_pos_df = pos_loader.assign_linear_var_map(combined_pos_df)
        combined_pos_df = combined_pos_df[[
            'cob_date', 'product', 'unit', 'region', 'books', 'position_type',
            'total_active_lots', 'settle_delta_1', 'exposure', 'trader_id', 'trader_name', 'counterparty_id',
            'counterparty_parent', 'instrument_name', 'bbg_ticker', 'underlying_bbg_ticker', 'generic_curve',
            'basis_series', 'delta', 'to_USD_conversion', 'lots_to_MT_conversion', 'position_index', 'linear_var_map'
        ]]
    elif method == 'non-linear (monte carlo)':
        uat_engine = get_engine('uat')
        pos_loader = PositionLoader(source=uat_engine)
        combined_pos_df = pos_loader.assign_monte_carlo_var_risk_factor(combined_pos_df)
        combined_pos_df = pos_loader.duplicate_basis_and_assign_ct1(combined_pos_df)
        combined_pos_df = combined_pos_df[[
            'cob_date', 'product', 'unit', 'region', 'books', 'position_type',
            'total_active_lots', 'settle_delta_1', 'exposure', 'trader_id', 'trader_name', 'counterparty_id',
            'counterparty_parent', 'instrument_name', 'bbg_ticker', 'underlying_bbg_ticker', 'generic_curve',
            'basis_series', 'delta', 'to_USD_conversion', 'lots_to_MT_conversion', 'position_index',
            'monte_carlo_var_risk_factor'
        ]]
    combined_pos_df['position_index'] = (
            combined_pos_df['product'].str[:3] + '_L_' +
            combined_pos_df['cob_date'].astype(str) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )
    return combined_pos_df