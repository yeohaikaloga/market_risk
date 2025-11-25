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
import pickle

# Core loaders and generators
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from contract_ref_loader.physical_contract_ref_loader import PhysicalContractRefLoader
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from price_series_loader.physical_price_loader import PhysicalPriceLoader
from price_series_loader.vol_series_loader import VolLoader
from price_series_generator.generic_curve_generator import GenericCurveGenerator

from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from position_loader.physical_position_loader import PhysicalPositionLoader
from position_loader.position_loader import PositionLoader
from utils.contract_utils import (load_instrument_ref_dict, month_codes, extract_instrument_from_product_code,
                                  get_USD_MT_conversion_from_product_code, obtain_product_code_from_instrument_name)
from financial_calculations.returns import relative_returns
from position_loader.physical_position_loader import fy24_unit_to_cotlook_basis_origin_dict
from workflow.shared.forex_workflow import load_forex

# Specialist processors (moved from utils/)
from workflow.shared.position_processing import calculate_phys_derivs_aggs, calculate_basis_adj_and_basis_pos

# External workflow (only if reused — otherwise inline or move logic)
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow
from contract_ref_loader.physical_contract_ref_loader import crop_dict

def load_raw_cotton_deriv_position(cob_date: str) -> pd.DataFrame:
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'cotton'

    # Step 2G: Load and process derivatives positions
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_position(
        date=cob_date,
        trader_id=None,
        counterparty_id=None,
        product=product,
        book=None
    )
    deriv_pos_df = deriv_pos_df[~deriv_pos_df['security_id'].astype(str).str.startswith('CR')]
    return deriv_pos_df

def load_raw_rubber_deriv_position(cob_date: str) -> pd.DataFrame:
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'rubber'
    # Step 2G: Load and process derivatives positions
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_position(
        date=cob_date,
        trader_id=None,
        counterparty_id=None,
        product=product,
        book=None
    )
    deriv_pos_df = deriv_pos_df[~deriv_pos_df['security_id'].astype(str).str.startswith('CR')]
    return deriv_pos_df

def load_raw_rms_deriv_position(cob_date: str) -> pd.DataFrame:
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'rms'

    # Step 2G: Load and process derivatives positions
    derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    deriv_pos_df = derivatives_loader.load_position(
        date=cob_date,
        trader_id=None,
        counterparty_id=None,
        product=product,
        book=None
    )
    deriv_pos_df = deriv_pos_df[~deriv_pos_df['security_id'].astype(str).str.startswith('CR')]
    deriv_pos_df['region'] = deriv_pos_df['portfolio']
    return deriv_pos_df

def generate_product_code_list_for_generic_curve(product: str, df: pd.DataFrame) -> list:
    if product == 'cotton':
        product_code_list = (
            df['product_code']
            .unique()
            .tolist()
        )

    elif product == 'rubber' or product == 'rms':
        product_code_list = (
            df['security_id']
            .astype(str)
            .str.extract(r'(\S+\s+\S+)')[0]  # first two space-separated tokens
            .unique()
            .tolist()
        )
    else:
        raise ValueError(f"Unsupported product: {product}")

    return product_code_list

def generate_instrument_vol_change_dict(instrument_list: list, cob_date: str, window: int) -> Dict[str, Any]:
    rms_engine = get_engine('rms')
    days_list = get_prev_biz_days_list(cob_date, window + 1)

    instrument_vol_dict = {}

    for instrument_name in instrument_list:
        vol_change_loader = VolLoader(
            instrument_name=instrument_name,
            source=rms_engine
        )
        vol_change_df = vol_change_loader.load_vol_change_for_generic_curve(
            start_date=days_list[0],
            end_date=cob_date,
            max_generic_curve=9,
            reindex_dates=None,
            instrument_name=instrument_name
        )

        instrument_vol_dict[instrument_name] = vol_change_df

    return instrument_vol_dict


def generate_instrument_generic_curves_dict(instrument_list: list, cob_date: str, window: int,
                                            usd_conversion_mode: str, fx_df: pd.DataFrame) -> Dict[str, Any]:
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

    instrument_dict = {}
    instrument_ref_dict = load_instrument_ref_dict('uat')
    returns_df = pd.DataFrame()
    prices_df = pd.DataFrame()

    for instrument_name in instrument_list:
        # Step 1A: Load contract metadata
        derivatives_contract = DerivativesContractRefLoader(
            instrument_name=instrument_name,
            source=prod_engine,
        )
        if instrument_name == 'CT':
            relevant_months = ['H', 'K', 'N', 'Z']
        else:
            relevant_months = None
        futures_contracts = derivatives_contract.load_contracts(
            mode='futures',
            relevant_months=relevant_months,
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
            label_prefix=instrument_name,
            usd_conversion_mode=usd_conversion_mode,
            fx_df=fx_df,
            cob_date=cob_date
        )

        # Clean and prepare returns
        generic_curves_df = (
            generic_curves_df
            .replace({pd.NA: np.nan})
            .astype(float)
        )

        relative_returns_df = relative_returns(generic_curves_df)
        relative_returns_df = relative_returns_df.fillna(0)
        relative_returns_dollarised_df = relative_returns_df * generic_curves_df.loc[cob_date]


        product_code = obtain_product_code_from_instrument_name(instrument_name, instrument_ref_dict)

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
        returns_df = pd.concat([returns_df, relative_returns_df], axis=1)
        prices_df = pd.concat([prices_df, generic_curves_df], axis=1)

        print(f"[INFO] Generated curves for {instrument_name}. Sample:\n{generic_curves_df.head()}")

    return instrument_dict, returns_df, prices_df

def generate_ex_gin_s6_returns_df(cob_date: str, window: int, fx_spot_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    uat_engine = get_engine('uat')
    days_list = get_prev_biz_days_list(cob_date, window + 1)
    ex_gin_s6 = PhysicalPriceLoader(instrument_name='EX GIN S6', source=uat_engine)
    ex_gin_s6_df = ex_gin_s6.load_ex_gins6_prices_from_staging(start_date=days_list[0], end_date=cob_date, data_source='EX GIN S6')
    ex_gin_s6_df['date'] = pd.to_datetime(ex_gin_s6_df['date'])
    ex_gin_s6_df = ex_gin_s6_df.set_index('date')[['price']].sort_index()
    ex_gin_s6_df = ex_gin_s6_df.rename(columns={'price': 'EX GIN S6'})
    ex_gin_s6_relative_returns_df = relative_returns(ex_gin_s6_df)
    print(ex_gin_s6_df.tail())
    print(ex_gin_s6_df.loc[cob_date])
    # Rs/Candy to Rs/MT conversion factor : 1000 / 355.56 = 2.8124648 (OPERA: 2.810304); same with CCL contract
    # ex_gin_s6_relative_returns_df['relative_returns_INR/Candy'] = (ex_gin_s6_relative_returns_df['EX GIN S6'] * ex_gin_s6_df.loc[cob_date, 'price'])
    ex_gin_s6_relative_returns_df['relative_returns_INR/Candy'] = (
                ex_gin_s6_relative_returns_df['EX GIN S6'])
    usdinr_spot_cob = fx_spot_df.loc[cob_date, 'USDINR']
    ex_gin_s6_relative_returns_df['relative_returns_USD/Candy'] = ex_gin_s6_relative_returns_df['relative_returns_INR/Candy'] / usdinr_spot_cob
    return ex_gin_s6_df, ex_gin_s6_relative_returns_df

def generate_cotlook_relative_returns_dict(cob_date: str, window: int) -> dict:
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

def prepare_returns_and_positions_data(product, product_code_list: list, cob_date: str, window: int) \
        -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
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

    #Step 1A: Generate market data
    instrument_ref_dict = load_instrument_ref_dict('uat')
    fx_spot_df = load_forex(cob_date=cob_date, window=window)
    instrument_list = []
    for product_code in product_code_list:
        instrument = instrument_ref_dict[product_code]['bbg_product_code']
        instrument_list.append(instrument)

    if product != 'rms':
        if product == 'cotton':
            usd_conversion_mode = 'post'
            instrument_dict, relative_returns_df, prices_df = generate_instrument_generic_curves_dict(instrument_list,
                                                                                                      cob_date, window,
                                                                                                      usd_conversion_mode,
                                                                                                      fx_spot_df)
        elif product == 'rubber':
            usd_conversion_mode = 'pre'
            instrument_dict, relative_returns_df, prices_df = generate_instrument_generic_curves_dict(instrument_list,
                                                                                                      cob_date, window,
                                                                                                      usd_conversion_mode,
                                                                                                      fx_spot_df)
    elif product == 'rms':
        usd_conversion_mode = 'pre'
        instrument_dict, relative_returns_df, prices_df = generate_instrument_generic_curves_dict(instrument_list,
                                                                                                  cob_date, window,
                                                                                                  usd_conversion_mode,
                                                                                                  fx_spot_df)
        instrument_vol_dict = generate_instrument_vol_change_dict(instrument_list, cob_date, window)
        for instrument in instrument_dict.keys():
            instrument_dict[instrument]['vol_change_df'] = instrument_vol_dict[instrument]
        pass
    print('generic curves done')

    if product == 'cotton':
        instrument_dict['PHYS'] = {}
        instrument_dict['PHYS']['EX GIN S6'] = {
            'price_series': None,
            'relative_returns_df': None
        }

        ex_gin_s6_price_series, ex_gin_s6_relative_returns_df = generate_ex_gin_s6_returns_df(cob_date, window, fx_spot_df)
        instrument_dict['PHYS']['EX GIN S6']['price_series'] = ex_gin_s6_price_series
        instrument_dict['PHYS']['EX GIN S6']['relative_returns_df'] = ex_gin_s6_relative_returns_df
        phys_relative_returns_df = ex_gin_s6_relative_returns_df[
            'relative_returns_USD/Candy'
        ].to_frame(name='EX GIN S6')
        relative_returns_df = pd.concat([relative_returns_df, phys_relative_returns_df], axis=1)
        prices_df = pd.concat([prices_df, ex_gin_s6_price_series], axis=1)
        print('ex gin s6 done')

        cotlook_relative_returns_dict = generate_cotlook_relative_returns_dict(cob_date, window)
        instrument_dict['PHYS']['COTLOOK'] = cotlook_relative_returns_dict
        for cotlook in cotlook_relative_returns_dict:
            relative_returns_df = pd.concat([relative_returns_df, cotlook_relative_returns_dict[cotlook]], axis=1)
        print('cotlook done')
        print("[INFO] Step 1A: Market data generation completed.")

        # Step 1B: Generate basis returns (cotton-specific)
        basis_df = fy24_cotton_basis_workflow(
            cob_date=cob_date,
            window=window,
            write_to_excel=True,
            apply_smoothing=True
        )
        basis_df = basis_df.reindex(relative_returns_df.index)
        absolute_returns_df = basis_df
        absolute_returns_df.columns = absolute_returns_df.columns.str.replace(' final AR series', '', regex=False)
        absolute_returns_df.columns = absolute_returns_df.columns.str.replace(' final AR (sm) series', '', regex=False)
        instrument_dict['BASIS'] = {}
        instrument_dict['BASIS']['abs_returns_$_df'] = absolute_returns_df
        print("[INFO] Step 1B: Basis returns generation completed.")

        #TODO for RMS, use generic curves from Risk DB (instead of OPERA view table) when forex is ready.
    if product != 'cotton':
        absolute_returns_df = pd.DataFrame()
    returns_df = pd.concat([relative_returns_df, absolute_returns_df], axis=1)
    instrument_dict['FOREX'] = fx_spot_df

    f = open('instrument_dict.pkl', 'wb')
    pickle.dump(instrument_dict, f)
    f.close()

    g = open('returns_df.pkl', 'wb')
    pickle.dump(returns_df, g)
    g.close()

    h = open('prices_df.pkl', 'wb')
    pickle.dump(prices_df, h)
    h.close()

    f = open('instrument_dict.pkl', 'rb')
    instrument_dict = pickle.load(f)
    f.close()

    g = open('returns_df.pkl', 'rb')
    returns_df = pickle.load(g)
    g.close()

    h = open('prices_df.pkl', 'rb')
    prices_df = pickle.load(h)
    h.close()

    # Step 2: Generate position data
    if product == 'cotton':
        combined_pos_df = generate_cotton_combined_position(cob_date, instrument_dict, prices_df)
        print("[INFO] Step 2 [cotton]: Position data generation completed.")
    elif product == 'rubber':
        combined_pos_df = generate_rubber_combined_position(cob_date, instrument_dict, prices_df)
        print("[INFO] Step 2 [rubber]: Position data generation completed.")
    elif product == 'rms':
        combined_pos_df = generate_rms_combined_position(cob_date, instrument_dict, prices_df)
        print("[INFO] Step 2 [rms]: Position data generation completed.")
    else:
        raise NotImplementedError(f"Product '{product}' not yet supported in data preparation workflow.")

    return instrument_dict, combined_pos_df, returns_df, prices_df

def generate_cotton_combined_position(cob_date: str, instrument_dict: Dict[str, Any], prices_df: pd.DataFrame) -> pd.DataFrame:
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
    instrument_ref_dict = load_instrument_ref_dict('uat')

    # Step 2A: Load physical positions from staging
    physical_loader = PhysicalPositionLoader(date=cob_date, source=uat_engine)
    conso_pos_df = physical_loader.load_cotton_phy_position_from_staging(cob_date=cob_date)
    conso_pos_df.columns = [col.lower() for col in conso_pos_df.columns]
    print('[DATA PREP] Step 2A [cotton] completed')

    # Step 2B: Classify position types
    pos_type_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='UNKNOWN')
    print('[DATA PREP] Step 2B [cotton] completed')

    # Step 2C: Temporary exposure for basis logic (to be replaced)
    exposure_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='UNKNOWN')
    print('[DATA PREP] Step 2C [cotton] completed')

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
    print('[DATA PREP] Step 2D [cotton] completed')

    # Step 2E: Calculate aggregates by region/product_code
    phys_derivs_aggs = conso_pos_df.groupby(['region', 'product_code']).apply(calculate_phys_derivs_aggs).reset_index()
    phys_derivs_aggs['outright_pos'] = phys_derivs_aggs['net_fixed_phys'] + phys_derivs_aggs['derivs']
    phys_derivs_aggs[['basis_adj', 'basis_pos']] = phys_derivs_aggs.apply(
        lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)),
        axis=1
    )
    print('[DATA PREP] Step 2E [cotton] completed')

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
                'to_USD_conversion': instrument_ref_dict['CM CCL']['to_USD_conversion']
            })

            # DERIVATIVES LEG
            derivs_leg = base_leg.copy()
            derivs_leg.update({
                'product_code': 'CM CT',
                'exposure': 'OUTRIGHT',
                'position_type': 'DERIVS',
                'delta': -row['delta'],  # negated
                'to_USD_conversion': instrument_ref_dict['CM CCL']['to_USD_conversion']
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

    basis_phy_pos_df['instrument_name'] = (
        basis_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )
    basis_phy_pos_df['lots_to_MT_conversion'] = 1
    basis_phy_pos_df = physical_loader.assign_bbg_tickers(basis_phy_pos_df, instrument_dict) # for MC VaR
    basis_phy_pos_df = physical_loader.assign_generic_curves(basis_phy_pos_df, instrument_dict) # for MC VaR
    basis_phy_pos_df['cob_date_price'] = 1
    basis_phy_pos_df = physical_loader.assign_basis_series(basis_phy_pos_df, fy24_unit_to_cotlook_basis_origin_dict)
    # Create unit-region mapping for derivatives
    cotton_unit_region_mapping = dict(zip(conso_pos_df['unit'], conso_pos_df['region']))
    print('[DATA PREP] Step 2F [cotton] completed')

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
    print('[DATA PREP] Step 2G [cotton] completed')

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

    print(f"[INFO] Combined [cotton] position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df

def generate_rubber_combined_position(cob_date: str, instrument_dict: Dict[str, Any], prices_df: pd.DataFrame) -> pd.DataFrame:
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
    conso_pos_df['quantity'] = pd.to_numeric(conso_pos_df['quantity'], errors='coerce')
    conso_pos_df['transaction type'] = 'PHYS'
    conso_pos_df['purchases type'] = conso_pos_df['purchases type'].astype(str).str.upper()
    conso_pos_df['purchases type'] = conso_pos_df['purchases type'].str.replace('PURCHASES', 'PURCHASE')
    conso_pos_df['typ'] = conso_pos_df['trade type'] + ' ' + conso_pos_df['purchases type']

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
    conso_pos_df['region'] = conso_pos_df['unit'].map(rubber_unit_region_mapping)
    def adjusted_delta(product_type, delta_quantity):
        if product_type == 'PRIMARY PR':
            return delta_quantity * 0.68
        else:
            return delta_quantity
    conso_pos_df['quantity'] = conso_pos_df.apply(lambda row: adjusted_delta(row['product_type'], row['quantity']), axis=1)

    def rubber_portfolio_product_code_mapping(portfolio):
        if portfolio == 'CHINA TRAD' or portfolio == 'CHINATRAD2' or portfolio == 'CHINA':
            return 'IM RT'
        else:
            return 'CM OR'
    conso_pos_df['product_code'] = conso_pos_df.apply(
        lambda row: rubber_portfolio_product_code_mapping(row['portfolio']), axis=1)
    print('[DATA PREP] Step 2A [rubber] completed')

    # Step 2B: Classify position types
    pos_type_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'FIXED STOCK']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='UNKNOWN')
    print('[DATA PREP] Step 2B [rubber] completed')

    # Step 2C: Temporary exposure for basis logic (to be replaced)
    exposure_conditions = [
        conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'FIXED STOCK']),
        conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
        conso_pos_df['typ'].isin(['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])
    ]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='UNKNOWN')
    print('[DATA PREP] Step 2C [rubber] completed')

    # Step 2D: Calculate aggregates by unit/contract
    phys_derivs_aggs = conso_pos_df.groupby(['region', 'terminal_month']).apply(calculate_phys_derivs_aggs).reset_index()
    phys_derivs_aggs['outright_pos'] = phys_derivs_aggs['net_fixed_phys'] + phys_derivs_aggs['derivs']
    phys_derivs_aggs[['basis_adj', 'basis_pos']] = phys_derivs_aggs.apply(
        lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)),
        axis=1
    )
    print('[DATA PREP] Step 2D [rubber] completed')

    # Step 2E: Prepare physical positions for combination
    phy_pos_df = conso_pos_df[conso_pos_df['position_type'] != 'DERIVS'].copy()
    phy_pos_df = phy_pos_df.rename(columns={'quantity': 'delta'})

    phy_grouped_pos_df = phy_pos_df.groupby([
        'unit', 'region', 'typ', 'position_type', 'terminal_month', 'product_code', 'counterparty_parent',
        'trader_name', 'product_type', 'source'
    ], as_index=False)['delta'].sum()

    phy_grouped_pos_df = phy_grouped_pos_df[phy_grouped_pos_df['delta'] != 0]
    phy_grouped_pos_df['subportfolio'] = phy_grouped_pos_df['typ']
    phy_grouped_pos_df['strike'] = np.nan

    # Split into basis and outright exposures: All physicals positions are basis; all fixed physical positions are outright
    basis_phy_pos_df = phy_grouped_pos_df.copy()
    basis_phy_pos_df['exposure'] = 'BASIS (NET PHYS)'

    outright_phy_pos_df = phy_grouped_pos_df[
        phy_grouped_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES', 'FIXED STOCK'])
    ].copy()
    outright_phy_pos_df['exposure'] = 'OUTRIGHT'

    # Assign Bloomberg tickers and generic curves
    #outright_phy_pos_df = physical_loader.assign_bbg_tickers(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df['terminal_month'] = pd.to_datetime(outright_phy_pos_df['terminal_month'], errors='coerce')
    outright_phy_pos_df['bbg_ticker'] = outright_phy_pos_df.apply(
        lambda row: (
                row['product_code']
                + [k for k, v in month_codes.items() if v == row['terminal_month'].month][0]
                + str(row['terminal_month'].year)[-1]
                + " Comdty"
        ),
        axis=1
    )
    outright_phy_pos_df['underlying_bbg_ticker'] = outright_phy_pos_df['bbg_ticker']
    outright_phy_pos_df['instrument_name'] = (
        outright_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )
    outright_phy_pos_df = physical_loader.assign_generic_curves(outright_phy_pos_df, instrument_dict)
    outright_phy_pos_df = physical_loader.assign_cob_date_price(outright_phy_pos_df, prices_df, cob_date)
    outright_phy_pos_df['to_USD_conversion'] = outright_phy_pos_df['product_code'].map(
        lambda x: instrument_ref_dict.get(x, {}).get('to_USD_conversion', np.nan)
    )
    basis_phy_pos_df['instrument_name'] = (
        basis_phy_pos_df['product_code']
        .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
    )
    #basis_phy_pos_df = physical_loader.assign_bbg_tickers(basis_phy_pos_df, instrument_ref_dict)  # for MC VaR
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

    print('[DATA PREP] Step 2E [rubber] completed')

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
        deriv_pos_df['product_code'] = deriv_pos_df['security_id'].str.split().str[:2].str.join(' ')
        deriv_pos_df = derivatives_loader.assign_bbg_tickers(deriv_pos_df)
        deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
        deriv_pos_df['instrument_name'] = (
            deriv_pos_df['product_code']
            .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
        )
        deriv_pos_df = derivatives_loader.assign_generic_curves(deriv_pos_df, instrument_dict)
        #deriv_pos_df['instrument_name'] = deriv_pos_df['product_code'].apply(extract_instrument_from_product_code)
        deriv_pos_df = derivatives_loader.assign_cob_date_price(deriv_pos_df, prices_df, cob_date)
        deriv_pos_df['position_type'] = 'DERIVS'
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
    print('[DATA PREP] Step 2F [rubber] completed')

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

    combined_pos_df['position_index'] = (
            product[:3] + '_L_' + str(cob_date) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )

    print('[DATA PREP] Step 2G [rubber] completed')
    print(f"[INFO] Combined [rubber] position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df

def generate_rms_combined_position(cob_date: str, instrument_dict: Dict[str, Any]) -> pd.DataFrame:
    uat_engine = get_engine('uat')  # TODO: Switch to 'prod' in production
    product = 'rms'
    instrument_ref_dict = load_instrument_ref_dict('uat')

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
    print('[DATA PREP] Step 2F [rms] completed')

    # Step 2G: Combine all positions
    combined_pos_df = pd.concat(
        [deriv_pos_df],
        axis=0,
        ignore_index=True
    )
    combined_pos_df = combined_pos_df.reset_index(drop=True)
    combined_pos_df['product'] = product
    combined_pos_df['cob_date'] = cob_date
    #combined_pos_df = combined_pos_df[
    #    combined_pos_df['region'].apply(lambda x: isinstance(x, str))
    #]

    combined_pos_df['position_index'] = (
            product[:3] + '_L_' + str(cob_date) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
    )

    print('[DATA PREP] Step 2G [rms] completed')
    print(f"[INFO] Combined [rms] position DataFrame generated. Shape: {combined_pos_df.shape}")
    print(combined_pos_df.head())

    return combined_pos_df

def prepare_pos_data_for_var(combined_pos_df: pd.DataFrame, method: str, trader: bool, counterparty: bool) -> pd.DataFrame:
    # Step 2H: Prepare minimal position data needed for VaR calculator
    # Define minimal base columns upfront

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
    print('[DATA PREP] Step 2H completed')
    return combined_pos_df