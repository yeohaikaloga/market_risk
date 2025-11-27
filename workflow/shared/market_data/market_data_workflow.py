import pandas as pd
import numpy as np
from typing import Dict, Any
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
from utils.contract_utils import product_specifications
from financial_calculations.returns import relative_returns
from workflow.shared.forex_workflow import load_forex
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow
from contract_ref_loader.physical_contract_ref_loader import crop_dict


def build_instrument_generic_curves(instrument_list: list, cob_date: str, window: int, usd_conversion_mode: str,
                                    forex_mode: str, fx_spot_df: pd.DataFrame) -> \
        (pd.DataFrame, pd.DataFrame, Dict[str, Any]):
    """
    STEP 1: Generate generic futures curves and returns for all relevant instruments.

    For each instrument (e.g., 'CT', 'VV'), generates:
    - generic_curves_df: rolled generic contract prices (CT1, CT2, ...)
    - relative_returns_df: percentage returns
    - relative_returns_$_df: dollar returns (scaled by COB price)
    - contract_to_curve_map: maps actual contract (e.g., CTZ4) to generic curve (e.g., CT1)

    Args:
        instrument_list
        cob_date: Close-of-business date (YYYY-MM-DD)
        window: Lookback window for returns calculation
        usd_conversion_mode
        fx_spot_df

    Returns:
        Dict[instrument_name, Dict[str, Any]] — structured market data per instrument
    """
    prod_engine = get_engine('prod')
    days_list = get_prev_biz_days_list(cob_date, window + 1)

    prices_df = pd.DataFrame()
    # raw_prices_df = pd.DataFrame()
    returns_df = pd.DataFrame()
    instrument_dict = {}

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
            forex_mode=forex_mode,
            fx_spot_df=fx_spot_df,
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

        # Create contract-to-curve mapping
        contract_to_curve_map = {}
        for k, v in active_contracts_df.loc[cob_date].to_dict().items():
            if v is not None:
                contract = v.replace(' COMB', '')
                if contract not in contract_to_curve_map:
                    contract_to_curve_map[contract] = k

        # Store data
        prices_df = pd.concat([prices_df, generic_curves_df], axis=1)
        returns_df = pd.concat([returns_df, relative_returns_df], axis=1)
        instrument_dict[instrument_name] = {
            'generic_curves_df': generic_curves_df,
            'relative_returns_df': relative_returns_df,
            'relative_returns_$_df': relative_returns_dollarised_df,
            'contract_to_curve_map': contract_to_curve_map
        }

        print(f"[INFO] Generated curves for {instrument_name}. Sample:\n{generic_curves_df.head()}")

    return prices_df, returns_df, instrument_dict


def build_ex_gin_s6_returns(cob_date: str, window: int, fx_spot_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP: Generate EX GIN S6 historical prices and relative returns.

    Returns:
        - prices_df: EX GIN S6 prices indexed by date
        - returns_df: EX GIN S6 relative returns (INR/Candy and USD/Candy)
    """
    uat_engine = get_engine('uat')
    days_list = get_prev_biz_days_list(cob_date, window + 1)
    ex_gin_s6 = PhysicalPriceLoader(instrument_name='EX GIN S6', source=uat_engine)
    ex_gin_s6_df = ex_gin_s6.load_ex_gins6_prices_from_staging(start_date=days_list[0], end_date=cob_date)
    ex_gin_s6_df['date'] = pd.to_datetime(ex_gin_s6_df['date'])
    ex_gin_s6_df = ex_gin_s6_df.set_index('date')[['price']].sort_index()
    ex_gin_s6_df = ex_gin_s6_df.rename(columns={'price': 'EX GIN S6'})
    ex_gin_s6_relative_returns_df = relative_returns(ex_gin_s6_df)
    print(ex_gin_s6_df.tail())
    print(ex_gin_s6_df.loc[cob_date])
    # Rs/Candy to Rs/MT conversion factor : 1000 / 355.56 = 2.8124648 (OPERA: 2.810304); same with CCL contract
    ex_gin_s6_relative_returns_df['relative_returns_INR/Candy'] = (
        ex_gin_s6_relative_returns_df['EX GIN S6'])
    usdinr_spot_cob = fx_spot_df.loc[cob_date, 'USDINR']
    ex_gin_s6_relative_returns_df['relative_returns_USD/Candy'] = ex_gin_s6_relative_returns_df[
                                                                      'relative_returns_INR/Candy'] / usdinr_spot_cob
    return ex_gin_s6_df, ex_gin_s6_relative_returns_df


def build_cotlook_relative_returns(cob_date: str, window: int) -> dict:
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
                    stitched_crop_series.loc[i, physical_contract.instrument_name] = (
                        crop_price_df_by_year.loc)[i, crop_year]
            stitched_crop_series = stitched_crop_series.reindex(biz_days, method='ffill')
            stitched_crop_series = stitched_crop_series.interpolate(method='linear', limit_direction='both', axis=0)
            # interpolate (newly added)
            stitched_crop_series_relative_returns_df = relative_returns(stitched_crop_series)
            stitched_crop_series_relative_returns_dollarised_df = (stitched_crop_series_relative_returns_df *
                                                                   stitched_crop_series.loc[cob_date,
                                                                   physical_contract.instrument_name])
        cotlook_dict[physical_contract.instrument_name] = stitched_crop_series_relative_returns_dollarised_df
    return cotlook_dict

def build_wood_returns(cob_date: str, window: int) -> dict:
    prod_engine = get_engine('prod')
    biz_days = pd.DatetimeIndex(get_prev_biz_days_list(date=cob_date, no_of_days=window))
    start_date = biz_days[0]
    wood_series = ['Wood_France_DKD_Sapele', 'Wood_Netherlands_FKD_Sapelli']
    physical_contracts_and_prices = []
    wood_dict = {}

    for wood in wood_series:
        wood_params = crop_dict.get(wood, {})
        wood_params['data_source'] = 'BU'
        physical_contract = PhysicalContractRefLoader(instrument_name=wood, source=prod_engine, params=wood_params)
        physical_contract.load_ref_data()
        wood_price = PhysicalPriceLoader(instrument_name=wood, source=prod_engine, params=wood_params)
        wood_price_df = wood_price.load_prices(start_date=start_date, end_date=cob_date, data_source='BU')

    return wood_dict

def build_instrument_vol_change_dict(instrument_list: list, cob_date: str, window: int) -> Dict[str, Any]:
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


def build_product_prices_returns_dfs(cob_date: str, product: str, window: int):
    instrument_list = product_specifications[product]['instrument_list']
    usd_conversion_mode = product_specifications[product]['usd_conversion_mode']
    forex_mode = product_specifications[product]['forex_mode']
    fx_spot_df = load_forex(cob_date=cob_date, window=window)

    prices_df, relative_returns_df, instrument_dict = build_instrument_generic_curves(instrument_list, cob_date, window,
                                                                                      usd_conversion_mode, forex_mode,
                                                                                      fx_spot_df)

    if product == 'cotton':

        instrument_dict['PHYS'] = {}
        instrument_dict['PHYS']['EX GIN S6'] = {
            'price_series': None,
            'relative_returns_df': None
        }
        ex_gin_s6_df, ex_gin_s6_relative_returns_df = build_ex_gin_s6_returns(cob_date, window, fx_spot_df)
        instrument_dict['PHYS']['EX GIN S6']['price_series'] = ex_gin_s6_df
        instrument_dict['PHYS']['EX GIN S6']['relative_returns_df'] = ex_gin_s6_relative_returns_df
        phys_relative_returns_df = ex_gin_s6_relative_returns_df[
            'relative_returns_USD/Candy'
        ].to_frame(name='EX GIN S6')
        prices_df = pd.concat([prices_df, ex_gin_s6_df], axis=1)
        relative_returns_df = pd.concat([relative_returns_df, phys_relative_returns_df], axis=1)

        cotlook_relative_returns_dict = build_cotlook_relative_returns(cob_date, window)
        instrument_dict['PHYS']['COTLOOK'] = cotlook_relative_returns_dict
        for cotlook in cotlook_relative_returns_dict:
            relative_returns_df = pd.concat([relative_returns_df, cotlook_relative_returns_dict[cotlook]], axis=1)

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

    else:
        absolute_returns_df = pd.DataFrame()

    if product == 'rms':
        instrument_vol_dict = build_instrument_vol_change_dict(instrument_list, cob_date, window)
        for instrument in instrument_dict.keys():
            instrument_dict[instrument]['vol_change_df'] = instrument_vol_dict[instrument]

    returns_df = pd.concat([relative_returns_df, absolute_returns_df], axis=1)
    instrument_dict['FOREX'] = fx_spot_df

    return prices_df, returns_df, instrument_dict
