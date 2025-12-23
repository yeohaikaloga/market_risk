import pandas as pd
import numpy as np
from typing import Dict, Any

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
from contract_ref_loader.physical_contract_ref_loader import crop_dict, wood_series_dict


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
        usd_conversion_mode:
        forex_mode:
        fx_spot_df:

    Returns:
        Dict[instrument_name, Dict[str, Any]] — structured market data per instrument
    """
    prod_engine = get_engine('prod')
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    returns_days_list = get_prev_biz_days_list(cob_date, window)
    prices_df = pd.DataFrame(index=prices_days_list)
    returns_df = pd.DataFrame(index=returns_days_list)
    # raw_prices_df = pd.DataFrame()
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
            start_date=prices_days_list[0],
            end_date=cob_date,
            contracts=futures_contracts,
            reindex_dates=prices_days_list,
            instrument_name=instrument_name
        )

        # Step 1C: Generate generic curves
        curve_generator = GenericCurveGenerator(
            df=price_df,
            futures_contract=derivatives_contract
        )
        max_position = 11 if instrument_name == 'CT' else 6  # CT goes up to CT13, others to xx6

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


def build_ex_gin_s6_returns(cob_date: str, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP: Generate EX GIN S6 historical prices and relative returns.

    Returns:
        - prices_df: EX GIN S6 prices indexed by date
        - returns_df: EX GIN S6 relative returns (INR/Candy and USD/Candy)
    """
    # TODO Rewrite function after brought into prices master table
    uat_engine = get_engine('uat')

    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    ex_gin_s6 = PhysicalPriceLoader(instrument_name='EX GIN S6', source=uat_engine)
    ex_gin_s6_df = ex_gin_s6.load_ex_gin_s6_price_from_staging(start_date=prices_days_list[0], end_date=cob_date)
    ex_gin_s6_df['date'] = pd.to_datetime(ex_gin_s6_df['date'])
    ex_gin_s6_df = ex_gin_s6_df.set_index('date')[['price']].sort_index()
    ex_gin_s6_df = ex_gin_s6_df.rename(columns={'price': 'EX GIN S6'})
    ex_gin_s6_df = ex_gin_s6_df.reindex(prices_days_list)
    ex_gin_s6_df = ex_gin_s6_df.interpolate(method='linear', axis=0)
    ex_gin_s6_df = ex_gin_s6_df.bfill().ffill()
    ex_gin_s6_relative_returns_df = relative_returns(ex_gin_s6_df)
    print(ex_gin_s6_df.tail())
    print(ex_gin_s6_df.loc[cob_date])
    # Rs/Candy to Rs/MT conversion factor : 1000 / 355.56 = 2.8124648 (OPERA: 2.810304); same with CCL contract
    ex_gin_s6_relative_returns_df['relative_returns'] = ex_gin_s6_relative_returns_df['EX GIN S6']
    # THIS FX TREATMENT SHOULD NOT BE DONE HERE.
    return ex_gin_s6_df, ex_gin_s6_relative_returns_df


def build_garmmz_sugar_returns(cob_date: str, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # TODO Rewrite function after brought into prices master table
    uat_engine = get_engine('uat')
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    garmmz_sugar = PhysicalPriceLoader(instrument_name='GARMMZ SUGAR', source=uat_engine)
    garmmz_sugar_df = garmmz_sugar.load_garmmz_sugar_price_from_staging(start_date=prices_days_list[0],
                                                                        end_date=cob_date)
    garmmz_sugar_df['date'] = pd.to_datetime(garmmz_sugar_df['date'])
    garmmz_sugar_df = garmmz_sugar_df.set_index('date')[['price']].sort_index()
    garmmz_sugar_df = garmmz_sugar_df.rename(columns={'price': 'GARMMZ SUGAR'})
    garmmz_sugar_df = garmmz_sugar_df.reindex(prices_days_list)
    garmmz_sugar_df = garmmz_sugar_df.interpolate(method='linear', axis=0)
    garmmz_sugar_df = garmmz_sugar_df.bfill().ffill()
    garmmz_sugar_relative_returns_df = relative_returns(garmmz_sugar_df)
    print(garmmz_sugar_df.tail())
    print(garmmz_sugar_df.loc[cob_date])
    # Rs/Candy to Rs/MT conversion factor : 1000 / 355.56 = 2.8124648 (OPERA: 2.810304); same with CCL contract
    garmmz_sugar_relative_returns_df['relative_returns'] = garmmz_sugar_relative_returns_df['GARMMZ SUGAR']
    return garmmz_sugar_df, garmmz_sugar_relative_returns_df


def build_maize_up_returns(cob_date: str, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # TODO Rewrite function after brought into prices master table
    uat_engine = get_engine('uat')
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    maize_up = PhysicalPriceLoader(instrument_name='MAIZE UP', source=uat_engine)
    maize_up_df = maize_up.load_maize_up_price_from_staging(start_date=prices_days_list[0], end_date=cob_date)
    maize_up_df['date'] = pd.to_datetime(maize_up_df['date'])
    maize_up_df = maize_up_df.set_index('date')[['price']].sort_index()
    maize_up_df = maize_up_df.rename(columns={'price': 'MAIZE UP'})
    maize_up_df = maize_up_df.reindex(prices_days_list)
    maize_up_df = maize_up_df.interpolate(method='linear', axis=0)
    maize_up_df = maize_up_df.bfill().ffill()
    maize_up_relative_returns_df = relative_returns(maize_up_df)
    print(maize_up_df.tail())
    print(maize_up_df.loc[cob_date])
    # Rs/Candy to Rs/MT conversion factor : 1000 / 355.56 = 2.8124648 (OPERA: 2.810304); same with CCL contract
    maize_up_relative_returns_df['relative_returns'] = maize_up_relative_returns_df['MAIZE UP']
    return maize_up_df, maize_up_relative_returns_df


def build_cotlook_relative_returns(cob_date: str, window: int) -> dict:
    prod_engine = get_engine('prod')
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)

    cif_crops = ['Burkina Faso Bola/s', 'Brazilian', 'Ivory Coast Manbo/s', 'Mali Juli/s', 'Memphis/Orleans/Texas',
                 'A Index']
    physical_contracts_and_prices = []
    cotlook_dict = {}
    for crop_name in cif_crops:
        crop_params = crop_dict.get(crop_name, {})
        crop_params['data_source'] = 'cotlook'
        physical_contract = PhysicalContractRefLoader(
            instrument_name=crop_name,
            source=prod_engine,
            params=crop_params
        )
        physical_contract.load_ref_data()

        crop_price = PhysicalPriceLoader(
            instrument_name=crop_name,
            source=prod_engine,
            params=physical_contract.params
        )
        crop_price_df = crop_price.load_prices(
            start_date=prices_days_list[0],
            end_date=cob_date,
            data_source='cotlook',
            params=physical_contract.params
        )
        crop_price_df_by_year = crop_price_df.pivot(index='tdate', columns='crop_year', values='px_settle')
        physical_contracts_and_prices.append((physical_contract, crop_price_df_by_year))

        stitched_crop_series = pd.DataFrame(index=crop_price_df_by_year.index)
        crop_year_type = physical_contract.params.get('crop_year_type')
        instrument_name = physical_contract.params.get('instrument_name')

        for crop_year in crop_price_df_by_year.columns:
            if crop_year_type == 'cross':
                ref_year, ref_next_year = map(int, crop_year.split("/"))
            else:
                ref_year = int(crop_year)
                ref_next_year = ref_year + 1
            crop_year_start = pd.Timestamp(f"{ref_year}-08-01")
            crop_year_end = pd.Timestamp(f"{ref_next_year}-07-31")
            for i in crop_price_df_by_year.index:
                if crop_year_start <= i <= crop_year_end:
                    stitched_crop_series.loc[i, instrument_name] = crop_price_df_by_year.loc[i, crop_year]

        stitched_crop_series = stitched_crop_series.reindex(prices_days_list, method='ffill')
        stitched_crop_series = stitched_crop_series.interpolate(method='linear', limit_direction='both', axis=0)

        stitched_crop_series_relative_returns_df = relative_returns(stitched_crop_series)
        # stitched_crop_series_relative_returns_dollarised_df = (
        #        stitched_crop_series_relative_returns_df * stitched_crop_series.loc[cob_date, instrument_name]
        # )

        cotlook_dict[instrument_name] = {'relative_returns': stitched_crop_series_relative_returns_df,
                                         'price_series': stitched_crop_series}
    return cotlook_dict


def build_average_wood_returns(cob_date: str, window: int) -> dict:
    prod_engine = get_engine('prod')
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    wood_series = ['France_DKD_Sapele', 'Netherlands_FKD_Sapelli']
    wood_dict = {}
    all_wood_prices = []

    for wood in wood_series:
        wood_params = wood_series_dict.get(wood, {})
        wood_params['data_source'] = 'BU'
        # Step 1: Load contract metadata
        physical_contract = PhysicalContractRefLoader(
            instrument_name=wood,
            source=prod_engine,
            params=wood_params
        )
        physical_contract.load_ref_data()

        # Step 2: Load price series
        wood_price = PhysicalPriceLoader(
            instrument_name=wood,
            source=prod_engine,
            params=physical_contract.params
        )
        wood_price_df = wood_price.load_prices(
            start_date=prices_days_list[0],
            end_date=cob_date,
            data_source='BU',
            params=physical_contract.params
        )

        wood_price_df = wood_price_df[['tdate', 'px_settle']].set_index('tdate')
        wood_price_df.rename(columns={'px_settle': wood}, inplace=True)
        all_wood_prices.append(wood_price_df)

    merged = pd.concat(all_wood_prices, axis=1)
    merged['avg_price'] = merged.mean(axis=1)
    avg_price = merged[['avg_price']]
    avg_price = avg_price.reindex(prices_days_list)
    avg_price = avg_price.interpolate(method='linear', axis=0)
    avg_price = avg_price.bfill().ffill()
    avg_price_relative_returns = relative_returns(avg_price)
    wood_dict['WOOD AVG'] = {}
    wood_dict['WOOD AVG']['price'] = avg_price.rename(columns={'avg_price': 'WOOD AVG'})
    wood_dict['WOOD AVG']['relative_returns'] = avg_price_relative_returns.rename(columns={'avg_price': 'WOOD AVG'})
    # TODO Where should WOOD AVG FX treatment be?
    return wood_dict


def build_biocane_returns(cob_date: str, window: int):
    biocane_dict = {}
    sugar_df, sugar_relative_returns_df = build_garmmz_sugar_returns(cob_date, window)
    maize_df, maize_relative_returns_df = build_maize_up_returns(cob_date, window)
    biocane_dict['GARMMZ SUGAR'] = {'price': sugar_df,
                                    'relative_returns': sugar_relative_returns_df['relative_returns']
                                    .rename('GARMMZ SUGAR')}
    biocane_dict['MAIZE UP'] = {'price': maize_df,
                                'relative_returns': maize_relative_returns_df['relative_returns'].rename('MAIZE UP')}
    return biocane_dict


def build_instrument_vol_change_dict(instrument_list: list, cob_date: str, window: int) -> Dict[str, Any]:
    rms_engine = get_engine('rms')
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)

    instrument_vol_dict = {}

    for instrument_name in instrument_list:
        vol_change_loader = VolLoader(
            instrument_name=instrument_name,
            source=rms_engine
        )
        vol_change_df = vol_change_loader.load_vol_change_for_generic_curve(
            start_date=prices_days_list[0],
            end_date=cob_date,
            max_generic_curve=9,
            reindex_dates=None,
            instrument_name=instrument_name
        )

        instrument_vol_dict[instrument_name] = vol_change_df

    return instrument_vol_dict


def build_product_prices_returns_dfs_for_hist_sim(cob_date: str, product: str, window: int, simulation_method: str):
    instrument_list = product_specifications[product]['instrument_list']
    usd_conversion_mode = product_specifications[product][simulation_method]['usd_conversion_mode']
    forex_mode = product_specifications[product][simulation_method]['forex_mode']
    fx_spot_df = load_forex(cob_date=cob_date, window=window+1)

    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    returns_days_list = get_prev_biz_days_list(cob_date, window)
    prices_df = pd.DataFrame(index=prices_days_list)
    relative_returns_df = pd.DataFrame(index=returns_days_list)
    instrument_dict = {}

    if product == 'cotton' or product == 'rubber':
        prices_df, relative_returns_df, instrument_dict = build_instrument_generic_curves(instrument_list, cob_date,
                                                                                          window, usd_conversion_mode,
                                                                                          forex_mode, fx_spot_df)

    if product == 'rms':
        # TODO Temporary code until all products updated in product master table
        from position_loader.derivatives_position_loader import DerivativesPositionLoader
        uat_engine = get_engine('uat')
        derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
        rms_deriv_pos_df = derivatives_loader.load_position(
            date=cob_date,
            trader_id=None,
            counterparty_id=None,
            product=product,
            book=None
        )
        rms_deriv_pos_df = rms_deriv_pos_df[~rms_deriv_pos_df['security_id'].astype(str).str.startswith('CR')]
        rms_deriv_pos_df['product_code'] = rms_deriv_pos_df['security_id'].str.split().str[:2].str.join(' ')
        from utils.contract_utils import extract_instrument_from_product_code, load_instrument_ref_dict
        instrument_ref_dict = load_instrument_ref_dict('uat')
        rms_deriv_pos_df['instrument_name'] = (
            rms_deriv_pos_df['product_code']
            .apply(lambda x: extract_instrument_from_product_code(x, instrument_ref_dict))
        )
        instrument_list = list(rms_deriv_pos_df['instrument_name'].unique())
        prices_df, relative_returns_df, instrument_dict = build_instrument_generic_curves(instrument_list, cob_date,
                                                                                          window, usd_conversion_mode,
                                                                                          forex_mode, fx_spot_df)

    if product == 'cotton':

        instrument_dict['PHYS'] = {}
        instrument_dict['PHYS']['EX GIN S6'] = {
            'price_series': None,
            'relative_returns_df': None,
            'currency': 'INR'
        }
        ex_gin_s6_df, ex_gin_s6_relative_returns_df = build_ex_gin_s6_returns(cob_date, window)
        instrument_dict['PHYS']['EX GIN S6']['price_series'] = ex_gin_s6_df
        instrument_dict['PHYS']['EX GIN S6']['relative_returns_df'] = ex_gin_s6_relative_returns_df
        phys_relative_returns_df = ex_gin_s6_relative_returns_df['relative_returns'].to_frame(name='EX GIN S6')
        prices_df = pd.concat([prices_df, ex_gin_s6_df], axis=1)
        relative_returns_df = pd.concat([relative_returns_df, phys_relative_returns_df], axis=1)

        # cotlook_relative_returns_dict = build_cotlook_relative_returns(cob_date, window)
        # instrument_dict['PHYS']['COTLOOK'] = cotlook_relative_returns_dict
        # for cotlook in cotlook_relative_returns_dict:
        #     relative_returns_df = pd.concat([relative_returns_df, cotlook_relative_returns_dict[cotlook]], axis=1)

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
        absolute_returns_df.columns = absolute_returns_df.columns + '_abs'
        instrument_dict['BASIS'] = {}
        instrument_dict['BASIS']['abs_returns_$_df'] = absolute_returns_df

    else:
        absolute_returns_df = pd.DataFrame()

    if product == 'wood':
        wood_dict = build_average_wood_returns(cob_date, window)
        instrument_dict['PHYS'] = {}
        instrument_dict['PHYS']['WOOD'] = wood_dict
        instrument_dict['PHYS']['WOOD']['currency'] = 'EUR'
        prices_df = pd.concat([prices_df, wood_dict['WOOD AVG']['price']], axis=1)
        relative_returns_df = pd.concat([relative_returns_df, wood_dict['WOOD AVG']['relative_returns']], axis=1)

    if product == 'biocane':
        biocane_dict = build_biocane_returns(cob_date, window)
        instrument_dict['PHYS'] = {}
        instrument_dict['PHYS']['BIOCANE'] = biocane_dict
        instrument_dict['PHYS']['BIOCANE']['currency'] = 'INR'
        prices_df = pd.concat([prices_df, biocane_dict['GARMMZ SUGAR']['price'],
                               biocane_dict['MAIZE UP']['price']], axis=1)
        relative_returns_df = pd.concat([relative_returns_df, biocane_dict['GARMMZ SUGAR']['relative_returns'],
                                         biocane_dict['MAIZE UP']['relative_returns']], axis=1)

    if product == 'rms':
        # instrument_vol_dict = build_instrument_vol_change_dict(instrument_list, cob_date, window)
        # for instrument in instrument_dict.keys():
        #     instrument_dict[instrument]['vol_change_df'] = instrument_vol_dict[instrument]
        pass

    returns_df = pd.concat([relative_returns_df, absolute_returns_df], axis=1)
    instrument_dict['FOREX'] = fx_spot_df
    prices_df = prices_df.reindex(returns_days_list)
    returns_df = returns_df.reindex(returns_days_list)

    return prices_df, returns_df, fx_spot_df, instrument_dict


def build_product_prices_returns_dfs_for_mc_sim(cob_date: str, product: str, window: int, simulation_method: str):
    all_products = ['cotton', 'rubber', 'wood', 'biocane', 'rms']
    if simulation_method == 'mc_sim':
        # Load all products for MC simulation
        products_to_load = all_products
    else:
        # Load only the specified product for Historical VaR
        products_to_load = [product]

    usd_conversion_mode = product_specifications[product][simulation_method]['usd_conversion_mode']
    forex_mode = product_specifications[product][simulation_method]['forex_mode']
    fx_spot_df = load_forex(cob_date=cob_date, window=window+1)

    # --- STEP 2: Aggregate ALL instruments across the loading scope ---
    combined_instrument_list = []

    # 1. Collect derivatives/futures instruments from product specifications
    for p_code in products_to_load:
        # Avoid KeyError if a product isn't in specs (shouldn't happen if ALL_PRODUCTS is correct)
        if p_code in product_specifications and 'instrument_list' in product_specifications[p_code]:
            instrument_list = product_specifications[p_code]['instrument_list']
            if instrument_list is not None:
                combined_instrument_list.extend(instrument_list)

    # Remove duplicates from the combined list
    combined_instrument_list = list(set(combined_instrument_list))

    # --- STEP 3: Load all primary futures/derivatives data once ---
    prices_days_list = get_prev_biz_days_list(cob_date, window+1)
    returns_days_list = get_prev_biz_days_list(cob_date, window)
    prices_df = pd.DataFrame(index=prices_days_list)
    relative_returns_df = pd.DataFrame(index=returns_days_list)
    instrument_dict = {}

    if combined_instrument_list:
        # Assuming build_instrument_generic_curves can handle an empty or partially overlapping list
        prices_df, relative_returns_df, instrument_dict = build_instrument_generic_curves(
            combined_instrument_list, cob_date, window, usd_conversion_mode, forex_mode, fx_spot_df
        )

    # --- STEP 4: Load product-specific physical and basis data ---

    # Loop over all products we actually need to load specific data for
    for p_code in products_to_load:

        # 4a. Cotton specific physical/basis data
        if p_code == 'cotton':
            if 'PHYS' not in instrument_dict:
                instrument_dict['PHYS'] = {}

            # EX GIN S6
            instrument_dict['PHYS']['EX GIN S6'] = {
                'price_series': None, 'relative_returns_df': None, 'currency': 'INR'
            }
            ex_gin_s6_df, ex_gin_s6_relative_returns_df = build_ex_gin_s6_returns(cob_date, window)
            instrument_dict['PHYS']['EX GIN S6']['price_series'] = ex_gin_s6_df
            instrument_dict['PHYS']['EX GIN S6']['relative_returns_df'] = ex_gin_s6_relative_returns_df

            phys_relative_returns_df = ex_gin_s6_relative_returns_df['relative_returns'].to_frame(name='EX GIN S6')
            prices_df = pd.concat([prices_df, ex_gin_s6_df], axis=1)
            relative_returns_df = pd.concat([relative_returns_df, phys_relative_returns_df], axis=1)

            # COTLOOK
            cotlook_dict = build_cotlook_relative_returns(cob_date, window)
            instrument_dict['PHYS']['COTLOOK'] = cotlook_dict
            for cotlook in cotlook_dict:
                prices_df = pd.concat([prices_df, cotlook_dict[cotlook]['price_series']], axis=1)
                relative_returns_df = pd.concat([relative_returns_df, cotlook_dict[cotlook]['relative_returns']],
                                                axis=1)

        # 4b. Wood specific physical data
        elif p_code == 'wood':
            wood_dict = build_average_wood_returns(cob_date, window)
            if 'PHYS' not in instrument_dict:
                instrument_dict['PHYS'] = {}

            instrument_dict['PHYS']['WOOD'] = wood_dict
            instrument_dict['PHYS']['WOOD']['currency'] = 'EUR'

            prices_df = pd.concat([prices_df, wood_dict['WOOD AVG']['price']], axis=1)
            relative_returns_df = pd.concat([relative_returns_df, wood_dict['WOOD AVG']['relative_returns']], axis=1)

        # 4c. Biocane specific physical data
        elif p_code == 'biocane':
            biocane_dict = build_biocane_returns(cob_date, window)
            if 'PHYS' not in instrument_dict:
                instrument_dict['PHYS'] = {}

            instrument_dict['PHYS']['BIOCANE'] = biocane_dict
            instrument_dict['PHYS']['BIOCANE']['currency'] = 'INR'

            prices_df = pd.concat([prices_df, biocane_dict['GARMMZ SUGAR']['price'],
                                   biocane_dict['MAIZE UP']['price']], axis=1)
            relative_returns_df = pd.concat([relative_returns_df, biocane_dict['GARMMZ SUGAR']['relative_returns'],
                                             biocane_dict['MAIZE UP']['relative_returns']], axis=1)

    # Add FX data to the dictionary (applies to both single and multi-product scope)
    instrument_dict['FOREX'] = fx_spot_df
    prices_df = prices_df.reindex(returns_days_list)
    relative_returns_df = relative_returns_df.reindex(returns_days_list)

    return prices_df, relative_returns_df, fx_spot_df, instrument_dict
