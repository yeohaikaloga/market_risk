import pandas as pd
import numpy as np
from datetime import datetime
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from contract_ref_loader.derivatives_contract_ref_loader import instrument_ref_dict
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from price_series_generator.generic_curve_generator import GenericCurveGenerator
from financial_calculations.returns import relative_returns
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from financial_calculations.var_calculator import VaRCalculator
from pnl_analyzer.pnl_analyzer import PnLAnalyzer
from utils.var_utils import calculate_unit_and_aggregate_var
from utils.var_utils import format_var_results
from utils.var_utils import get_cotton_region_aggregates
from utils.contract_utils import extract_instrument_name
from quality_checker.cotton_var_checker import check_pivot_positions
from position_loader.physical_position_loader import PhysicalPositionLoader
from utils.position_utils import calculate_aggregates
from utils.position_utils import calculate_basis_adj_and_basis_pos
from quality_checker.cotton_var_checker import validate_combined_position_df
from workflow.var_report_builder_workflow import build_var_report

def generate_product_generic_curves(product, cob_date, window) -> dict:
    prod_engine = get_engine('prod')
    days_list = get_prev_biz_days_list(cob_date, window + 1)  # need to change!
    if product == 'cotton':
        instruments_list = ['CT', 'VV', 'CCL', 'S ', 'W ', 'C ', 'SB']
    else:
        instruments_list = ['CT']

    instrument_dict = {}
    for instrument_name in instruments_list:
        # Step 1A: Load contract_ref_loader metadata
        derivatives_contract = DerivativesContractRefLoader(instrument_name=instrument_name, source=prod_engine)
        relevant_months = None
        # relevant_months = ('H', 'K', 'N', 'Z') if instrument_name == 'CT' else None # TODO Re-included when validation completed
        futures_contracts = derivatives_contract.load_contracts(mode='futures', relevant_months=relevant_months,
                                                                relevant_years=None, relevant_options=None)

        # Step 1B: Load prices data for these futures contracts
        futures_price = DerivativesPriceLoader(mode='futures', instrument_name=instrument_name, source=prod_engine)
        price_df = futures_price.load_prices(start_date=days_list[0],
                                             end_date=cob_date,
                                             contracts=futures_contracts,
                                             reindex_dates=days_list,
                                             instrument_name=instrument_name)

        # TODO: Add in code for forex calcs for non-USD denom contracts
        # Step 1C: Load forex prices data

        # Step 1D: Generate relative returns for generic price series to get linear PnL vectors
        price_series = GenericCurveGenerator(price_df, futures_contract=derivatives_contract)
        max_position = 13 if instrument_name == 'CT' else 6  # NOTE CT is calculated up till CT13, rest up till xx6
        generic_curves_df, active_contracts_df = price_series.generate_generic_curves_df_up_to(
            max_position=max_position,
            roll_days=14,
            adjustment='ratio',
            label_prefix=instrument_name)
        print(generic_curves_df.head())
        generic_curves_df = generic_curves_df.replace({pd.NA: np.nan}, inplace=False).astype(float).round(
            3)  # TODO .astype(float).round(3) is limitation of BBG BDH formula -> to remove in future
        print(active_contracts_df.head())
        print(generic_curves_df.loc[cob_date])
        instrument_dict[instrument_name] = {'generic_curves_df': generic_curves_df,
                                            'relative_returns_df': relative_returns(generic_curves_df),
                                            'relative_returns_$_df': relative_returns(generic_curves_df) *
                                                                     generic_curves_df.loc[cob_date],
                                            'contract_to_curve_map': {v.replace(' COMB', ''): k for k, v in
                                                                      active_contracts_df.loc[cob_date].
                                                                      to_dict().items() if v is not None}}
    return instrument_dict


def generate_cotton_combined_position(cob_date, instrument_dict) -> pd.DataFrame:
    product = 'cotton'
    uat_engine = get_engine('uat')

    # Step 4A: Generate physical positions
    # TODO working with staging data for now, will need to change to master physicals position table later.
    physicals = PhysicalPositionLoader(date=cob_date, source=uat_engine)
    conso_pos_df = physicals.load_cotton_phy_position_from_staging(cob_date=cob_date)
    conso_pos_df.columns = [col.lower() for col in conso_pos_df.columns]
    pos_type_conditions = [conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),
                           conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),
                           conso_pos_df['typ'].isin(
                               ['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])]
    pos_type_values = ['FIXED PHYS', 'DIFF PHYS', 'DERIVS']
    conso_pos_df['position_type'] = np.select(pos_type_conditions, pos_type_values, default='Unknown')

    # This is placeholder - only to calculate basis adjustment needed under old Basis definition for Step 4A-1
    exposure_conditions = [conso_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES']),  # Outright condition
                           conso_pos_df['typ'].isin(['DIFF PURCHASE', 'DIFF SALES']),  # Basis condition
                           conso_pos_df['typ'].isin(
                               ['FUTURES', 'LONG CALLS', 'SHORT CALLS', 'LONG PUTS', 'SHORT PUTS'])]
    exposure_values = ['OUTRIGHT', 'BASIS', 'OUTRIGHT']
    conso_pos_df['exposure_for_old_basis'] = np.select(exposure_conditions, exposure_values, default='Unknown')
    conso_pos_df['contract'] = '-'
    for i in conso_pos_df.index:
        unit = conso_pos_df.loc[i, 'unit']
        position_type = conso_pos_df.loc[i, 'position_type']
        exposure = conso_pos_df.loc[i, 'exposure_for_old_basis']
        if unit == 'CHINA' and position_type == 'FIXED PHYS' and exposure == 'OUTRIGHT':  # CHINA adjustment
            conso_pos_df.loc[i, 'contract'] = 'VV'
        else:
            conso_pos_df.loc[i, 'contract'] = 'CT'
        # TODO India adjustment
    print(conso_pos_df.head())

    # Step 4A-1: Prepare aggregated outright, basis position
    # NOTE This basis position is not used, instead basis = net phys so all physical positions are Basis, while
    # all fixed physical positions are Outright as well
    aggregated_df = conso_pos_df.groupby(['region', 'contract']).apply(calculate_aggregates).reset_index()
    aggregated_df['outright_pos'] = aggregated_df['net_fixed_phys'] + aggregated_df['derivs']
    aggregated_df[['basis_adj', 'basis_pos']] = (
        aggregated_df.apply(lambda row: pd.Series(calculate_basis_adj_and_basis_pos(row)), axis=1))
    print(aggregated_df)

    phy_pos_df = conso_pos_df[conso_pos_df['position_type'] != 'DERIVS']
    phy_pos_df = phy_pos_df.rename(columns={'contract': 'instrument_name', 'quantity': 'delta'})
    phy_grouped_pos_df = phy_pos_df.groupby(['region', 'typ', 'position_type', 'terminal_month', 'instrument_name'],
                                            as_index=False)['delta'].sum()
    phy_grouped_pos_df = phy_grouped_pos_df[phy_grouped_pos_df['delta'] != 0]
    phy_grouped_pos_df['subportfolio'] = phy_grouped_pos_df['typ']
    phy_grouped_pos_df['strike'] = np.nan
    phy_grouped_pos_df['books'] = 'PHYSICALS'
    phy_grouped_pos_df['trader_id'] = 1
    phy_grouped_pos_df['trader_name'] = None
    phy_grouped_pos_df['counterparty_id'] = 0
    phy_grouped_pos_df['counterparty_parent'] = None
    phy_grouped_pos_df = physicals.assign_bbg_tickers(phy_grouped_pos_df)
    phy_grouped_pos_df = physicals.assign_generic_curves(phy_grouped_pos_df, instrument_dict)
    phy_grouped_pos_df['to_USD_conversion'] = (phy_grouped_pos_df['instrument_name']
                                               .map(lambda x: instrument_ref_dict.get(x, {})
                                                    .get('to_USD_conversion', np.nan)))
    print(phy_grouped_pos_df.head())
    basis_phy_pos_df = phy_grouped_pos_df.copy()
    basis_phy_pos_df['exposure'] = 'BASIS (NET PHYS)'
    outright_phy_pos_df = phy_grouped_pos_df[phy_grouped_pos_df['typ'].isin(['FIXED PURCHASE', 'FIXED SALES'])]
    outright_phy_pos_df['exposure'] = 'OUTRIGHT'

    cotton_unit_region_mapping = dict(zip(conso_pos_df['unit'], conso_pos_df['region']))
    print(cotton_unit_region_mapping)

    # Step 4B: Generate derivatives positions and generic curve mapping
    # NOTE All derivatives = Outright position
    derivatives = DerivativesPositionLoader(date=cob_date,
                                            source=uat_engine)  # TODO change to prod_engine later.
    deriv_pos_df = derivatives.load_position(date=cob_date, trader_id='all', counterparty_id='all', product=product,
                                             books=None)
    sensitivity_df = derivatives.load_opera_sensitivities(deriv_pos_df, sensitivity_type='settle_delta_1')
    deriv_pos_df = derivatives.assign_opera_sensitivities(deriv_pos_df, sensitivity_df,
                                                          sensitivity_type='settle_delta_1')
    deriv_pos_df = derivatives.assign_bbg_tickers(deriv_pos_df)
    deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
    deriv_pos_df = derivatives.assign_generic_curves(deriv_pos_df, instrument_dict)
    deriv_pos_df['instrument_name'] = deriv_pos_df['product_code'].apply(extract_instrument_name)
    deriv_pos_df = derivatives.assign_cotton_unit(deriv_pos_df)
    deriv_pos_df['region'] = deriv_pos_df['unit'].map(
        cotton_unit_region_mapping)  # Overwrite unit-region in DB with that from CONSOLIDATE tab
    deriv_pos_df['position_type'] = 'DERIVS'
    deriv_pos_df['exposure'] = 'OUTRIGHT'
    deriv_pos_df['to_USD_conversion'] = (deriv_pos_df['instrument_name']
                                         .map(lambda x: instrument_ref_dict.get(x, {})
                                              .get('to_USD_conversion', np.nan)))
    deriv_pos_df['lots_to_MT_conversion'] = (deriv_pos_df['instrument_name']
                                             .map(lambda x: instrument_ref_dict.get(x, {})
                                                  .get('lots_to_MT_conversion', np.nan)))
    deriv_pos_df['conversion_factor'] = deriv_pos_df['to_USD_conversion'] * deriv_pos_df['lots_to_MT_conversion']
    deriv_pos_df['delta'] = (deriv_pos_df['total_active_lots'] * deriv_pos_df['settle_delta_1']
                             * deriv_pos_df['lots_to_MT_conversion'])
    print(deriv_pos_df.head())

    # Step 4C: Combine all positions into one df
    combined_pos_df = pd.concat([deriv_pos_df, outright_phy_pos_df, basis_phy_pos_df], axis=0)
    combined_pos_df['product'] = product
    combined_pos_df['cob_date'] = cob_date
    combined_pos_df = combined_pos_df.reset_index(drop=True)
    combined_pos_df = combined_pos_df[combined_pos_df['region'].apply(lambda x: isinstance(x, str))]
    combined_pos_df = combined_pos_df[['cob_date', 'product', 'unit', 'region', 'books', 'position_type', 'exposure',
                                       'trader_id', 'trader_name', 'counterparty_id', 'counterparty_parent',
                                       'instrument_name', 'bbg_ticker', 'underlying_bbg_ticker', 'generic_curve',
                                       'delta', 'to_USD_conversion']]
    print(combined_pos_df.head())
    validate_combined_position_df(combined_pos_df)
    return combined_pos_df


def generate_pnl_vectors(combined_pos_df, instrument_dict, method) -> pd.DataFrame():
    method_id_dict = {'linear': 'L', 'non-linear': 'NL'}
    product_list = combined_pos_df['product'].unique()
    cob_date_list = combined_pos_df['cob_date'].unique()
    if len(product_list) != 1:
        raise ValueError(f"Expected a single unique product, found: {product_list}")

    if len(cob_date_list) != 1:
        raise ValueError(f"Expected a single unique cob_date, found: {cob_date_list}")

    product = product_list[0]
    cob_date = cob_date_list[0]

    combined_pos_df['position_index'] = (product[:3] + '_' + method_id_dict[method] + '_' + str(cob_date) + '_' +
                                         combined_pos_df.index.map(lambda i: str(i).zfill(4)))
    # combined_pos_df['position_index'] = (combined_pos_df['generic_curve'] + '_' + combined_pos_df['subportfolio'] + '_'
    #                                     + combined_pos_df['region'] + '_' + combined_pos_df['position_type'] + '_'
    #                                     + combined_pos_df['trader_id'].astype(str) + '_'
    #                                     + combined_pos_df['counterparty_id'].astype(str))


    # Vectorized processing over rows (still a loop, but pandas operations inside)
    pnl_dfs = []
    for idx, row in combined_pos_df.iterrows():
        # print(f"[DEBUG] Row {idx}: instrument_name={row.get('instrument_name')}, "
        #      f"generic_curve={row.get('generic_curve')}, position_index={row.get('position_index')}, "
        #      f"delta={row.get('delta')}, to_usd={row.get('to_USD_conversion')}")
        instrument_name = row['instrument_name']
        generic_curve = row['generic_curve']
        position_index = row['position_index']
        delta = row['delta']
        to_usd = row['to_USD_conversion']
        if method == 'linear':
            returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][generic_curve]
            pnl_series = delta * returns_series * to_usd

            # Create a DataFrame for this position's pnl with pnl_date as index
            df = pnl_series.to_frame(name='lookback_pnl')
            df['inverse_pnl'] = -df['lookback_pnl']
            df['position_index'] = position_index
            df['cob_date'] = cob_date
            df['method'] = method
            df = df.reset_index().rename(columns={'date': 'pnl_date'})
            pnl_dfs.append(df)
        long_pnl_df = pd.concat(pnl_dfs, ignore_index=True)
    return long_pnl_df


def pnl_analyser(product, long_pnl_df, combined_pos_df) -> dict:
    analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)
    outright_analyzer = analyzer.filter(exposure='OUTRIGHT')
    unit_outright_lookback_pnl_df = outright_analyzer.pivot(index='pnl_date', columns='region', values='lookback_pnl')
    unit_outright_inverse_pnl_df = outright_analyzer.pivot(index='pnl_date', columns='region', values='inverse_pnl')
    print(unit_outright_lookback_pnl_df.head())
    print(unit_outright_inverse_pnl_df.head())

    if product == 'cotton':
        price_pos_indices = combined_pos_df[combined_pos_df['books'] == 'PRICE']['position_index'].unique()
        if not price_pos_indices.any():
            print("No PRICE positions found.")
            price_unit_lookback_pnl_df = pd.DataFrame()
        else:
            price_outright_analyzer = outright_analyzer.filter(position_index=price_pos_indices)

            price_unit_lookback_pnl_df = price_outright_analyzer.pivot(index='pnl_date', columns='region',
                                                                       values='lookback_pnl')
            price_unit_inverse_pnl_df = price_outright_analyzer.pivot(index='pnl_date', columns='region',
                                                                      values='inverse_pnl')

    basis_analyzer = analyzer.filter(exposure='BASIS (NET PHYS)')
    unit_basis_lookback_pnl_df = basis_analyzer.pivot(index='pnl_date', columns='region', values='lookback_pnl')
    unit_basis_inverse_pnl_df = basis_analyzer.pivot(index='pnl_date', columns='region', values='inverse_pnl')

    unit_overall_lookback_pnl_df = analyzer.pivot(index='pnl_date', columns='region', values='lookback_pnl')
    unit_overall_inverse_pnl_df = analyzer.pivot(index='pnl_date', columns='region', values='inverse_pnl')
    print(unit_overall_lookback_pnl_df.head())
    print(unit_overall_inverse_pnl_df.head())

    unit_outright_combined_pnl_df = pd.concat([unit_outright_lookback_pnl_df, unit_outright_inverse_pnl_df],
                                              keys=['lookback', 'inverse'])
    print(unit_outright_combined_pnl_df.head())
    unit_basis_combined_pnl_df = pd.concat([unit_basis_lookback_pnl_df, unit_basis_inverse_pnl_df],
                                           keys=['lookback', 'inverse'])
    print(unit_basis_combined_pnl_df.head())
    unit_overall_combined_pnl_df = pd.concat([unit_overall_lookback_pnl_df, unit_overall_inverse_pnl_df],
                                             keys=['lookback', 'inverse'])
    print(unit_overall_combined_pnl_df.head())

    outright_lookback_pnl_df = outright_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'],
                                                       values='lookback_pnl')
    outright_inverse_pnl_df = outright_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'],
                                                      values='inverse_pnl')
    basis_lookback_pnl_df = basis_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'],
                                                 values='lookback_pnl')
    cob_date = combined_pos_df['cob_date'].unique()[0]
    filename = f"{cob_date}_var_output.xlsx"
    with pd.ExcelWriter(filename, mode='a', if_sheet_exists='replace') as writer:
        combined_pos_df.to_excel(writer, sheet_name='combined_pos')
        outright_lookback_pnl_df.to_excel(writer, sheet_name='outright_lookback')
        outright_inverse_pnl_df.to_excel(writer, sheet_name='outright_inverse')
        basis_lookback_pnl_df.to_excel(writer, sheet_name='basis_lookback')
    return {'delta': combined_pos_df,
            'main': {'outright_lookback_pnl': outright_lookback_pnl_df, 'outright_inverse_pnl': outright_inverse_pnl_df,
                     'basis_lookback_pnl': basis_lookback_pnl_df},
            'unit': {'outright_lookback_pnl': unit_outright_lookback_pnl_df,
                     'outright_inverse_pnl': unit_outright_inverse_pnl_df,
                     'basis_lookback_pnl': unit_basis_lookback_pnl_df, 'basis_inverse_pnl': unit_basis_inverse_pnl_df},
            'price': {'price_outright_lookback_pnl': price_unit_lookback_pnl_df,
                      'price_outright_inverse_pnl': price_unit_inverse_pnl_df}}


def generate_var_table(product, aggregate_unit_dict, pnl_dict, cob_date, percentiles, window, method) -> dict:
    unit_outright_lookback_pnl_df = pnl_dict['unit']['outright_lookback_pnl']
    unit_outright_inverse_pnl_df = pnl_dict['unit']['outright_inverse_pnl']
    unit_basis_lookback_pnl_df = pnl_dict['unit']['basis_lookback_pnl']
    unit_basis_inverse_pnl_df = pnl_dict['unit']['basis_inverse_pnl']
    price_unit_lookback_pnl_df = pnl_dict['price']['price_outright_lookback_pnl']
    price_unit_inverse_pnl_df = pnl_dict['price']['price_outright_inverse_pnl']

    var_calc = VaRCalculator()
    unit_var_dict, aggregate_var_dict = calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict, cob_date,
                                                                         percentiles, window,
                                                                         unit_outright_lookback_pnl_df,
                                                                         unit_outright_inverse_pnl_df,
                                                                         unit_basis_lookback_pnl_df,
                                                                         unit_basis_inverse_pnl_df)
    if product == 'cotton':
        price_unit_var_dict, price_aggregate_var_dict = calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict,
                                                                                         cob_date, percentiles, window,
                                                                                         price_unit_lookback_pnl_df,
                                                                                         price_unit_inverse_pnl_df)

    # Step 8: Store VaR results into VaR table
    # TODO: VaR table should eventually go into DB for storage daily
    var_table_df = format_var_results(unit_var_dict, aggregate_var_dict, cob_date=cob_date, method=method)
    if product == 'cotton':
        price_var_table_df = format_var_results(price_unit_var_dict, price_aggregate_var_dict, cob_date=cob_date,
                                            method=method, exposure_override='PRICE')
    return {'main': var_table_df, 'price': price_var_table_df}


def generate_product_var_workflow(product, method, cob_date, window) -> dict:
    method_id_dict = {'linear': 'L', 'non-linear': 'NL'}
    prod_engine = get_engine('prod')
    uat_engine = get_engine('uat')
    days_list = get_prev_biz_days_list(cob_date, window + 1)

    # Step 1: Generate relative returns for generic price series to get linear PnL vectors
    instrument_dict = generate_product_generic_curves(product, cob_date, window)

    # Step 2: Generate basis absolute return series
    if product == 'cotton':
        basis_abs_ret_df = fy24_cotton_basis_workflow(write_to_excel=False, apply_smoothing=False)
        print(basis_abs_ret_df.head())

    # Step 3: Generate relative return series for physical prices (Ex Gin S6)

    # Step 4: Combine all positions into one df
    if product == 'cotton':
        combined_pos_df = generate_cotton_combined_position(cob_date, instrument_dict)

    # Step 5: PnL calculation methodology
    if method == 'linear':
        long_pnl_df = generate_pnl_vectors(combined_pos_df, instrument_dict, method)
    elif method == 'non-linear (repricing)':
        #generate_pnl_vectors(combined_pos_df, instrument_dict, method)
        pass
    elif method == 'non-linear (MC)':
        # generate_pnl_vectors(combined_pos_df, instrument_dict, method)
        pass
    else:
        pass

    # Step 6: Store PnLs in dict by unit and position
    pnl_dict = pnl_analyser(product, long_pnl_df, combined_pos_df)
    unit_outright_lookback_pnl_df = pnl_dict['unit']['outright_lookback_pnl']
    unit_outright_inverse_pnl_df = pnl_dict['unit']['outright_inverse_pnl']
    unit_basis_lookback_pnl_df = pnl_dict['unit']['basis_lookback_pnl']
    unit_basis_inverse_pnl_df = pnl_dict['unit']['basis_inverse_pnl']
    price_unit_lookback_pnl_df = pnl_dict['price']['price_outright_lookback_pnl']
    price_unit_inverse_pnl_df = pnl_dict['price']['price_outright_inverse_pnl']


    # Step 7: VaR calculation
    percentiles = [95, 99]
    if product == 'cotton':
        aggregate_unit_dict = get_cotton_region_aggregates(combined_pos_df['region'].unique().tolist())

    var_calc = VaRCalculator()
    unit_var_dict, aggregate_var_dict = calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict, cob_date,
                                                                         percentiles, window,
                                                                         unit_outright_lookback_pnl_df,
                                                                         unit_outright_inverse_pnl_df,
                                                                         unit_basis_lookback_pnl_df,
                                                                         unit_basis_inverse_pnl_df)
    if product == 'cotton':
        price_unit_var_dict, price_aggregate_var_dict = calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict,
                                                                                         cob_date, percentiles, window,
                                                                                         price_unit_lookback_pnl_df,
                                                                                         price_unit_inverse_pnl_df)


    # Step 8: Generate and store VaR results into VaR table
    # TODO: VaR table should eventually go into DB for storage daily
    var_dict = generate_var_table(product, aggregate_unit_dict, pnl_dict, cob_date, percentiles, window, method)

    # Step 9: Build VaR report
    all_books_var = var_dict['main']
    var_report = build_var_report(product=product, books='all', cob_date=cob_date, pos_df=combined_pos_df,
                                  var_df=all_books_var)
    if product == 'cotton':
        price_pos = combined_pos_df[combined_pos_df['books'] == 'PRICE']
        price_book_var = var_dict['price']
        cotton_price_var_report = build_var_report(product=product, books='price', cob_date=cob_date,
                                                   pos_df=price_pos, var_df=price_book_var)
        return var_report, cotton_price_var_report
