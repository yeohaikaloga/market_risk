import pandas as pd
import numpy as np
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
from quality_checker.cotton_var_checker import check_positions


def generate_var_workflow(product, method, cob_date, window) -> dict[str, pd.DataFrame]:
    """
        Generates VaR data for a given product and method.

        Args:
            product (str): Product name (e.g., 'cotton').
            method (str): VaR calculation method ('linear', 'non-linear').
            cob_date (str): COB date in 'YYYY-MM-DD'.
            window (int): Lookback window in business days.

        Returns:
            dict[str, pd.DataFrame]: Dictionary with 'main' VaR table,
                                     and optionally 'price' VaR table for cotton.
    """

    method_id_dict = {'linear': 'L', 'non-linear': 'NL'}
    prod_engine = get_engine('prod')
    uat_engine = get_engine('uat')
     # ('for 4 Aug COB, need to put in 2025-08-05')
    days_list = get_prev_biz_days_list(cob_date, window+1) # need to change!

    instruments_list = ['CT', 'VV', 'CCL', 'S ', 'W ', 'C ', 'SB']
    instrument_dict = {}
    for instrument_name in instruments_list:

        # Step 1A: Load contract_ref_loader metadata
        derivatives_contract = DerivativesContractRefLoader(instrument_name=instrument_name, source=prod_engine)
        relevant_months = ('H', 'K', 'N', 'Z') if instrument_name == 'CT' else None
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
        #Step 1C: Load forex prices data

        # Step 1D: Generate relative returns for generic price series to get linear PnL vectors
        price_series = GenericCurveGenerator(price_df, futures_contract=derivatives_contract)
        generic_curves_df, active_contracts_df = price_series.generate_generic_curves_df_up_to(max_position=6,
                                                                          roll_days=14,
                                                                          adjustment='ratio',
                                                                          label_prefix=instrument_name)
        print(generic_curves_df.head())
        generic_curves_df = generic_curves_df.replace({pd.NA: np.nan}, inplace=False).astype(float).round(
            3)  # .astype(float).round(3) is limitation of BBG BDH formula -> to remove in future
        print(active_contracts_df.head())
        print(generic_curves_df.loc[cob_date])
        instrument_dict[instrument_name] = {'generic_curves_df': generic_curves_df,
                                            'relative_returns_df': relative_returns(generic_curves_df),
                                            'relative_returns_$_df': relative_returns(generic_curves_df) *
                                                                     generic_curves_df.loc[cob_date],
                                            'contract_to_curve_map': {v.replace(' COMB', ''): k for k, v in
                                                                      active_contracts_df.loc[cob_date].
                                                                      to_dict().items() if v is not None}}

    # Step 2: Generate basis return series
    if product == 'cotton':
        #basis_abs_ret_df = fy24_cotton_basis_workflow(write_to_excel=False, apply_smoothing=False)
        #print(basis_abs_ret_df.head())
        pass

    # Step 3: Generate relative return series for physical prices (Ex Gin S6)

    # Step 4A: Generate physical positions
    phys_pos_df = pd.DataFrame()

    # physicals = LoadedPhysicalsPosition(date=cob_date, source=uat_engine)
    # TODO will need to change to prod_engine later.
    # phys_pos_df = physicals.load_position(date=cob_date, ors_product='cto')
    # TODO check ors_product
    if product == 'cotton':
        pass

    # Step 4B: Generate derivatives positions and generic curve mapping
    derivatives = DerivativesPositionLoader(date=cob_date,
                                            source=uat_engine)  # will need to change to prod_engine later.
    deriv_pos_df = derivatives.load_position(date=cob_date, trader_id='all', counterparty_id='all', product=product,
                                             books=None)
    deriv_pos_df = derivatives.load_sensitivities(deriv_pos_df, sensitivity_type='settle_delta_1')
    deriv_pos_df = derivatives.assign_bbg_tickers(deriv_pos_df)
    deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
    deriv_pos_df = derivatives.assign_generic_curves(deriv_pos_df, instrument_dict)

    if product == 'cotton':
        deriv_pos_df = derivatives.assign_cotton_unit(deriv_pos_df)
        deriv_pos_df['exposure'] = 'OUTRIGHT'
    elif product == 'rubber':
        # deriv_pos_df = derivatives.assign_rubber_unit(deriv_pos_df)
        pass
    elif product == 'rms':
        pass
    print(deriv_pos_df.head())

    # Step 5: Repricing of options. Generate delta for all options (positions)
    if method == 'linear':
        pass

    else:
        # Reprice
        pass

    # Step 6A: Calculate for outright, basis adjustment and basis position at unit level.
    if method == 'linear':
        pass

    else:
        pass
    basis_adj_pos_df = pd.DataFrame()
    # Step 6B: Combine all positions into one position table

    combined_pos_df = pd.concat([deriv_pos_df, phys_pos_df, basis_adj_pos_df], axis=1)
    combined_pos_df['product'] = product
    combined_pos_df['cob_date'] = cob_date
    combined_pos_df = combined_pos_df.reset_index(drop=True)
    combined_pos_df['position_index'] = (product[:3] + '_' + method_id_dict[method] + '_' + str(cob_date) + '_' +
                                         combined_pos_df.index.map(lambda i: str(i).zfill(4)))

    combined_pos_df['instrument_name'] = combined_pos_df['product_code'].apply(extract_instrument_name)
    combined_pos_df['to_USD_conversion'] = (combined_pos_df['instrument_name']
                                            .map(lambda x: instrument_ref_dict.get(x, {})
                                                 .get('to_USD_conversion', np.nan)))
    combined_pos_df['lots_to_MT_conversion'] = (combined_pos_df['instrument_name']
                                                .map(lambda x: instrument_ref_dict.get(x, {})
                                                     .get('lots_to_MT_conversion', np.nan)))
    combined_pos_df['conversion_factor'] = combined_pos_df['to_USD_conversion'] * combined_pos_df['lots_to_MT_conversion']
    combined_pos_df['delta'] = (combined_pos_df['total_active_lots'] * combined_pos_df['settle_delta_1']
                                * combined_pos_df['lots_to_MT_conversion'])

    if product == 'cotton':
        check_positions(combined_pos_df, cob_date, units='delta')

    # Step 7: Depending on method, calculate (linear/non-linear) PnL vectors using prices and positions
    # Adjust lookback pnl df etc. from here on...

    # TODO vectorising it by taking to_usd and lots_to_mt from a df instead
    # Prepare a list to collect all pnl DataFrames
    pnl_dfs = []

    # Vectorized processing over rows (still a loop, but pandas operations inside)
    for idx, row in combined_pos_df.iterrows():
        instrument_name = row['instrument_name']
        generic_curve = row['generic_curve']
        position_index = row['position_index']
        delta = row['delta']
        to_usd = row['to_USD_conversion']

        if method == 'linear':
            #TODO validate PnL is correctly calculated; suspect it is different by a multiple
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

        elif method == 'non-linear (repricing)':
            # TODO add non-linear (repricing) methodology
            pass

        elif method == 'non-linear (MC)':
            pass

        else:
            print('wrong method specified')

    long_pnl_df = pd.concat(pnl_dfs, ignore_index=True)
    print(long_pnl_df.head())

    analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)
    outright_analyzer = analyzer.filter(exposure='OUTRIGHT')
    unit_outright_lookback_pnl_df = outright_analyzer.pivot(index='pnl_date', columns='unit', values='lookback_pnl')
    unit_outright_inverse_pnl_df = outright_analyzer.pivot(index='pnl_date', columns='unit', values='inverse_pnl')
    print(unit_outright_lookback_pnl_df.head())
    print(unit_outright_inverse_pnl_df.head())

    if product == 'cotton':
        price_pos_indices = combined_pos_df[combined_pos_df['books'] == 'PRICE']['position_index'].unique()
        price_outright_analyzer = outright_analyzer.filter(position_index=price_pos_indices)

        price_unit_lookback_pnl_df = price_outright_analyzer.pivot(index='pnl_date', columns='unit',
                                                                   values='lookback_pnl')
        price_unit_inverse_pnl_df = price_outright_analyzer.pivot(index='pnl_date', columns='unit',
                                                                  values='inverse_pnl')

    basis_analyzer = analyzer.filter(exposure='BASIS')
    unit_basis_lookback_pnl_df = basis_analyzer.pivot(index='pnl_date', columns='unit', values='lookback_pnl')
    unit_basis_inverse_pnl_df = basis_analyzer.pivot(index='pnl_date', columns='unit', values='inverse_pnl')

    unit_overall_lookback_pnl_df = analyzer.pivot(index='pnl_date', columns='unit', values='lookback_pnl')
    unit_overall_inverse_pnl_df = analyzer.pivot(index='pnl_date', columns='unit', values='inverse_pnl')
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

    # Step 8: VaR calculation
    percentiles = [95, 99]

    if product == 'cotton':
        aggregate_unit_dict = get_cotton_region_aggregates(combined_pos_df['region'].unique().tolist())
    elif product == 'rubber':
        aggregate_unit_dict = {}
        pass
    else:
        aggregate_unit_dict = {}
        pass

    var_calc = VaRCalculator()
    unit_var_dict, aggregate_var_dict = calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict, cob_date,
                                                                         percentiles, window,
                                                                         unit_outright_lookback_pnl_df,
                                                                         unit_outright_inverse_pnl_df,
                                                                         unit_basis_lookback_pnl_df,
                                                                         unit_basis_inverse_pnl_df)
    price_unit_var_dict, price_unit_var_dict = calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict,
                                                                                cob_date, percentiles, window,
                                                                                price_unit_lookback_pnl_df,
                                                                                price_unit_inverse_pnl_df)

    # Step 9. Store VaR results into VaR table
    # TODO: VaR table should eventually go into DB for storage daily
    var_table_df = format_var_results(unit_var_dict, aggregate_var_dict, cob_date=cob_date, method=method)
    price_var_table_df = format_var_results(price_unit_var_dict, price_unit_var_dict, cob_date=cob_date,
                                            method=method, exposure_override='PRICE')
    print(var_table_df.head())
    print(price_var_table_df.head())
    return {'delta': combined_pos_df, 'main': var_table_df,
            **({'price': price_var_table_df} if product == 'cotton' else {})}

