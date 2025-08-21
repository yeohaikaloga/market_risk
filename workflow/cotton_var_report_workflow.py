import pandas as pd
import numpy as np
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from contract_ref_loader.derivatives_contract_ref_loader import instrument_ref_dict
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from price_series_generator.generic_curve_generator import GenericCurveGenerator
from financial_calculations.returns import relative_returns
from financial_calculations.VaR import calculate_var
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from workflow.cotton_basis_workflow import fy24_cotton_basis_workflow
from position_loader.derivatives_position_loader import DerivativesPositionLoader


def cotton_var_report_workflow(method, cob_date) -> dict:
    prod_engine = get_engine('prod')
    uat_engine = get_engine('uat')
    product = 'cotton'
     # ('for 4 Aug COB, need to put in 2025-08-05')
    days_list = get_prev_biz_days_list(cob_date, 261)
    actual_cob_date = get_prev_biz_days_list(cob_date, 1)[0]  # need to change!
    # NOT COMPLETED YET

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

        # NEED TO ACCOUNT FOR FOREX

        # Step 1C: Generate relative returns for generic price series to get linear PnL vectors
        price_series = GenericCurveGenerator(price_df, futures_contract=derivatives_contract)
        generic_curves_df, active_contracts_df = price_series.generate_generic_curves_df_up_to(max_position=6,
                                                                          roll_days=14,
                                                                          adjustment='ratio',
                                                                          label_prefix=instrument_name)
        print(generic_curves_df.head())
        generic_curves_df = generic_curves_df.replace({pd.NA: np.nan}, inplace=False).astype(float).round(
            3)  # .astype(float).round(3) is limitation of BBG BDH formula -> to remove in future
        print(active_contracts_df.head())
        instrument_dict[instrument_name] = {'relative_returns_df': relative_returns(generic_curves_df),
                                            'contract_to_curve_map': {v.replace(' COMB', ''): k for k, v in
                                                                      active_contracts_df.loc[actual_cob_date].
                                                                      to_dict().items() if v is not None}}
    # Step 2: Generate basis return series
    #basis_abs_ret_df = fy24_cotton_basis_workflow(write_to_excel=False, apply_smoothing=False)
    #print(basis_abs_ret_df.head())

    # Step 3: Generate relative return series for physical prices (Ex Gin S6)

    # Step 4: Generate derivative positions
    position = [1, 1, 1, 1, 1, 1]

    # Step 4A: Generate physical positions
    phys_pos_df = pd.DataFrame()
    # physicals = LoadedPhysicalsPosition(date=cob_date, source=uat_engine)  # will need to change to prod_engine
    # later.
    # phys_pos_df = physicals.load_position(date=cob_date, ors_product='cto')  # check ors_product

    # Step 4B: Determine basis position from physical positions

    # Step 4C: Generate derivatives positions and generic curve mapping
    derivatives = DerivativesPositionLoader(date=cob_date,
                                            source=uat_engine)  # will need to change to prod_engine later.
    deriv_pos_df = derivatives.load_position(date=cob_date, trader_id='all', counterparty_id='all', product=product,
                                             books=None)
    deriv_pos_df = derivatives.assign_bbg_tickers(deriv_pos_df)
    deriv_pos_df = deriv_pos_df[deriv_pos_df['total_active_lots'] != 0]
    deriv_pos_df = derivatives.assign_generic_curves(deriv_pos_df, instrument_dict)
    print(deriv_pos_df.head())


    if product == 'cotton':
        deriv_pos_df = derivatives.assign_cotton_unit(deriv_pos_df)
        # check_derivatives_positions(deriv_pos_df)
    print(deriv_pos_df)

    # Step 4D: Combine all positions into one position table
    combined_pos_df = pd.concat([deriv_pos_df, phys_pos_df], axis=1)
    price_pos_df = deriv_pos_df[deriv_pos_df['books'] == 'PRICE']

    # Step 5: Depending on method, calculate (linear/non-linear) PnL vectors using prices and positions
    for pos in combined_pos_df.index:

        if method == 'linear':

            pnl_mt_df = (relative_returns_df * generic_curves_df.loc[actual_cob_date] *
                     instrument_ref_dict[instrument_name]['price_conversion'] * position)
            pass
        elif method == 'non-linear (sensitivity report)':
            print('sensitivity report method')
            relative_returns_df = pd.DataFrame()
            generic_curves_df = pd.DataFrame()
            pnl_mt_df = pd.DataFrame()
        elif method == 'non-linear (repricing)':
            print('repricing method')
            relative_returns_df = pd.DataFrame()
            generic_curves_df = pd.DataFrame()
            pnl_mt_df = pd.DataFrame()
        else:
            print('wrong method specified')
            relative_returns_df = pd.DataFrame()
            generic_curves_df = pd.DataFrame()
            pnl_mt_df = pd.DataFrame()

    # Step 6: VaR calculation
    var_95 = calculate_var(cob_date, pnl_mt_df, -pnl_mt_df, 95, 260).loc[actual_cob_date]
    var_99 = calculate_var(cob_date, pnl_mt_df, -pnl_mt_df, 99, 260).loc[actual_cob_date]

    final_result = {"price_series": generic_curves_df, "returns": relative_returns_df, "PnL": pnl_mt_df,
                    "VaR_95": var_95, "VaR_99": var_99}
    print(generic_curves_df.head())
    print(relative_returns_df.head())
    print(pnl_mt_df.head())
    print(var_95)
    print(var_99)
    return final_result
