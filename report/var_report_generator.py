import pandas as pd
import numpy as np
from contract_ref_loader.derivatives_contract_ref_loader import FuturesContract
from contract_ref_loader.derivatives_contract_ref_loader import instrument_ref_dict
from price_series_loader.derivatives_price_loader import LoadedFuturesPrice
from price_series_generator.generic_curve_generator import GenericCurveGenerator
from financial_calculations.returns import relative_returns
from financial_calculations.VaR import calculate_var


def cotton_var_report_workflow(instruments_list: list, cob_date: str, days_list: list[str], position: list[float],
                               engine, product, method) -> dict:
    # NOT COMPLETED YET
    instruments_list = ['CT', 'VV', 'CCL', 'BO']
    for instrument in instruments_list:
        # Step 1: Load contract_ref_loader metadata
        futures_contract = FuturesContract(instrument_id=instrument, source=engine)
        futures_contract.load_ref_data()
        contracts = futures_contract.load_contracts()

        # Step 2: Load price_series_loader data for these contracts
        futures_price = LoadedFuturesPrice(instrument_id=futures_contract.instrument_id, source=engine)
        if instrument == 'CT':
            selected_contracts = [c for c in contracts if c[-2] in {'H', 'K', 'N', 'Z'}]
        else:
            selected_contracts = contracts
        price_df = futures_price.load_prices(start_date=days_list[0],
                                             end_date=cob_date,
                                             selected_contracts=selected_contracts,
                                             reindex_dates=days_list,
                                             instrument_id=instrument)

        # Step 3: Generate generic price_series_loader series for linear PnL vectors
        price_series = GenericCurveGenerator(price_df, futures_contract=futures_contract)
        generic_curves_df = price_series.generate_generic_curves_df_up_to(max_position=6,
                                                                          roll_days=14,
                                                                          adjustment='ratio',
                                                                          label_prefix=instrument)
        print(generic_curves_df)
        generic_curves_df = generic_curves_df.replace({pd.NA: np.nan}, inplace=False).astype(float).round(
            3)  # .astype(float).round(3) is limitation of BBG BDH formula -> to remove in future
        relative_returns_df = relative_returns(generic_curves_df)

        # Step 4: Generate basis return series
        basis_abs_ret_df = fy24_cotton_basis_workflow(write_to_excel=True, apply_smoothing=False)

        # Step 5: Generate positions and generic curve mapping

        # Step 6: Calculate returns & PnL using prices and positions

        if method == 'L':
            pnl_mt_df = (relative_returns_df * generic_curves_df.loc[cob_date] *
                         instrument_ref_dict[instrument]['conversion'] * position)
        elif method == 'NL-S':
            pass
        elif method == 'NL-R':
            pass

        # Step 5: VaR calculation
        var_95 = calculate_var(cob_date, pnl_mt_df, -pnl_mt_df, 95, 260)
        var_99 = calculate_var(cob_date, pnl_mt_df, -pnl_mt_df, 99, 260)

        return {"price_series": generic_curves_df, "returns": relative_returns_df, "PnL": pnl_mt_df,
                "VaR_95": var_95, "VaR_99": var_99}
