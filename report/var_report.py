import pandas as pd
import numpy as np
from contract.futures_contract import FuturesContract
from contract.futures_contract import tickers_ref_dict
from loaded_price_series.loaded_futures_price import LoadedFuturesPrice
from generated_price_series.generic_curve import GenericCurveGenerator
from financial_calculations.returns import relative_returns
from financial_calculations.VaR import calculate_var


def generate_var_report(instruments_list: list, cob_date: str, days_list: list[str], position: list[float],
                        prod_engine) -> dict:
    # NOT COMPLETED YET
    for instrument in instruments_list:
        # Step 1: Load contract metadata
        futures_contract = FuturesContract(instrument_id=instrument, source=prod_engine)
        futures_contract.load_ref_data()
        contracts = futures_contract.load_contracts()
        futures_expiry_dates = futures_contract.load_expiry_dates()

        # Step 2: Load loaded_price_series data for these contracts
        futures_price = LoadedFuturesPrice(instrument_id=futures_contract.instrument_id, source=prod_engine)
        if instrument == 'CT':
            # active_contracts = [c for c in active_contracts if c[-2] in {'H', 'K', 'N', 'V', 'Z'}]
            # active_contracts = [c for c in active_contracts if c[-2] in {'H', 'K', 'N', 'Z'}]
            selected_contracts = ['CTV4', 'CTZ4', 'CTH5', 'CTK5', 'CTN5', 'CTV5', 'CTZ5',
                                  'CTH6']  # to remove V contracts in future
        else:
            selected_contracts = contracts
        price_df = futures_price.load_prices(start_date=days_list[0],
                                             end_date=cob_date,
                                             selected_contracts=selected_contracts,
                                             reindex_dates=days_list,
                                             instrument_id=instrument)

        # Step 3: Generate generic loaded_price_series series for linear PnL vectors
        price_series = GenericCurveGenerator(price_df, futures_contract=futures_contract)
        generic_curves_df = price_series.generate_generic_curves_df_up_to(max_position=2,
                                                                          roll_days=14,
                                                                          adjustment='ratio',
                                                                          label_prefix=instrument)
        print(generic_curves_df)
        generic_curves_df = generic_curves_df.replace({pd.NA: np.nan}, inplace=False).astype(float).round(
            3)  # .astype(float).round(3) is limitation of BBG BDH formula -> to remove in future

        # Step 4: Calculate returns & PnL using prices and positions
        relative_returns_df = relative_returns(generic_curves_df)
        pnl_mt_df = relative_returns_df * generic_curves_df.loc[cob_date] * tickers_ref_dict[instrument][
            'conversion'] * position

        # Step 5: VaR calculation
        var_95 = calculate_var(cob_date, pnl_mt_df, -pnl_mt_df, 95, 260)
        var_99 = calculate_var(cob_date, pnl_mt_df, -pnl_mt_df, 99, 260)

        return {"price_series": generic_curves_df, "returns": relative_returns_df, "PnL": pnl_mt_df,
        "VaR_95": var_95, "VaR_99": var_99}
