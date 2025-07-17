from db.db_connection import get_engine
from sqlalchemy import text
import pandas as pd
from contract.futures_contract import FuturesContract
from price.futures_price import FuturesPrice
from generated_price_series.generic_curve import GenericCurveGenerator
from utils.date_utils import no_of_days_list

if __name__ == '__main__':
    prod_engine = get_engine('prod')

    COB_DATE = '2025-07-10'
    days_260_list = no_of_days_list(COB_DATE, 260)
    START_DATE = days_260_list[0]


    instruments_list = ['CT', 'VV']

    for instrument in instruments_list:
        # Step 1: Load contract metadata
        futures_contract = FuturesContract(instrument_id=instrument, source=prod_engine)
        futures_contract.load_ref_data()
        contracts = futures_contract.load_contracts()
        futures_expiry_dates = futures_contract.load_expiry_dates()
        print('test', contracts)
        print(futures_expiry_dates)

        # Step 2: Load price data for these contracts
        futures_price = FuturesPrice(instrument_id=futures_contract.instrument_id, source=prod_engine)
        if instrument == 'CT':
            #active_contracts = [c for c in active_contracts if c[-2] in {'H', 'K', 'N', 'V', 'Z'}]
            #active_contracts = [c for c in active_contracts if c[-2] in {'H', 'K', 'N', 'Z'}]
            selected_contracts = ['CTV4', 'CTZ4', 'CTH5', 'CTK5', 'CTN5', 'CTV5', 'CTZ5', 'CTH6']
            price_df = futures_price.load_prices(start_date=START_DATE,
                                                 end_date=COB_DATE,
                                                 selected_contracts=selected_contracts,
                                                 reindex_dates=days_260_list,
                                                 instrument_id=instrument)
        else:
            price_df = futures_price.load_prices(start_date=START_DATE,
                                                 end_date=COB_DATE,
                                                 selected_contracts=contracts,
                                                 reindex_dates=days_260_list,
                                                 instrument_id=instrument)
        print(price_df.head())
        print(price_df.tail())

        price_series = GenericCurveGenerator(price_df, futures_contract=futures_contract)
        generic_curve_1 = price_series.generate_generic_curve(position=1, roll_days=14, adjustment='ratio')
        generic_curve_2 = price_series.generate_generic_curve(position=2, roll_days=14, adjustment='ratio')
        generic_curve_3 = price_series.generate_generic_curve(position=3, roll_days=14, adjustment='ratio')
        print(generic_curve_1.sort_index(ascending=False).head(30))
        print(generic_curve_2.sort_index(ascending=False).head(30))
        print(generic_curve_3.sort_index(ascending=False).head(30))
