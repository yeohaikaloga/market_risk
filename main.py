from db.db_connection import get_engine
from sqlalchemy import text
import pandas as pd
from price.futures_price import FuturesPrice
from ticker.futures_ticker import FuturesTicker

if __name__ == '__main__':
    prod_engine = get_engine('prod')

    START_DATE = '2024-03-11'
    END_DATE = '2024-12-31'

    tickers_list = ['CT']
    for ticker in tickers_list:
        futures_ticker = FuturesTicker(instrument_id=ticker, source=prod_engine)
        futures_ticker.load_ref_data()
        active_contracts = futures_ticker.load_active_contracts(start_date=START_DATE, end_date=END_DATE)
        active_expiry_dates = futures_ticker.load_active_expiry_dates(start_date=START_DATE, end_date=END_DATE)
        print(active_expiry_dates)
        futures_price = FuturesPrice(instrument_id=futures_ticker.instrument_id, source=prod_engine)
        price_df = futures_price.load_prices(start_date=START_DATE, end_date=END_DATE)
        print(price_df.head(10))

        generic_curve = generate_generic_curve(price_df, 1, 14)
        print(generic_curve.head())
