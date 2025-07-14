import pandas as pd
from sqlalchemy import text
from ticker.ticker import Ticker  # Adjust if your import path differs


class FuturesTicker(Ticker):
    def load_ref_data(self):
        # STILL NEEDS TO BE CLEANED UP ###
        query = f"""
        SELECT DISTINCT currency_id, ticker, futures_category
        FROM ref.derivatives_contract
        WHERE ticker LIKE '{self.instrument_id}%'
        LIMIT 5
        """
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
            if not df.empty:
                row = df.iloc[0]
                self.currency_id = row['currency_id']
                self.futures_cat = row['futures_category']
                # self.unit = row['unit']
                # self.lot_size = row['lot_size']
                # self.exchange = row['exchange']
                print(f"Loaded ref data for {self.instrument_id}: {self.currency_id}, {self.futures_cat} ,{self.unit}, "
                      f"{self.lot_size}, {self.exchange}")
            else:
                print(f"No reference data found for instrument {self.instrument_id}")

    def load_active_contracts(self, start_date, end_date):
        query = f"""
        SELECT DISTINCT dc.ticker
        FROM ref.derivatives_contract dc
        JOIN market.market_price mp 
            ON dc.traded_contract_id = mp.traded_contract_id
        WHERE dc.ticker LIKE '{self.instrument_id}%'
        AND LENGTH(dc.ticker) = {len(self.instrument_id) + 2}
        AND mp.tdate BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY dc.ticker
        """
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
            self.active_contracts = df['ticker'].tolist()
            print(f"Loaded active contracts for {self.instrument_id} from {start_date} to {end_date}: "
                  f"{self.active_contracts}")
            return self.active_contracts

    def load_active_expiry_dates(self, start_date, end_date):
        query = f"""
                SELECT DISTINCT dc.ticker, dc.last_tradeable_dt
                FROM ref.derivatives_contract dc
                JOIN market.market_price mp 
                    ON dc.traded_contract_id = mp.traded_contract_id
                WHERE dc.ticker LIKE '{self.instrument_id}%'
                AND LENGTH(dc.ticker) = {len(self.instrument_id) + 2}
                AND mp.tdate BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY dc.ticker
                """
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
            self.active_expiry_dates = df['last_tradeable_dt'].tolist()
            print(
                f"Loaded active expiry dates for {self.instrument_id} from {start_date} to {end_date}: "
                f"{self.active_expiry_dates}")
            return self.active_expiry_dates

def custom_monthly_contract_sort_key(futures_ticker):
    ticker_length = len(futures_ticker)
    year_digit = int(futures_ticker[ticker_length - 1])
    month_char = futures_ticker[ticker_length - 2]
    return year_digit, month_char
