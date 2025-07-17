import pandas as pd
from sqlalchemy import text
from contract.contract import Contract  # Adjust if your import path differs

tickers_ref_dict = {'CT': {'futures_category': 'Fibers'},
                        'VV': {'futures_category': 'Fibers'}}
class FuturesContract(Contract):
    def load_ref_data(self):

        instrument_key = self.instrument_id
        instrument_length = len(instrument_key)

        if instrument_key not in tickers_ref_dict:
            print(f"Instrument '{instrument_key}' not found in reference dictionary.")
            return pd.DataFrame()

        futures_category = tickers_ref_dict[instrument_key]['futures_category']
        query = f"""
        SELECT DISTINCT currency_id, ticker, futures_category
        FROM ref.derivatives_contract
        WHERE ticker LIKE '{instrument_key}%' 
        AND futures_category = '{futures_category}' 
        AND LENGTH(ticker) = {instrument_length + 2}
        LIMIT 5
        """
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
            print(df.head())
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

    def _load_contract_data(self) -> pd.DataFrame:
        """
        Internal method to fetch contract metadata (contract + expiry date).
        """
        query = f"""
            SELECT DISTINCT dc.ticker, dc.last_tradeable_dt
            FROM ref.derivatives_contract dc
            JOIN market.market_price mp 
                ON dc.traded_contract_id = mp.traded_contract_id
            WHERE dc.ticker LIKE '{self.instrument_id}%'
            AND LENGTH(dc.ticker) = {len(self.instrument_id) + 2}
        """
        if self.instrument_id == 'CT':
            query += " AND dc.feed_source = 'eNYB'"

        query += " ORDER BY dc.ticker"

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        df['last_tradeable_dt'] = pd.to_datetime(df['last_tradeable_dt'], errors='coerce')
        return df
    def load_contracts(self) -> list:
        """
        Public method to load list of contract tickers.
        """
        df = self._load_contract_data()
        self.contracts = df['ticker'].dropna().tolist()
        print(f"Loaded contracts for {self.instrument_id}: {self.contracts}")
        return self.contracts

    def load_expiry_dates(self) -> dict:
        """
        Public method to load contract expiry dates as a dictionary.
        """
        df = self._load_contract_data()
        self.expiry_dates = dict(zip(df['ticker'], df['last_tradeable_dt']))
        print(f"Loaded expiry dates for {self.instrument_id}:")
        for contract, expiry in self.expiry_dates.items():
            print(f"  {contract}: {expiry}")
        return self.expiry_dates

def custom_monthly_contract_sort_key(futures_ticker):
    month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    ticker_length = len(futures_ticker)
    year_digit = int(futures_ticker[ticker_length - 1])
    month_char = futures_ticker[ticker_length - 2]
    year = 2000 + year_digit
    month = month_codes.get(month_char, 0)
    return (year, month)


