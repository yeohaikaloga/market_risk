import pandas as pd
from sqlalchemy import text
from contract_ref_loader.contract import Contract

instrument_ref_dict = {'CT': {'futures_category': 'Fibers', 'conversion': 22.046},  # clb to $/MT
                    'VV': {'futures_category': 'Fibers'}, 'conversion': 1}  # yet to bring in forex


class FuturesContract(Contract):
    def load_ref_data(self):

        instrument_key = self.instrument_id
        instrument_length = len(instrument_key)

        if instrument_key not in instrument_ref_dict:
            print(f"Instrument '{instrument_key}' not found in reference dictionary.")
            return pd.DataFrame()

        futures_category = instrument_ref_dict[instrument_key]['futures_category']
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
                self.unit = row.get('unit', None)
                self.lot_size = row.get('lot_size', None)
                self.exchange = row.get('exchange', None)
                print(f"Loaded ref data for {self.instrument_id}: {self.currency_id}, {self.futures_cat} ,{self.unit}, "
                      f"{self.lot_size}, {self.exchange}")
            else:
                print(f"No reference data found for instrument {self.instrument_id}")

    def _load_contract_data(self, relevant_months: set = None) -> pd.DataFrame:
        """
        Internal method to fetch contract_ref_loader metadata (contract_ref_loader + expiry date),
        optionally filtering for specific contract_ref_loader months.
        """
        query = f"""
            SELECT DISTINCT dc.ticker, dc.last_tradeable_dt, dc.fut_first_trade_dt
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
        df['fut_first_trade_dt'] = pd.to_datetime(df['fut_first_trade_dt'], errors='coerce')

        if relevant_months:
            df = df[df['ticker'].str[-2].isin(relevant_months)]

        return df
    def load_contracts(self, relevant_months=None) -> list:
        """
        Public method to load list of contract_ref_loader tickers.
        """
        df = self._load_contract_data(relevant_months=relevant_months)
        self.contracts = sorted(df['ticker'].dropna().tolist(), key=custom_monthly_contract_sort_key
        )
        print(f"Loaded contracts for {self.instrument_id}: {self.contracts}")
        return self.contracts

    def load_expiry_dates(self, relevant_months=None) -> dict:
        """
        Public method to load contract_ref_loader expiry dates as a dictionary.
        """
        df = self._load_contract_data(relevant_months=relevant_months)
        self.expiry_dates = dict(zip(df['ticker'], df['last_tradeable_dt']))
        print(f"Loaded expiry dates for {self.instrument_id}:")
        for contract, expiry in self.expiry_dates.items():
            print(f"  {contract}: {expiry}")
        return self.expiry_dates

    def load_start_dates(self, relevant_months=None) -> dict:
        """
        Public method to load contract_ref_loader expiry dates as a dictionary.
        """
        df = self._load_contract_data(relevant_months=relevant_months)
        self.start_dates = dict(zip(df['ticker'], df['fut_first_trade_dt']))
        print(f"Loaded start dates for {self.instrument_id}:")
        for contract, start in self.start_dates.items():
            print(f"  {contract}: {start}")
        return self.start_dates


def custom_monthly_contract_sort_key(futures_ticker):
    month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    ticker_length = len(futures_ticker)
    year_digit = int(futures_ticker[ticker_length - 1])
    month_char = futures_ticker[ticker_length - 2]
    year = 2000 + year_digit
    month = month_codes.get(month_char, 0)
    return year, month
