from .price import Price  # assuming Price is in price.py
from contract.futures_contract import custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text


class FuturesPrice(Price):  # Futures inherits from Price

    def __init__(self, instrument_id, source):
        super().__init__(instrument_id, source)
        self.instrument_id = instrument_id
        self.source = source
        self.contracts: list[str] | None = None

    def load_prices(self, start_date, end_date, selected_contracts=None, reindex_dates=None, instrument_id=None):
        self.contracts = selected_contracts or self.contracts

        if not self.contracts:
            print("No contracts loaded.")
            return pd.DataFrame()

        contracts_formatted = "(" + ",".join(f"'{contract}'" for contract in self.contracts) + ")"
        print(contracts_formatted)

        query = f"""
            SELECT mp.tdate::date as tdate, dc.ticker, mp.px_settle
            FROM ref.derivatives_contract dc
            JOIN market.market_price mp
            ON dc.traded_contract_id = mp.traded_contract_id
            JOIN ref.session_type st 
            ON dc.session_type_id = st.id
            WHERE dc.ticker IN {contracts_formatted}
            AND mp.tdate BETWEEN '{start_date}' AND '{end_date}'
        """
        if instrument_id == 'CT':
            query += " AND dc.feed_source = 'eNYB'"

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if df.empty:
            print("No price data found for given parameters.")
            return pd.DataFrame()

        # Convert to datetime
        df['tdate'] = pd.to_datetime(df['tdate'])

        # Group by contract and date, take last px_settle for duplicates and unstack tickers as columns
        price_df = df.groupby(['ticker', 'tdate'])['px_settle'].last().unstack(level=0)

        # Sort columns using your custom monthly contract sort key
        sorted_columns = sorted(price_df.columns, key=custom_monthly_contract_sort_key)
        price_df = price_df[sorted_columns]

        # Reindex dates
        if reindex_dates is not None:
            price_df = price_df.reindex(reindex_dates)

        # Optional: reindex to include full date range, fill missing dates with NaN; note that reindex will reorder
        # the index in ascending order by default full_dates = pd.date_range(start=start_date, end=end_date) price_df
        # = price_df.reindex(full_dates)

        self.price_history = price_df
        return price_df
