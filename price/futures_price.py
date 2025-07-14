from .price import Price  # assuming Price is in price.py
from ticker.futures_ticker import custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text


class FuturesPrice(Price):  # Futures inherits from Price
    def load_prices(self, start_date, end_date, active_tickers=None):
        # Assume self.active_tickers is already populated list of tickers (e.g. ['CTH5', 'CTN5', ...])
        if not self.active_tickers:
            print("No active tickers loaded.")
            return pd.DataFrame()

        # Format the tickers list for SQL IN clause: ('CTH5', 'CTN5', ...)
        tickers_formatted = "(" + ",".join(f"'{ticker}'" for ticker in self.active_tickers) + ")"

        query = f"""
            SELECT mp.tdate::date as tdate, dc.ticker, mp.px_settle
            FROM ref.derivatives_contract dc
            JOIN market.market_price mp
            ON dc.traded_contract_id = mp.traded_contract_id
            WHERE dc.ticker IN {tickers_formatted}
            AND mp.tdate BETWEEN '{start_date}' AND '{end_date}'
        """

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if df.empty:
            print("No price data found for given parameters.")
            return pd.DataFrame()

        # Convert to datetime
        df['tdate'] = pd.to_datetime(df['tdate'])

        # Group by ticker and date, take last px_settle for duplicates and unstack tickers as columns
        price_df = df.groupby(['ticker', 'tdate'])['px_settle'].last().unstack(level=0)
        print(price_df.head())

        # Sort columns using your custom monthly contract sort key
        sorted_columns = sorted(price_df.columns, key=custom_monthly_contract_sort_key)
        price_df = price_df[sorted_columns]

        # Sort index (dates) descending
        price_df = price_df.sort_index(ascending=False)

        # Optional: reindex to include full date range, fill missing dates with NaN
        full_dates = pd.date_range(start=start_date, end=end_date)
        price_df = price_df.reindex(full_dates)

        self.price_history = price_df
        return price_df

    def construct_curve(self):
        # Example stub, implement your curve construction here
        print("Constructing price curve for futures...")
