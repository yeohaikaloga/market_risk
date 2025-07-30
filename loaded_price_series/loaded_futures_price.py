from loaded_price_series.loaded_price_series import LoadedPrice
from contract.futures_contract import custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text


class LoadedFuturesPrice(LoadedPrice):

    def __init__(self, instrument_id, source):
        super().__init__(instrument_id, source)
        self.price_history = None
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

        # Only include prices from start date and up to each contract's expiry to avoid spurious post-expiry data from BBG
        # Use mp.px_settle_last_dt instead of tdate so that prices are not rolled over for market holidays
        query = f"""
            SELECT mp.px_settle_last_dt::date as date, dc.ticker, mp.px_settle
            FROM ref.derivatives_contract dc
            JOIN market.market_price mp
            ON dc.traded_contract_id = mp.traded_contract_id
            JOIN ref.session_type st 
            ON dc.session_type_id = st.id
            WHERE dc.ticker IN {contracts_formatted}
            AND mp.px_settle_last_dt BETWEEN '{start_date}' AND '{end_date}'
            AND mp.px_settle_last_dt <= dc.last_tradeable_dt
            AND mp.px_settle_last_dt >= dc.fut_first_trade_dt
        """
        if instrument_id == 'CT':
            query += " AND dc.feed_source = 'eNYB'"

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if df.empty:
            print("No loaded_price_series data found for given parameters.")
            return pd.DataFrame()

        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by contract and date, take last px_settle for duplicates and unstack tickers as columns
        price_df = df.groupby(['ticker', 'date'])['px_settle'].last().unstack(level=0)

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
