from price_series_loader.price_series_loader import PriceLoader
from utils.contract_utils import custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text


class DerivativesPriceLoader(PriceLoader):

    def __init__(self, instrument_name, mode, source):
        super().__init__(instrument_name, source)
        self.price_history = None
        self.instrument_id = instrument_name
        self.contracts: list[str] | None = None
        valid_modes = {'futures', 'options'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        self.mode = mode
        self.source = source


    def load_prices(self, start_date, end_date, contracts=None, reindex_dates=None, instrument_name=None):
        self.contracts = contracts or self.contracts

        if not self.contracts:
            print("No contracts loaded.")
            return pd.DataFrame()

        contracts_formatted = "(" + ",".join(f"'{contract}'" for contract in self.contracts) + ")"
        print('formatted:', contracts_formatted)

        # Use mp.px_settle_last_dt instead of tdate so that prices are not rolled over for market holidays
        query = f"""
                SELECT mp.px_settle_last_dt::date as date, dc.unique_id_fut_opt, dc.ticker, mp.px_settle
                FROM ref.derivatives_contract dc
                JOIN market.market_price mp
                  ON dc.traded_contract_id = mp.traded_contract_id
                JOIN ref.session_type st 
                  ON dc.session_type_id = st.id
                WHERE dc.unique_id_fut_opt IN {contracts_formatted}
                AND mp.px_settle_last_dt BETWEEN '{start_date}' AND '{end_date}'
                AND mp.px_settle_last_dt <= dc.last_tradeable_dt
            """
        if self.mode == 'futures':
            query += " AND mp.px_settle_last_dt >= dc.fut_first_trade_dt"
        elif self.mode == 'options':
            query += " AND mp.px_settle_last_dt >= dc.opt_first_trade_dt"
        else:
            raise ValueError("Mode not correctly specified. 'futures' only 'options' only.")

        if instrument_name == 'CT':
            query += " AND dc.feed_source = 'eNYB'"
        #print(query)

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if df.empty:
            print("No price_series_loader data found for given parameters.")
            return pd.DataFrame()

        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by contract_ref_loader and date, take last px_settle for duplicates and unstack tickers as columns
        price_df = df.groupby(['unique_id_fut_opt', 'date'])['px_settle'].last().unstack(level=0)
        price_df.columns = [col.replace(' COMB','').replace(' Comdty', '') for col in price_df.columns]
        # Sort columns using your custom monthly contract_ref_loader sort key
        sorted_columns = sorted(price_df.columns,
                                key=lambda ticker: custom_monthly_contract_sort_key(contract=ticker))
        price_df = price_df[sorted_columns]

        # Reindex dates
        if reindex_dates is not None:
            price_df = price_df.reindex(reindex_dates)

        # Optional: reindex to include full date range, fill missing dates with NaN; note that reindex will reorder
        # the index in ascending order by default full_dates = pd.date_range(start=start_date, end=end_date) price_df
        # = price_df.reindex(full_dates)

        self.price_history = price_df
        return price_df
