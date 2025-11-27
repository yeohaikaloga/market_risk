from itertools import product
import pandas as pd
from sqlalchemy import text

from price_series_loader.price_series_loader import PriceLoader
from contract_ref_loader.forex_ref_loader import ForexRefLoader

class ForexPriceLoader:
    """
    FX price loader that uses metadata from ForexRefLoader to apply filters automatically.
    """

    def __init__(self, source, ref_loader: ForexRefLoader = None):
        self.source = source
        self.ref_loader = ref_loader
        self.price_history = None

    def load_prices(
            self,
            start_date: str,
            end_date: str,
            reindex_dates: pd.DatetimeIndex = None,
            type: str | None = None,
    ) -> pd.DataFrame:

        if self.ref_loader is None:
            raise ValueError("Must provide a ForexRefLoader instance with loaded metadata")

        # Pull filters from ref_loader
        base_ccy = self.ref_loader._last_base_ccy
        quote_ccy = self.ref_loader._last_quote_ccy
        fwd_mon = self.ref_loader._last_fwd_mon
        type_filter = type or self.ref_loader._last_type

        if base_ccy is None or quote_ccy is None:
            raise ValueError("ForexRefLoader must have metadata loaded first")

        # Convert to lists
        if isinstance(base_ccy, str):
            base_ccy = [base_ccy.upper()]
        else:
            base_ccy = [b.upper() for b in base_ccy]

        if isinstance(quote_ccy, str):
            quote_ccy = [quote_ccy.upper()]
        else:
            quote_ccy = [q.upper() for q in quote_ccy]

        cur_pairs = [f"{b}{q}" for b, q in product(base_ccy, quote_ccy)]

        # SQL
        params = {"pairs": tuple(cur_pairs)}
        query = """
        SELECT
            mkt_date::date AS date,
            cur_pair AS fx_ticker,
            last_price AS fx_rate
        FROM staging.vw_combined_fx_data
        WHERE cur_pair IN :pairs
        """

        if fwd_mon is not None:
            if str(fwd_mon).upper() == "NULL":
                query += " AND fwd_mon IS NULL"
            else:
                query += " AND fwd_mon = :fwd_mon"
                params["fwd_mon"] = fwd_mon

        if type_filter:
            query += " AND type = :type"
            params["type"] = type_filter

        query += " AND mkt_date BETWEEN :start AND :end"
        params["start"] = start_date
        params["end"] = end_date

        query += " ORDER BY mkt_date, cur_pair"

        expanded_query = query
        for key, value in params.items():
            if isinstance(value, tuple):
                replacement = "(" + ", ".join(f"'{v}'" for v in value) + ")"
            else:
                replacement = f"'{value}'"
            expanded_query = expanded_query.replace(f":{key}", replacement)

        print(expanded_query)

        # Execute query safely using bound parameters
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)

        if df.empty:
            print(f"[ForexPriceLoader] No FX data found for pairs {cur_pairs} "
                  f"between {start_date} and {end_date}")
            return pd.DataFrame()

        # Pivot
        df["date"] = pd.to_datetime(df["date"])
        price_df = df.pivot_table(
            index="date",
            columns="fx_ticker",
            values="fx_rate",
            aggfunc="last"
        ).sort_index(axis=1)

        if reindex_dates is not None:
            price_df = price_df.reindex(reindex_dates)

        price_df = price_df.ffill()

        self.price_history = price_df
        return price_df

