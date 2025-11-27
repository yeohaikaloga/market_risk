import pandas as pd
from itertools import product
from sqlalchemy import text
from contract_ref_loader.contract_ref_loader import ContractRefLoader
from utils.log_utils import get_logger

class ForexRefLoader(ContractRefLoader):
    """
    FX metadata loader that inherits from ContractRefLoader.
    Loads: base_ccy, quote_ccy, tenor, fixing type, points_rate, bbr_code.
    Stores last-used filters for use in ForexPriceLoader.
    """

    def __init__(self, source, params=None):
        super().__init__(instrument_name="FOREX", source=source, params=params)
        # store last-used filters
        self._last_base_ccy = None
        self._last_quote_ccy = None
        self._last_fwd_mon = None
        self._last_type = None

    def load_metadata(
        self,
        base_ccy: str | list = None,
        quote_ccy: str | list = None,
        fwd_mon: str | None = None,  # None=no filter, 'NULL'=filter blanks
        type: str | None = None
    ) -> pd.DataFrame:
        if base_ccy is None or quote_ccy is None:
            raise ValueError("Both base_ccy and quote_ccy must be provided")

        # store filters
        self._last_base_ccy = base_ccy
        self._last_quote_ccy = quote_ccy
        self._last_fwd_mon = fwd_mon
        self._last_type = type

        # Convert to lists if single strings
        if isinstance(base_ccy, str):
            base_ccy = [base_ccy.upper()]
        else:
            base_ccy = [b.upper() for b in base_ccy]

        if isinstance(quote_ccy, str):
            quote_ccy = [quote_ccy.upper()]
        else:
            quote_ccy = [q.upper() for q in quote_ccy]

        # Generate all base-quote combinations
        cur_pairs = [f"{b}{q}" for b, q in product(base_ccy, quote_ccy)]
        bbr_codes = tuple(f"{cp} Curncy" for cp in cur_pairs)
        params = {"pairs": tuple(cur_pairs), "bbr_codes": bbr_codes}

        query = """
        SELECT
            cur_pair,
            fwd_mon,
            type,
            fixing,
            points_rate,
            bbr_code
        FROM staging.ors_bbr_series_am_ex
        WHERE cur_pair IN :pairs
          AND bbr_code IN :bbr_codes
        """

        # Apply fwd_mon filter
        if fwd_mon is not None:
            if str(fwd_mon).upper() == "NULL":
                query += " AND fwd_mon IS NULL"
            else:
                query += " AND fwd_mon = :fwd_mon"
                params["fwd_mon"] = fwd_mon

        # Apply type filter
        if type:
            query += " AND type = :type"
            params["type"] = type

        # Sorting
        query += """
        ORDER BY 
            CASE WHEN fwd_mon ~ '^[0-9]+[MDY]$' THEN 1 ELSE 2 END,
            fwd_mon
        """

        # Execute
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)

        if df.empty:
            raise ValueError(f"[ForexRefLoader] No metadata found for FX pairs: {cur_pairs}")

        # Ensure base/quote columns exist
        if "base_ccy" not in df.columns or "quote_ccy" not in df.columns:
            df["base_ccy"] = df["cur_pair"].str[:3]
            df["quote_ccy"] = df["cur_pair"].str[3:]

        # Drop duplicates
        df = df.drop_duplicates(subset=["cur_pair", "fwd_mon", "type", "fixing", "points_rate", "bbr_code"])
        return df