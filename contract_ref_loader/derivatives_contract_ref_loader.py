import pandas as pd
from sqlalchemy import text
from contract_ref_loader.contract_ref_loader import ContractRefLoader
from utils.contract_utils import month_codes, instrument_ref_dict
from utils.contract_utils import custom_monthly_contract_sort_key


class DerivativesContractRefLoader(ContractRefLoader):


    def __init__(self, instrument_name, source, params=None):
        super().__init__(instrument_name, source, params)
        self.currency = None
        self.futures_category = None
        self._df_tickers = None

    def _build_ticker_regex(self, mode):
        month_letters = ''.join(month_codes.keys())
        base_pattern = f"^{self.instrument_name}[{month_letters}][0-9]"
        suffix_pattern = r"(?: COMB)? Comdty$"

        if mode == 'futures':
            pattern = base_pattern + suffix_pattern           # e.g. CTK4
        elif mode == 'options':
            pattern = base_pattern + r"[A-Z] \d+" + suffix_pattern     # e.g. CTK4P
        elif mode == 'spreads':
            pattern = (f"^{self.instrument_name}[{month_letters}][0-9]"
                       f"{self.instrument_name}[{month_letters}][0-9]" + suffix_pattern)
        else:
            pattern = base_pattern + suffix_pattern    # futures or options, e.g. CTK4 or CTK4P

        print(f"Regex pattern: {pattern}")
        return pattern

    def load_tickers(self, mode, contracts=None, relevant_months=None, relevant_years=None,
                     relevant_options=None, relevant_strikes=None) -> pd.DataFrame:
    #TODO Rewrite query to use company id.
        def normalize_list(x):
            if isinstance(x, (str, int)):
                return [str(x)]
            elif x is not None:
                return [str(i) for i in x]
            return None

        relevant_months = normalize_list(relevant_months)
        relevant_years = normalize_list(relevant_years)
        relevant_options = normalize_list(relevant_options)
        relevant_strikes = normalize_list(relevant_strikes)

        futures_category = instrument_ref_dict.get(self.instrument_name, {}).get('futures_category')
        if not futures_category:
            print(f"Instrument '{self.instrument_name}' not found in reference dictionary.")
            return pd.DataFrame()

        if mode == 'futures':
            if relevant_options is not None or relevant_strikes is not None:
                raise ValueError("Options and strikes should be None when mode is 'futures'")

        month_letters = ''.join(month_codes.keys())
        regex_comb = f"^{self.instrument_name}[{month_letters}][0-9] COMB Comdty$"
        regex_non_comb = f"^{self.instrument_name}[{month_letters}][0-9] Comdty$"

        query = f"""
            WITH base AS (
                SELECT dc.*, i.name AS instrument_name, tc.type, c.currency
                FROM ref.derivatives_contract dc
                LEFT JOIN ref.currency c ON c.id = dc.currency_id 
                LEFT JOIN ref.instrument i ON i.id = dc.instrument_id
                LEFT JOIN ref.traded_contract tc ON tc.id = dc.traded_contract_id
                WHERE dc.futures_category = '{futures_category}'),
            comb AS (
                SELECT DISTINCT unique_id_fut_opt FROM base
                WHERE unique_id_fut_opt ~ '{regex_comb}'),
            final AS (
                SELECT * FROM base
                WHERE unique_id_fut_opt ~ '{regex_comb}'
                UNION
                SELECT * FROM base
                WHERE unique_id_fut_opt ~ '{regex_non_comb}'
                AND REGEXP_REPLACE(unique_id_fut_opt, ' Comdty$', '') || ' COMB Comdty' NOT IN (
                    SELECT unique_id_fut_opt FROM comb))
            SELECT DISTINCT instrument_name, type, unique_id_fut_opt, ticker, opt_exer_typ, futures_category, 
                            currency, real_underlying_ticker, fut_first_trade_dt, opt_first_trade_dt, 
                            last_tradeable_dt, opt_expire_dt
            FROM final
        """

        # Conditional filters
        conditions = []

        if contracts:
            tickers_list = "', '".join(contracts)
            conditions.append(f"unique_id_fut_opt IN ('{tickers_list}')")

        if relevant_months:
            month_filter = "', '".join(relevant_months)
            conditions.append(f"SUBSTRING(ticker, {len(self.instrument_name) + 1}, 1) IN ('{month_filter}')")

        if relevant_years:
            year_filter = "', '".join(relevant_years)
            conditions.append(f"SUBSTRING(ticker, {len(self.instrument_name) + 2}, 1) IN ('{year_filter}')")

        if mode == 'options':
            if relevant_options:
                options_filter = "', '".join(relevant_options)
                conditions.append(f"LENGTH(ticker) = {len(self.instrument_name) + 3}")
                conditions.append(f"RIGHT(ticker, 1) IN ('{options_filter}')")

            if relevant_strikes:
                strike_filter = "', '".join(relevant_strikes)
                conditions.append(f"SPLIT_PART(unique_id_fut_opt, ' ', 2) IN ('{strike_filter}')")

        # Append all conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY unique_id_fut_opt"
        #print(query)

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if not df.empty:
            self.currency = df.iloc[0]['currency']
            self.futures_category = df.iloc[0]['futures_category']

        self._df_tickers = df
        print(df.head())
        return df

    def load_contracts(self, mode, **kwargs) -> list:
        df = self.load_tickers(mode, **kwargs)
        if df.empty:
            return []
        return sorted(df['unique_id_fut_opt'].tolist(),
                      key=lambda contract: custom_monthly_contract_sort_key(contract=contract))

    def load_underlying_futures(self, **kwargs) -> dict:
        df = self.load_tickers(mode='options', **kwargs)
        print('load underlying futures for options')
        return dict(zip(df['unique_id_fut_opt'], df['real_underlying_ticker']))

    def load_underlying_futures_expiry_dates(self, mode, **kwargs) -> dict:
        df = self.load_tickers(mode, **kwargs)
        print('load underlying futures expiry dates')

        if mode == 'futures':
            expiry_date_col = 'last_tradeable_dt'
        elif mode == 'options':
            expiry_date_col = 'opt_undl_exp_dt'
        else:
            raise ValueError("Invalid mode. Must be 'futures' or 'options'.")

        if df.empty or expiry_date_col not in df.columns:
            print("Warning: no data or missing expected column.")
            return {}

        return dict(zip(df['unique_id_fut_opt'], pd.to_datetime(df[expiry_date_col], errors='coerce')))

    def load_options_expiry_dates(self, **kwargs) -> dict:
        df = self.load_tickers(mode='options', **kwargs)
        print('load underlying options expiry dates')
        return dict(zip(df['unique_id_fut_opt'], pd.to_datetime(df['opt_expire_dt'], errors='coerce')))

    def load_underlying_futures_start_dates(self, mode, **kwargs) -> dict:
        df = self.load_tickers(mode, **kwargs)
        print('load underlying futures start dates')

        if mode == 'futures':
            start_date_col = 'fut_first_trade_dt'
        else:
            raise ValueError("Invalid mode. Must be 'futures' only.")

        if df.empty or start_date_col not in df.columns:
            print("Warning: no data or missing expected column.")
            return {}

        return dict(zip(df['unique_id_fut_opt'], pd.to_datetime(df[start_date_col], errors='coerce')))

    def load_options_start_dates(self, **kwargs) -> dict:
        df = self.load_tickers(mode='options', **kwargs)
        print('load underlying options start dates')
        return dict(zip(df['unique_id_fut_opt'], pd.to_datetime(df['opt_first_trade_dt'], errors='coerce')))

    def load_ref_data(self, mode) -> pd.DataFrame:
        if self.instrument_name not in instrument_ref_dict:
            raise ValueError(f"Instrument '{self.instrument_name}' not found in reference dictionary.")

        futures_category = instrument_ref_dict[self.instrument_name]['futures_category']
        month_letters = ''.join(month_codes.keys())
        regex_ticker_base = f"^{self.instrument_name}[{month_letters}][0-9]"

        if mode == 'futures':
            ticker_regex = regex_ticker_base + "$"  # Only futures, exactly 4 chars (e.g., CTH4)
        elif mode == 'options':
            ticker_regex = regex_ticker_base + "[CP]$"  # Only options, exactly 5 chars (e.g., CTH4P)
        else:
            raise ValueError("Mode not correctly specified. 'futures' only 'options' only.")

        query = f"""
        SELECT DISTINCT c.currency, dc.futures_category
        FROM ref.derivatives_contract dc
        JOIN ref.currency c
        ON c.id = dc.currency_id 
        WHERE dc.ticker ~ '{ticker_regex}'
        AND futures_category = '{futures_category}'
        """
        # print(query)

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
            print(df)

            if df.empty:
                raise ValueError(f"No reference data found for instrument '{self.instrument_name}'")

            if len(df) > 1:
                raise ValueError(
                    f"Multiple reference entries found for instrument '{self.instrument_name}', expected one.")

            row = df.iloc[0]
            self.futures_category = row['futures_category']
            self.currency = row['currency']

        print(f"Loaded ref data for {self.instrument_name}: {self.currency}, {self.futures_category}")
        return df
