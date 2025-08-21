import pandas as pd
from sqlalchemy import text
from contract_ref_loader.contract_ref_loader import ContractRefLoader
from utils.contract_utils import month_codes
from utils.contract_utils import custom_monthly_contract_sort_key

instrument_ref_dict = {'CT': {'futures_category': 'Fibers', 'to_USD_conversion': 2204.6/100, 'currency': 'USc',
                              'units': 'lbs', 'lots_to_MT_conversion': 22.679851220176},
                       'VV': {'futures_category': 'Fibers', 'to_USD_conversion': 1, 'currency': 'CNY', 'units': 'MT',
                              'lots_to_MT_conversion': 5},
                       'CCL': {'futures_category': 'Fibers', 'to_USD_conversion': 1000/355.56, 'currency': 'INR',
                               'units': 'candy', 'MT_conversion': 22.679851220176},
                       'OR': {'futures_category': 'Industrial Material'},
                       'JN': {'futures_category': 'Industrial Material'},
                       'SRB': {'futures_category': 'Industrial Material'},
                       'RT': {'futures_category': 'Industrial Material'},
                       'BDR': {'futures_category': 'Industrial Material'},
                       'RG': {'futures_category': 'Industrial Material'},
                       'C ': {'futures_category': 'Corn', 'to_USD_conversion': 39.36821/100, 'currency': 'USc'},
                       'EP': {'futures_category': 'Corn'},
                       'CRD': {'futures_category': 'Corn'},
                       'AC': {'futures_category': 'Corn'},
                       'CA': {'futures_category': 'Wheat'},
                       'W ': {'futures_category': 'Wheat', 'to_USD_conversion': 36.74371/100, 'currency': 'USc'},
                       'KW': {'futures_category': 'Wheat'},
                       'MW': {'futures_category': 'Wheat'},
                       'KFP': {'futures_category': 'Wheat'},
                       'S ': {'futures_category': 'Soy', 'to_USD_conversion': 36.74371/100, 'currency': 'USc'},
                       'SM': {'futures_category': 'Soy'},
                       'BO': {'futures_category': 'Soy'},
                       'AE': {'futures_category': 'Soy'},
                       'SRS': {'futures_category': 'Soy'},
                       'AK': {'futures_category': 'Soy'},
                       'BP': {'futures_category': 'Soy'},
                       'SH': {'futures_category': 'Soy'},
                       'DL': {'futures_category': 'Refined Products'},
                       'QS': {'futures_category': 'Refined Products'},
                       'THE': {'futures_category': 'Refined Products'},
                       'HO': {'futures_category': 'Refined Products'},
                        'SB': {'futures_category': 'Foodstuff', 'to_USD_conversion': 2204.6/100, 'currency': 'USc',
                               'units': 'lbs'},
                       'QW': {'futures_category': 'Foodstuff'},
                       'DF': {'futures_category': 'Foodstuff'},
                       'CC': {'futures_category': 'Foodstuff'},
                       'KC': {'futures_category': 'Foodstuff'},
                       'QC': {'futures_category': 'Foodstuff'},
                       'AX': {'futures_category': 'Foodstuff'},
                       'KO': {'futures_category': 'Foodstuff'},
                       'PAL': {'futures_category': 'Foodstuff'},
                       'VPC': {'futures_category': 'Foodstuff'},
                       'MDS': {'futures_category': 'Foodstuff'},
                       'DA': {'futures_category': 'Foodstuff'},
                       'IJ': {'futures_category': 'Other Grain'},
                       'RS': {'futures_category': 'Other Grain'},
                       'ZRR': {'futures_category': 'Other Grain'},
                       'LHD': {'futures_category': 'Livestock'}}


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

        if isinstance(relevant_months, str):
            relevant_months = [relevant_months]
        elif relevant_months is not None:
            relevant_months = [str(m) for m in relevant_months]

        if isinstance(relevant_years, str) or isinstance(relevant_years, int):
            relevant_years = [str(relevant_years)]
        elif relevant_years is not None:
            relevant_years = [str(y) for y in relevant_years]

        if isinstance(relevant_options, str):
            relevant_options = [relevant_options]
        elif relevant_options is not None:
            relevant_options = [str(o) for o in relevant_options]

        if isinstance(relevant_strikes, str) or isinstance(relevant_strikes, int):
            relevant_strikes = [str(relevant_strikes)]
        elif relevant_strikes is not None:
            relevant_strikes = [str(s) for s in relevant_strikes]

        futures_category = instrument_ref_dict.get(self.instrument_name, {}).get('futures_category')
        if not futures_category:
            print(f"Instrument '{self.instrument_name}' not found in reference dictionary.")
            return pd.DataFrame()

        # Enforce option-only fields
        if mode == 'futures':
            if relevant_options is not None:
                raise ValueError("relevant_options should be None when mode is 'futures'")
            if relevant_strikes is not None:
                raise ValueError("relevant_strikes should be None when mode is 'futures'")

        regex_pattern = self._build_ticker_regex(mode=mode)

        query = f"""
            SELECT DISTINCT i.name, tc.type, dc.unique_id_fut_opt, dc.ticker, dc.opt_exer_typ, dc.futures_category, 
            c.currency, dc.real_underlying_ticker, dc.fut_first_trade_dt, dc.opt_first_trade_dt, dc.last_tradeable_dt, 
            dc.opt_expire_dt
            FROM ref.derivatives_contract dc
            LEFT JOIN ref.currency c
            ON c.id = dc.currency_id 
            LEFT JOIN ref.instrument i
            on i.id = dc.instrument_id
            LEFT JOIN ref.traded_contract tc
            on tc.id = dc.traded_contract_id
            WHERE dc.unique_id_fut_opt ~ '{regex_pattern}'
            AND dc.futures_category = '{futures_category}'
        """
        # print(query)
        if contracts:
            tickers_list = "', '".join(contracts)
            query += f" AND dc.unique_id_fut_opt IN ('{tickers_list}')"

        if relevant_months:
            month_filter = "', '".join(relevant_months)
            query += f" AND SUBSTRING(dc.ticker, {len(self.instrument_name) + 1}, 1) IN ('{month_filter}')"

        if relevant_years:
            year_filter = "', '".join(relevant_years)
            query += f" AND SUBSTRING(dc.ticker, {len(self.instrument_name) + 2}, 1) IN ('{year_filter}')"

        if mode == 'options':

            if relevant_options:
                options_filter = "', '".join(relevant_options)
                query += (f" AND (LENGTH(dc.ticker) = {len(self.instrument_name) + 3} AND RIGHT(dc.ticker, 1) IN "
                          f"('{options_filter}'))")

            if relevant_strikes:
                # Extract the strike portion from ticker: assume format like CTH5C 105 (with space)
                strike_filter = "', '".join(str(s) for s in relevant_strikes)
                query += f" AND SPLIT_PART(dc.unique_id_fut_opt , ' ', 2) IN ('{strike_filter}')"

        query += " ORDER BY dc.unique_id_fut_opt"
        print(query)

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
