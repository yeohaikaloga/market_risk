from position_loader.position_loader import PositionLoader
from utils.contract_utils import instrument_ref_dict, custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text
import re

product_map = {'cotton': ['cto'], 'rms-cfs only': ['cfs'], 'rms': ['cfs', 'rmc'], 'rubber': ['rba', 'rbc']}

class DerivativesPositionLoader(PositionLoader):

    def __init__(self, date, source):
        super().__init__(source)
        self.source = source
        self.date = date

    def load_position(self, date, product, trader_id=None, counterparty_id=None, region=None,
                      book=None) -> pd.DataFrame:

        # Add more products as needed

        if product not in product_map:
            raise ValueError(f"Unsupported product: {product}")

        if product != 'cotton':
            if book is not None:
                raise ValueError("The 'book' parameter is only valid for product='cotton'.")
            if region is not None:
                raise ValueError("The 'region' parameter is only valid for product='cotton'.")

        opera_products = product_map[product]
        opera_products_sql = "(" + ", ".join(f"'{p}'" for p in opera_products) + ")"
        date_sql = f"'{date}'"

        select_cols = ["pos.cob_date", "sp.subportfolio", "pf.portfolio", "sec.security_id", "sec.strike",
                       "sec.derivative_type", "sec.currency"]
        group_by_cols = select_cols.copy()

        joins = ["JOIN position_opera.sub_portfolio sp ON pos.sub_portfolio_id = sp.id",
                 "JOIN position_opera.portfolio pf ON sp.portfolio_id = pf.id",
                 "JOIN position_opera.security sec ON pos.risk_security_id = sec.id"]

        where_conditions = [f"pos.opera_product IN {opera_products_sql}", f"pos.cob_date = {date_sql}",
                            "sp.subportfolio != 'CONSO-CT'",
                            "CAST(pos.tdate AS DATE) BETWEEN CAST(pos.cob_date AS DATE) AND CAST(pos.cob_date AS DATE) "
                            "+ INTERVAL '1 day'"]  # <-- tdate-cob_date filter as Fri/Sat/Sun tdate maps to Fri cob_date

        # Cotton-specific joins and filters
        if product == 'cotton':
            joins.append("JOIN staging.portfolio_region_mapping_table_ex prmte ON prmte.unit_sd = sp.subportfolio")

            select_cols = ["pos.cob_date", "sec.security_id", "sp.subportfolio", "pf.portfolio", "sec.strike",
                           "sec.derivative_type", "sec.product_code", "sec.contract_month", "sec.currency",
                           "prmte.region", "prmte.books"]
            group_by_cols = select_cols.copy()

            where_conditions.append("""
                prmte.updated_timestamp = (SELECT MAX(updated_timestamp) FROM staging.portfolio_region_mapping_table_ex)
            """)

            excluded_book = ['ADMIN', 'NON OIL', 'POOL']

            if book is None or book == 'all':
                where_conditions.append(
                    "prmte.books NOT IN (" + ", ".join(f"'{b}'" for b in excluded_book) + ")"
                )
            else:
                if isinstance(book, str):
                    book = [book]
                book_sql = ", ".join(f"'{b}'" for b in book)
                excluded_book_sql = ", ".join(f"'{b}'" for b in excluded_book)
                where_conditions.append(f"prmte.books IN ({book_sql})")
                where_conditions.append(f"prmte.books NOT IN ({excluded_book_sql})")

            if region is not None:
                if isinstance(region, (list, tuple, set)):
                    region_list_sql = ", ".join(f"'{r}'" for r in region)
                    where_conditions.append(f"prmte.region IN ({region_list_sql})")
                else:
                    where_conditions.append(f"prmte.region = '{region}'")

        # Trader filter
        if trader_id is not None or trader_id == 'all':
            joins.append("JOIN position_opera.trader tr ON pos.trader_id = tr.id")
            select_cols += ["tr.id AS trader_id", "tr.name AS trader_name"]
            group_by_cols += ["tr.id", "tr.name"]
            if trader_id != 'all':
                where_conditions.append(f"tr.id = {trader_id}")

        # Counterparty filter
        if counterparty_id is not None or counterparty_id == 'all':
            joins.append("JOIN position_opera.counterparty cp ON pos.counterparty_id = cp.id")
            select_cols += ["cp.id AS counterparty_id", "cp.counterparty_parent"]
            group_by_cols += ["cp.id", "cp.counterparty_parent"]
            if counterparty_id != 'all':
                where_conditions.append(f"cp.id = {counterparty_id}")

        query = f"""
            SELECT {', '.join(select_cols)},
                   SUM(pos.total_active_lots) AS total_active_lots
            FROM position_opera.position pos
            {' '.join(joins)}
            WHERE {' AND '.join(where_conditions)}
            AND derivative_type in ('future', 'vanilla_call', 'vanilla_put')
            GROUP BY {', '.join(group_by_cols)}
        """

        print(query)  # Debug
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        return df

    @staticmethod
    def map_cotton_unit(row):
        portfolio = row['portfolio']

        if not isinstance(portfolio, str):
            return 'NULL'

        # CC portfolio special handling
        cc_num_match = re.match(r'^CC_PR(\d+)', portfolio)
        if cc_num_match:
            return f"CENTRAL {cc_num_match.group(1)}"

        if portfolio.startswith('CC_SP'):
            return 'SPREADS'

        if portfolio == 'CC_BK':
            return 'CEN SPR'

        if portfolio.startswith('CC'):
            return 'CENTRAL'

        # Special hardcoded exact matches
        exact_match_map = {'WAF_CP': 'BFASO', 'MILL OPTNS': 'NON OIL', 'BRZ_GROWER': 'NON OIL', 'US_GROWER': 'NON OIL',
                           'USA_LIB_OTC': 'LIBERTY', 'USA_LIB_PR': 'LIBERTY', 'USA_PR_OTC': 'USA', 'USA_EQ_PR': 'USA',
                           'LIQ_SWAP': 'LIQ SWAP'}

        if portfolio in exact_match_map:
            return exact_match_map[portfolio]

        # USA EQUITY-specific
        if portfolio.startswith(('USA_LIB', 'USA_EQ')):
            return 'USA EQUITY'

        # LIBERTY (not under USA)
        if portfolio.startswith('LIB'):
            return 'LIBERTY'

        # CHAD override
        if portfolio.startswith('CHAD'):
            return 'CHAD TRADE'

        # General prefix-to-unit mapping
        prefix_to_unit_map = {'AUST': 'AUSTRALIA', 'BENIN': 'BENIN', 'BFASO': 'BFASO', 'BF': 'BFASO', 'BRZ': 'BRAZIL',
                              'CAM': 'CAMEROON', 'CHINA': 'CHINA', 'GREECE': 'GREECE', 'IND': 'INDIA', 'MALI': 'MALI',
                              'MEX': 'MEXICO', 'MOZ': 'MOZAM', 'SECO': 'SECO', 'SPAIN': 'SPAIN', 'TNZ': 'TANZANIA',
                              'TOGO': 'TOGO', 'USA': 'USA', 'US': 'USA', 'ZIMB': 'ZIMBABWE'}

        for prefix, unit in prefix_to_unit_map.items():
            if portfolio.startswith(prefix):
                return unit

        return 'NULL'

    def assign_cotton_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['unit'] = df.apply(self.map_cotton_unit, axis=1)

        invalid_units = df[df['unit'] == 'NULL']
        if not invalid_units.empty:
            print("Warning: Found rows with unit == 'NULL':")
            print(invalid_units)
            # You can also raise an exception here if you want to stop processing
            # raise ValueError("Invalid units found in DataFrame.")
        else:
            print("All units valid.")

        return df

    @staticmethod
    def map_bbg_tickers(security_id: str) -> tuple[str | None, str | None]:
        """
        Returns both Bloomberg-style option ticker and underlying ticker.
        Returns: (option_ticker, underlying_ticker)
        """
        if not isinstance(security_id, str):
            return None, None

        security_id = security_id.strip()
        if not (security_id.startswith("CM ") or security_id.startswith("IM ")):
            return None, None

        core = security_id[3:].strip()

        try:
            # Handle options
            if '.' in core:
                pattern = r'^(\w+)\s+([A-Z])(\d{2})\.([A-Z])(\d{2})([CP])\s+(\d+)$'
                m = re.match(pattern, core)
                if not m:
                    return None, None

                asset_raw, opt_month, opt_year, und_month, und_year, opt_type, strike = m.groups()
                asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw

                option_ticker = f"{asset}{opt_month}{opt_year[1]}{opt_type} {strike} Comdty"
                underlying_ticker = f"{asset}{und_month}{und_year[1]} Comdty"

                return option_ticker, underlying_ticker

            # Handle futures
            else:
                parts = re.split(r'\s+', core)
                if len(parts) != 2:
                    return None, None

                asset_raw, expiry = parts
                asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw

                m = re.match(r'^([A-Z])(\d{2})$', expiry)
                if not m:
                    return None, None

                month_code, year_code = m.groups()
                fut_ticker = f"{asset}{month_code}{year_code[1]} Comdty"

                # For futures, both option and underlying ticker are the same
                return fut_ticker, fut_ticker
        except Exception:
            return None, None

    def assign_bbg_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[['bbg_ticker', 'underlying_bbg_ticker']] = (df['security_id'].
                                                       apply(lambda x: pd.Series(self.map_bbg_tickers(x))))

        for col in ['bbg_ticker', 'underlying_bbg_ticker']:
            invalid = df[df[col].isna()]
            if not invalid.empty:
                print(f"Warning: Found {len(invalid)} invalid security_ids for column '{col}':")
                print(invalid[['security_id']])
            else:
                print(f"All security_ids successfully mapped to '{col}'.")
        return df

    #TODO Add fallback assignment here. necessary for rubber.
    @staticmethod
    def map_generic_curve_with_fallback(contract, contract_to_curve_map):
        """
        Try to find the best generic curve match for a given contract.
        Uses fallback logic if the exact contract is missing.
        """
        if contract in contract_to_curve_map:
            return contract_to_curve_map[contract]

        # Sort contracts using the provided sort key
        sorted_contracts = sorted(contract_to_curve_map.keys(), key=custom_monthly_contract_sort_key)

        # Find the next available later contract
        for c in sorted_contracts:
            if custom_monthly_contract_sort_key(c) > custom_monthly_contract_sort_key(contract):
                return contract_to_curve_map[c]

        # If no later contract is found — fallback to the last available
        if len(sorted_contracts) > 0:
            last_contract = sorted_contracts[-1]
            print(f"[Fallback] {contract} not found — using last available contract {last_contract}")
            return contract_to_curve_map[last_contract]

        return None

    @staticmethod
    def map_generic_curve(row, instrument_dict):
        instrument = row['product_code'].replace('CM ', '').replace('IM ','')
        if len(instrument) == 1:
            instrument = instrument + ' '
        contract = row['underlying_bbg_ticker'].replace(' Comdty', '')
        instrument_info = instrument_dict.get(instrument)
        if not instrument_info or not isinstance(instrument_info, dict):
            print(f"[WARN] Instrument '{instrument}' not found or invalid in instrument_dict.")
            return None

        curve_map = instrument_info.get('contract_to_curve_map', {})

        if contract in curve_map:
            return curve_map[contract]

        # Fallback to next available or last contract
        fallback = DerivativesPositionLoader.map_generic_curve_with_fallback(
            contract, curve_map
        )
        if fallback:
            print(
                f"[Fallback] {contract} mapped to {fallback} under instrument {instrument} for {row.get('security_id', '')}")
            return fallback

        print(f"[WARN] No mapping or fallback found for contract {contract} under instrument {instrument}")
        return None

    def assign_generic_curves(self, df: pd.DataFrame, instrument_dict: dict) -> pd.DataFrame:
        df = df.copy()
        df['generic_curve'] = df.apply(lambda row: self.map_generic_curve(row, instrument_dict), axis=1)

        invalid = df[df['generic_curve'].isna()]
        if not invalid.empty:
            print(f"Warning: {len(invalid)} positions could not be mapped to a generic curve.")
            print(invalid[['security_id', 'product_code', 'underlying_bbg_ticker']])
        else:
            print("All positions successfully mapped to a generic curve.")
        return df

    @staticmethod
    def map_conversion_to_mt(row):
        """
        Extract instrument prefix and return total_active_lots * conversion factor.
        """
        security_id = row['security_id']
        if not isinstance(security_id, str):
            return 0

        match = re.match(r'^CM\s+([A-Z]{1,3}\s?)\s+\w+', security_id)
        if match:
            prefix = match.group(1)
            # Normalize prefix: strip trailing space for matching dict keys
            prefix_norm = prefix.rstrip()

            # The dict keys may have trailing spaces, so check both forms:
            if prefix in instrument_ref_dict and 'lots_to_MT_conversion' in instrument_ref_dict[prefix]:
                conversion = instrument_ref_dict[prefix]['lots_to_MT_conversion']
                return row['total_active_lots'] * conversion
            elif prefix_norm in instrument_ref_dict and 'lots_to_MT_conversion' in instrument_ref_dict[prefix_norm]:
                conversion = instrument_ref_dict[prefix_norm]['lots_to_MT_conversion']
                return row['total_active_lots'] * conversion

        # No match or no conversion found
        return 0

    def assign_total_active_mt(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['total_active_MT'] = df.apply(self.map_conversion_to_mt, axis=1)

        invalid_conversions = df[df['total_active_MT'] == 0]
        if not invalid_conversions.empty:
            print("Warning: Found rows with zero conversion:")
            print(invalid_conversions[['security_id', 'total_active_lots', 'total_active_MT']])
            # You can raise an error here if you want

        else:
            print("All conversions applied successfully.")

        return df

    def load_opera_sensitivities(self, position_df: pd.DataFrame, sensitivity_types: list[str], product: str) -> pd.DataFrame:
        """
        Fetch and deduplicate OPERA sensitivities from the DB for the given positions,
        loading multiple sensitivity columns at once.

        Args:
            position_df (pd.DataFrame): DataFrame containing positions with 'security_id' and 'cob_date'.
            sensitivity_types (list[str]): List of sensitivity column names to load.
            product (str): Product name.

        Returns:
            pd.DataFrame: sensitivity_df with columns ['security_id', 'cob_date', *sensitivity_types]
        """
        if position_df.empty:
            print("No positions available; skipping sensitivity load.")
            return pd.DataFrame()

        supported_types = ['settle_delta_1', 'settle_delta_2', 'settle_gamma_11', 'settle_gamma_12', 'settle_gamma_21',
                           'settle_gamma_22', 'settle_vega_1', 'settle_vega_2', 'settle_theta', 'settle_chi']
        for s_type in sensitivity_types:
            if s_type not in supported_types:
                raise ValueError(f"Unsupported sensitivity_type '{s_type}'. Must be one of {supported_types}.")

        if product not in product_map:
            raise ValueError(f"Unsupported product: {product}")

        # Prepare query values
        position_df['cob_date'] = pd.to_datetime(position_df['cob_date'], errors='coerce')
        cob_dates = position_df['cob_date'].dt.strftime('%Y-%m-%d').unique()
        security_ids = position_df['security_id'].unique()
        opera_product = product_map[product]

        cob_dates_sql = ", ".join(f"'{d}'" for d in cob_dates)
        security_ids_sql = ", ".join(f"'{sid}'" for sid in security_ids)
        opera_product_sql = ", ".join(f"'{p}'" for p in opera_product)
        sensitivities_sql = ", ".join(sensitivity_types)

        sens_query = f"""
            SELECT security_id, cob_date, {sensitivities_sql}
            FROM staging.opera_security_settlement
            WHERE cob_date IN ({cob_dates_sql})
            AND security_id IN ({security_ids_sql})
            AND opera_product IN ({opera_product_sql})
        """
        print(sens_query)

        with self.source.connect() as conn:
            raw_df = pd.read_sql_query(text(sens_query), conn)
            raw_df['cob_date'] = pd.to_datetime(raw_df['cob_date'], errors='coerce')

        if raw_df.empty:
            return pd.DataFrame()

        # Deduplicate
        deduplicated_df = raw_df.groupby(['security_id', 'cob_date'], as_index=False)[sensitivity_types].first()

        # Debug: show what was removed
        raw_df['key'] = raw_df[['security_id', 'cob_date']].apply(tuple, axis=1)
        deduplicated_df['key'] = deduplicated_df[['security_id', 'cob_date']].apply(tuple, axis=1)
        removed_rows = raw_df[~raw_df['key'].isin(deduplicated_df['key'])]
        retained_rows = raw_df[raw_df['key'].isin(deduplicated_df['key'])]

        if not removed_rows.empty:
            print(f"Removed {len(removed_rows)} duplicate rows:")
            print(removed_rows[[sensitivity_types, 'security_id', 'cob_date']])
            print(f"Retained {len(retained_rows)} rows.")

        return deduplicated_df.drop(columns='key', errors='ignore')

    @staticmethod
    def assign_opera_sensitivities(position_df: pd.DataFrame, sensitivity_df: pd.DataFrame,
                                   sensitivity_types: list[str]) -> pd.DataFrame:
        """
        Merge multiple sensitivities onto the position DataFrame with validation.

        Args:
            position_df (pd.DataFrame): Original positions DataFrame.
            sensitivity_df (pd.DataFrame): DataFrame containing sensitivities.
            sensitivity_types (list[str]): List of sensitivity columns to merge.

        Returns:
            pd.DataFrame: positions DataFrame with sensitivity columns added.
        """
        original_count = len(position_df)

        if sensitivity_df.empty:
            print(f"No sensitivity data ({sensitivity_types}) found for the positions.")
            for s_type in sensitivity_types:
                position_df[s_type] = None
            return position_df

        merged_df = position_df.merge(sensitivity_df, on=['security_id', 'cob_date'], how='left')

        # Validate row count after merge
        if len(merged_df) > original_count:
            dupes = merged_df.duplicated(subset=['security_id', 'cob_date'], keep=False)
            print("Merge increased row count — possible Cartesian product!")
            print(merged_df[dupes][['security_id', 'cob_date', sensitivity_types]])
            raise ValueError("Merge caused row duplication. Check for 1:N merge.")

        # Check for missing sensitivities
        missing = merged_df[sensitivity_types].isna().all(axis=1)
        if missing.any():
            missing_rows = merged_df[missing]
            print(f"{len(missing_rows)} positions missing all of {sensitivity_types}:")
            print(missing_rows[['security_id', 'cob_date', 'portfolio', 'trader_id']])
        else:
            print(f"All positions successfully mapped to sensitivities: {sensitivity_types}")

        return merged_df
