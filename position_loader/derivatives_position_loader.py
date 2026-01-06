from position_loader.position_loader import PositionLoader
from utils.contract_utils import load_instrument_ref_dict, custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text
import re
from datetime import datetime
import numpy as np
from utils.log_utils import get_logger

logger = get_logger(__name__)
product_map = {'cotton': ['cto'], 'rms-cfs only': ['cfs'], 'rms': ['cfs', 'rmc'], 'rubber': ['rba', 'rbc']}

class DerivativesPositionLoader(PositionLoader):

    def __init__(self, date, source):
        super().__init__(source)
        self.source = source
        self.date = date

    def load_position(self, date, product, trader_id=None, counterparty_id=None, region=None,
                      book=None) -> pd.DataFrame:

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

        select_cols = ["pos.cob_date", "sec.security_id", "sp.subportfolio", "pf.portfolio", "sec.strike",
                       "sec.derivative_type", "sec.product_code", "sec.contract_month", "sec.currency"]
        group_by_cols = select_cols.copy()

        joins = ["JOIN position_opera.sub_portfolio sp ON pos.sub_portfolio_id = sp.id",
                 "JOIN position_opera.portfolio pf ON sp.portfolio_id = pf.id",
                 "JOIN position_opera.security sec ON pos.risk_security_id = sec.id"]
        excluded_subportfolios = ['CONSO-CT', 'US_GROWER', 'MILL OPTNS', 'USA_LIB_OTC', 'BRZ_GROWER', 'USA_LIB_PR']
        excluded_subportfolios_sql = ", ".join(f"'{p}'" for p in excluded_subportfolios)

        where_conditions = [
            f"pos.opera_product IN {opera_products_sql}",
            f"pos.cob_date = {date_sql}",
            f"sp.subportfolio NOT IN ({excluded_subportfolios_sql})"
        ]

        # Cotton-specific joins and filters
        if product == 'cotton':

            prmte_join_conditions = []
            prmte_join_conditions.append("prmte.load_date = (SELECT MAX(load_date) "
                                         "FROM staging.portfolio_region_mapping_table_ex)")
            excluded_book = ['ADMIN', 'NON OIL', 'POOL']
            excluded_book_sql = ", ".join(f"'{b}'" for b in excluded_book)

            if book is not None and book != 'all':
                if isinstance(book, str):
                    book = [book]
                book_sql = ", ".join(f"'{b}'" for b in book)
                # Only the explicit inclusion filter remains in the ON clause
                prmte_join_conditions.append(f"prmte.books IN ({book_sql})")

            if region is not None:
                if isinstance(region, (list, tuple, set)):
                    region_list_sql = ", ".join(f"'{r}'" for r in region)
                    prmte_join_conditions.append(f"prmte.region IN ({region_list_sql})")
                else:
                    prmte_join_conditions.append(f"prmte.region = '{region}'")

            on_clause = " AND ".join(prmte_join_conditions)
            prmte_join_string = (
                f"LEFT JOIN staging.portfolio_region_mapping_table_ex prmte "
                f"ON prmte.unit_sd = sp.subportfolio AND {on_clause}"
            )
            joins.append(prmte_join_string)
            where_conditions.append(f"prmte.books NOT IN ({excluded_book_sql})")

            select_cols = ["pos.cob_date", "sec.security_id", "sp.subportfolio", "pf.portfolio", "sec.strike",
                           "sec.derivative_type", "sec.product_code", "sec.contract_month", "sec.currency",
                           "prmte.region", "prmte.books"]
            group_by_cols = select_cols.copy()

        # Trader filter
        if trader_id is not None:
            joins.append("JOIN position_opera.trader tr ON pos.trader_id = tr.id")
            select_cols += ["tr.id AS trader_id", "tr.name AS trader_name"]
            group_by_cols += ["tr.id", "tr.name"]
            if trader_id != 'all':
                where_conditions.append(f"tr.id = {trader_id}")

        # Counterparty filter
        if counterparty_id is not None:
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
                AND derivative_type NOT IN ('avg_cc_swap', 'fx_forward_df', 'fx_forward_ndf', 'fx_vanilla_call', 
                'fx_vanilla_put')
                GROUP BY {', '.join(group_by_cols)}
                """

        print(query)
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if product == 'cotton':
            unmapped = df[df['region'].isna() | df['books'].isna()]
            if not unmapped.empty:
                # Use 'portfolio' column to identify the source of the unmapped data
                unmapped_ports = unmapped['portfolio'].unique().tolist()
                print(
                    f"[WARNING]: {len(unmapped.index)} position(s) are missing prmte mapping data."
                    f"\n  The following portfolios are not mapped in portfolio_region_mapping_table_ex: {unmapped_ports}"
                    f"\n  These positions are included in the results but their Region/Books columns are NULL."
                )

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
        Maps internal security_id to Bloomberg-style option and underlying tickers.
        Handles:
          - Futures (e.g. "CM CT Z25")
          - Vanilla options (e.g. "CM CT Z25.Z25C 67")
          - Inter-month options (e.g. "CM CT Z25.H26C 67")
          - Dated options (e.g. "CM CT 20251103.H25C 63.95")
          - Exotic ADOC/ADOP/ADIC/ADIP/AUOC/AUOP/AUIC/AUIP
          - OTC monthly-style European options like:
            "CM OR F26.F26 20251201 20251231 EUR"

        Returns:
            (option_ticker, underlying_ticker)
        """
        import re

        if not isinstance(security_id, str):
            return None, None

        security_id = security_id.strip()
        if not (security_id.startswith("CM ") or security_id.startswith("IM ")):
            return None, None

        core = security_id[3:].strip()

        try:
            # ============================================================
            # 0️⃣ NEW: Handle OTC structured options like:
            #    OR F26.F26 20251201 20251231 EUR
            # ============================================================
            # Format:
            # <ASSET> <OPT_EXP>.<UND_EXP> <START> <END> <CCY>
            m = re.match(
                r'^(\w+)\s+([A-Z]\d{2})\.([A-Z]\d{2})\s+\d{8}\s+\d{8}\s+[A-Z]{3}$',
                core
            )
            if m:
                asset_raw, opt_exp, und_exp = m.groups()
                asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw
                und_month = und_exp[0]
                und_year = und_exp[1:]  # e.g., "26"
                underlying_ticker = f"{asset}{und_month}{und_year[1]} Comdty"
                # Option ticker is NOT constructible from this format → return None
                return None, underlying_ticker

            # ============================================================
            # 1️⃣ Handle exotic ADOC/ADOP/ADIC/ADIP etc.
            # ============================================================
            if any(tag in core for tag in ("ADOC", "ADOP", "ADIC", "ADIP", "AUOC", "AUOP", "AUIC", "AUIP")):
                m = re.search(r'(\w+)\s+\d{8}\.([A-Z])(\d{2})', core)
                if m:
                    asset_raw, und_month, und_year = m.groups()
                    asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw
                    underlying_ticker = f"{asset}{und_month}{und_year[1]} Comdty"
                    return None, underlying_ticker
                return None, None

            # ============================================================
            # 2️⃣ Handle dated or regular options
            # ============================================================
            if '.' in core:
                # Dated option pattern
                pattern = r'^(\w+)\s+(?:\d{8}\.)?([A-Z])(\d{2})([CP])\s+([\d.]+)$'
                m = re.match(pattern, core)
                if m:
                    asset_raw, und_month, und_year, opt_type, strike = m.groups()
                    asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw
                    option_ticker = f"{asset}{und_month}{und_year[1]}{opt_type} {strike} Comdty"
                    underlying_ticker = f"{asset}{und_month}{und_year[1]} Comdty"
                    return option_ticker, underlying_ticker

                # Regular or inter-month (CT Z25.H26C 67)
                pattern2 = r'^(\w+)\s+([A-Z])(\d{2})\.([A-Z])(\d{2})([CP])\s+([\d.]+)$'
                m = re.match(pattern2, core)
                if m:
                    asset_raw, opt_month, opt_year, und_month, und_year, opt_type, strike = m.groups()
                    asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw
                    option_ticker = f"{asset}{opt_month}{opt_year[1]}{opt_type} {strike} Comdty"
                    underlying_ticker = f"{asset}{und_month}{und_year[1]} Comdty"
                    return option_ticker, underlying_ticker

            # ============================================================
            # 3️⃣ Handle futures
            # ============================================================
            parts = re.split(r'\s+', core)
            if len(parts) >= 2:
                asset_raw, expiry = parts[0], parts[1]
                asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw
                m = re.match(r'^([A-Z])(\d{2})$', expiry)
                if m:
                    month_code, year_code = m.groups()
                    fut_ticker = f"{asset}{month_code}{year_code[1]} Comdty"
                    return fut_ticker, fut_ticker

            return None, None

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
        instrument = row['product_code'].replace('CM ', '').replace('IM ', '')

        # Skip if no underlying Bloomberg ticker
        if pd.isna(row.get('underlying_bbg_ticker')) or row.get('underlying_bbg_ticker') is None:
            print(f"[SKIP] No underlying_bbg_ticker for {row.get('security_id', 'unknown')} — skipping mapping.")
            return None

        contract = row['underlying_bbg_ticker'].replace(' Comdty', '')
        instrument_info = instrument_dict.get(instrument)

        if instrument_info is None:
            print(f"[WARN] Instrument '{instrument}' not found or invalid in instrument_dict.")
            return None

        curve_map = instrument_info.get('contract_to_curve_map', {})

        if contract in curve_map:
            return curve_map[contract]

        # Fallback to next available or last contract
        fallback = DerivativesPositionLoader.map_generic_curve_with_fallback(contract, curve_map)
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
            instrument_ref_dict = load_instrument_ref_dict('uat')
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
            print(missing_rows[['security_id', 'cob_date', 'portfolio']])
        else:
            print(f"All positions successfully mapped to sensitivities: {sensitivity_types}")

        return merged_df

    def assign_cob_date_price(self, position_df: pd.DataFrame, price_df: pd.DataFrame, cob_date: str) -> pd.DataFrame:
        """
        Assign cob_date_price to each position row for a single cob_date.

        Args:
            position_df: DataFrame with columns ['generic_curve', ...].
            price_df: pivot-like DataFrame, index=cob_date, columns=generic_curve.
            cob_date: string date to extract prices for.
        """
        position_df = position_df.copy()

        # Ensure index is datetime
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index)
        cob_date_dt = pd.to_datetime(cob_date)

        if cob_date_dt not in price_df.index:
            raise KeyError(f"cob_date {cob_date} not found in price_df index")

        # Extract prices for this cob_date
        prices_for_date = price_df.loc[cob_date_dt]

        # Map generic_curve -> price
        position_df['cob_date_price'] = position_df['generic_curve'].map(prices_for_date)

        # Detect missing prices
        missing = position_df['cob_date_price'].isna()
        if missing.any():
            logger.warning(f"{missing.sum()} positions missing cob_date_price:")
            print(position_df[missing][['generic_curve']])
        else:
            print("All positions mapped to cob_date_price.")

        return position_df


    def load_rms_screen(self):

        query = f"""
                SELECT * from staging.opera_sensitivities_rms osr
                WHERE settlement_date = '{self.date}'
                """
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        print(query)
        return df
    def assign_risk_factors_for_rms_screen(self, deriv_pos_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns risk_factor based on continuation series with a fallback mechanism.
        If the resolved series (e.g., SM8) is not in price_df, it falls back to
        the highest available previous number (e.g., SM7, SM6...).
        """
        # 1. Apply primary logic provided
        deriv_pos_df['risk_factor'] = np.where(
            deriv_pos_df['option_continuation_series'].isna(),
            deriv_pos_df['future_continuation_series'],
            deriv_pos_df['option_continuation_series']
        )

        available_columns = set(price_df.columns)

        def get_best_available_ticker(ticker):
            if pd.isna(ticker) or ticker in available_columns:
                return ticker

            # Regex to split ticker into root and number (e.g., "SM" and "8")
            match = re.match(r"([a-zA-Z\s]+)(\d+)", str(ticker))
            if not match:
                return ticker  # Cannot parse, return as is (will likely result in NaN later)

            root, num_str = match.groups()
            num = int(num_str)

            # Iteratively look back from num-1 down to 1
            for i in range(num - 1, 0, -1):
                fallback_ticker = f"{root}{i}"
                if fallback_ticker in available_columns:
                    return fallback_ticker

            return ticker  # No fallback found, return original

        # 2. Apply the fallback logic to items not found in columns
        # We only apply this to unique values that are actually missing to optimize performance
        missing_targets = set(deriv_pos_df['risk_factor'].dropna().unique()) - available_columns

        if missing_targets:
            mapping = {t: get_best_available_ticker(t) for t in missing_targets}
            deriv_pos_df['risk_factor'] = deriv_pos_df['risk_factor'].replace(mapping)

            # Optional: Log the substitutions for transparency
            for original, substituted in mapping.items():
                if original != substituted:
                    logger.warning(f"Fallback: {original} not in price_df, using {substituted} instead.")

        return deriv_pos_df