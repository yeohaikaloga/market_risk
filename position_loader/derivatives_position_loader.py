from position_loader.position_loader import PositionLoader
from contract_ref_loader.derivatives_contract_ref_loader import instrument_ref_dict
import pandas as pd
from sqlalchemy import text
import re


class DerivativesPositionLoader(PositionLoader):

    def __init__(self, date, source):
        super().__init__(source)
        self.source = source
        self.date = date

    def load_position(self, date, product, trader_id=None, counterparty_id=None, region=None,
                      books=None) -> pd.DataFrame:
        product_map = {'cotton': ['cto'], 'rms-cfs only': ['cfs'], 'rms': ['cfs', 'rmc'], 'rubber': ['rba', 'rbc']}
            # Add more products as needed

        if product not in product_map:
            raise ValueError(f"Unsupported product: {product}")

        if product != 'cotton':
            if books is not None:
                raise ValueError("The 'books' parameter is only valid for product='cotton'.")
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
                            "CAST(pos.tdate AS DATE) BETWEEN CAST(pos.cob_date AS DATE) AND CAST(pos.cob_date AS DATE) + INTERVAL '1 day'"]  # <-- tdate-cob_date filter as Fri/Sat/Sun tdate maps to Fri cob_date

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

            excluded_books = ['ADMIN', 'NON OIL', 'POOL']

            if books is None or books == 'all':
                where_conditions.append(
                    "prmte.books NOT IN (" + ", ".join(f"'{b}'" for b in excluded_books) + ")"
                )
            else:
                if isinstance(books, str):
                    books = [books]
                books_sql = ", ".join(f"'{b}'" for b in books)
                excluded_books_sql = ", ".join(f"'{b}'" for b in excluded_books)
                where_conditions.append(f"prmte.books IN ({books_sql})")
                where_conditions.append(f"prmte.books NOT IN ({excluded_books_sql})")

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
            return (None, None)

        security_id = security_id.strip()
        if not security_id.startswith("CM "):
            return (None, None)

        core = security_id[3:].strip()

        try:
            # Handle options
            if '.' in core:
                pattern = r'^(\w+)\s+([A-Z])(\d{2})\.([A-Z])(\d{2})([CP])\s+(\d+)$'
                m = re.match(pattern, core)
                if not m:
                    return (None, None)

                asset_raw, opt_month, opt_year, und_month, und_year, opt_type, strike = m.groups()
                asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw

                option_ticker = f"{asset}{opt_month}{opt_year[1]}{opt_type} {strike} Comdty"
                underlying_ticker = f"{asset}{und_month}{und_year[1]} Comdty"

                return (option_ticker, underlying_ticker)

            # Handle futures
            else:
                parts = re.split(r'\s+', core)
                if len(parts) != 2:
                    return (None, None)

                asset_raw, expiry = parts
                asset = asset_raw + ' ' if len(asset_raw) == 1 else asset_raw

                m = re.match(r'^([A-Z])(\d{2})$', expiry)
                if not m:
                    return (None, None)

                month_code, year_code = m.groups()
                fut_ticker = f"{asset}{month_code}{year_code[1]} Comdty"

                # For futures, both option and underlying ticker are the same
                return (fut_ticker, fut_ticker)
        except Exception:
            return (None, None)

    def assign_bbg_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[['bbg_ticker', 'underlying_bbg_ticker']] = (df['security_id'].
                                                       apply(lambda x: pd.Series(self.map_bbg_tickers(x))))

        for col in ['bbg_ticker', 'underlying_bbg_ticker']:
            invalid = df[df[col].isna()]
            if not invalid.empty:
                print(f"Warning: Found {len(invalid)} invalid or unconvertible security_ids for column '{col}':")
                print(invalid[['security_id']])
            else:
                print(f"All security_ids successfully mapped to '{col}'.")
        return df

    @staticmethod
    def map_generic_curve(row, instrument_dict):
        instrument = row['product_code'].replace('CM ', '')  # e.g., 'CM CT' → 'CT'
        if len(instrument) == 1:
            instrument = instrument + ' '
        contract = row['underlying_bbg_ticker'].replace(' Comdty', '')
        if instrument in instrument_dict:
            curve_map = instrument_dict[instrument].get('contract_to_curve_map', {})
            return curve_map.get(contract)
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

    def map_conversion_to_MT(self, row):
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

    def assign_total_active_MT(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['total_active_MT'] = df.apply(self.map_conversion_to_MT, axis=1)

        invalid_conversions = df[df['total_active_MT'] == 0]
        if not invalid_conversions.empty:
            print("Warning: Found rows with zero conversion:")
            print(invalid_conversions[['security_id', 'total_active_lots', 'total_active_MT']])
            # You can raise an error here if you want

        else:
            print("All conversions applied successfully.")

        return df

    def load_sensitivities(self, position_df: pd.DataFrame, sensitivity_type: str = 'delta') -> pd.DataFrame:
        """
        Load sensitivities for securities in the given position DataFrame and merge them.

        Sensitivities are only loaded if position_df is not empty.

        Args:
            position_df (pd.DataFrame): DataFrame containing positions with 'cob_date' and 'security_id' columns.
            sensitivity_type (str): Type of sensitivity to load (e.g., 'delta', 'gamma', etc.)

        Returns:
            pd.DataFrame: Merged DataFrame including sensitivity values.
        """
        if position_df.empty:
            print("No positions available; skipping sensitivity load.")
            return position_df

        supported_types = ['settle_delta_1', 'settle_delta_2', 'settle_gamma_11', 'settle_gamma_12', 'settle_gamma_21',
                           'settle_gamma2', 'settle_vega_1', 'settle_vega_2', 'settle_theta', 'settle_chi']
        if sensitivity_type not in supported_types:
            raise ValueError(f"Unsupported sensitivity_type '{sensitivity_type}'. Must be one of {supported_types}.")

        # Extract the cob_date(s) and security_id(s) present in the position_df to filter sens query
        position_df['cob_date'] = pd.to_datetime(position_df['cob_date'], errors='coerce')
        cob_dates = position_df['cob_date'].dt.strftime('%Y-%m-%d').unique()
        security_ids = position_df['security_id'].unique()

        cob_dates_sql = ", ".join(f"'{d}'" for d in cob_dates)
        security_ids_sql = ", ".join(f"'{sid}'" for sid in security_ids)

        #TODO Watch out for change in definition of cob_date and check whether it continues to match with 1201 or RMS reports
        sens_query = f"""
            SELECT security_id, cob_date, {sensitivity_type}
            FROM staging.opera_security_settlement
            WHERE cob_date IN ({cob_dates_sql})
              AND security_id IN ({security_ids_sql})
              AND {sensitivity_type} IS NOT NULL
        """
        print(sens_query)

        with self.source.connect() as conn:
            raw_sens_df = pd.read_sql_query(text(sens_query), conn)
            raw_sens_df['cob_date'] = pd.to_datetime(raw_sens_df['cob_date'], errors='coerce')

        if raw_sens_df.empty:
            print(f"No sensitivity data ({sensitivity_type}) found for the positions.")
            position_df[f'{sensitivity_type}'] = None
            return position_df

        # Deduplicate before merging (keep first value for each security_id + cob_date)
        sensitivity_df = raw_sens_df.groupby(['security_id', 'cob_date'], as_index=False)[sensitivity_type].first()

        raw_sens_df['key'] = raw_sens_df[['security_id', 'cob_date']].apply(tuple, axis=1)
        sensitivity_df['key'] = sensitivity_df[['security_id', 'cob_date']].apply(tuple, axis=1)

        removed_rows = raw_sens_df[~raw_sens_df['key'].isin(sensitivity_df['key'])]
        retained_rows = raw_sens_df[raw_sens_df['key'].isin(sensitivity_df['key'])]

        if not removed_rows.empty:
            print(f"\n❌ {len(removed_rows)} duplicate row(s) removed:")
            print(removed_rows[[sensitivity_type, 'security_id', 'cob_date']])

            print(f"\n✅ {len(retained_rows)} row(s) retained:")
            print(retained_rows[[sensitivity_type, 'security_id', 'cob_date']])

        # Clean up temp key column
        sensitivity_df.drop(columns='key', inplace=True)

        merged_df = position_df.merge(sensitivity_df, on=['security_id', 'cob_date'], how='left')
        missing_df = merged_df[f'{sensitivity_type}'].isnull()
        if missing_df.any():
            missing_rows = merged_df.loc[missing_df, ['security_id', 'cob_date', 'portfolio', 'books', 'trader_id']]
            missing_count = len(missing_rows)
            print(f"\n{missing_count} position(s) are missing '{sensitivity_type}':\n")
            print(missing_rows)
        else:
            print("All positions successfully mapped to a f'{sensitivity_type}'.")
        return merged_df