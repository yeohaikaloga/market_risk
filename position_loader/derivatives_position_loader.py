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
        # Add others like 'wood', etc. as needed

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

        select_cols = ["pos.tdate", "sp.subportfolio", "pf.portfolio", "sec.security_id", "sec.strike",
                       "sec.derivative_type", "sec.currency"]
        group_by_cols = select_cols.copy()
        joins = ["JOIN position_opera.sub_portfolio sp ON pos.sub_portfolio_id = sp.id",
                 "JOIN position_opera.portfolio pf ON sp.portfolio_id = pf.id",
                 "JOIN position_opera.security sec ON pos.risk_security_id = sec.id"]
        where_conditions = [f"pos.opera_product IN {opera_products_sql}", f"pos.tdate = {date_sql}",
                            "sp.subportfolio != 'CONSO-CT'"]

        if product == 'cotton':
            joins.append("JOIN staging.portfolio_region_mapping_table_ex prmte ON prmte.unit_sd = sp.subportfolio")
            select_cols = ["pos.tdate", "sec.security_id",  "sp.subportfolio", "pf.portfolio", "sec.strike",
                           "sec.derivative_type", "sec.product_code", "sec.contract_month", "sec.currency",
                           "prmte.region", "prmte.books"]
            group_by_cols = select_cols.copy()

            # Filter by books
            excluded_books = ['ADMIN', 'NON OIL', 'POOL']
            if books is None or books == 'all':
                # Just exclude the excluded_books
                where_conditions.append("prmte.books NOT IN (" + ", ".join(f"'{b}'" for b in excluded_books) + ")")
            else:
                # Allow books to be a string or iterable of strings
                if isinstance(books, str):
                    books = [books]  # convert to list for uniform handling

                # Always exclude excluded_books AND include only specified books
                books_sql = ", ".join(f"'{b}'" for b in books)
                excluded_books_sql = ", ".join(f"'{b}'" for b in excluded_books)
                where_conditions.append(f"prmte.books IN ({books_sql})")
                where_conditions.append(f"prmte.books NOT IN ({excluded_books_sql})")

            # Filter by region if provided
            if region is not None:
                # Assuming region filter is a list or a single value
                if isinstance(region, (list, tuple, set)):
                    region_list_sql = ", ".join(f"'{r}'" for r in region)
                    where_conditions.append(f"prmte.region IN ({region_list_sql})")
                else:
                    where_conditions.append(f"prmte.region = '{region}'")

        # Optional trader filter
        if trader_id is not None or trader_id == 'all':
            joins.append("JOIN position_opera.trader tr ON pos.trader_id = tr.id")
            select_cols += ["tr.id AS trader_id", "tr.name AS trader_name"]
            group_by_cols += ["tr.id", "tr.name"]
            if trader_id != 'all':
                where_conditions.append(f"tr.id = {trader_id}")

        # Optional counterparty filter
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

        print(query)
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
            print(2, instrument, contract)
            curve_map = instrument_dict[instrument].get('contract_to_curve_map', {})
            print(curve_map)
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

    def load_position_ref_data(self):
        pass
