from position_loader.position_loader import PositionLoader
import pandas as pd
from sqlalchemy import text
from utils.contract_utils import get_month_code, custom_monthly_contract_sort_key
from datetime import datetime

fy24_unit_to_cotlook_basis_origin_dict = {'USA EQUITY': 'Memphis/Orleans/Texas', 'USA': 'Memphis/Orleans/Texas',
                                          'BRAZIL': 'Brazilian', 'SECO': "Ivory Coast Manbo/s",
                                          'W. AFRICA': "Burkina Faso Bola/s", 'TOGO': "Burkina Faso Bola/s",
                                          'CHAD': "Burkina Faso Bola/s", 'E. AFRICA': "Burkina Faso Bola/s"}


class PhysicalPositionLoader(PositionLoader):

    def __init__(self, date, source):
        super().__init__(source)
        self.source = source
        self.date = date

    def load_position(self, date, opera_product, trader_id=None, counterparty_id=None) -> pd.DataFrame:
        # TODO load physical position from master physical position table; still WIP.
        base_query = '''
                SELECT pos.cob_date, sp.subportfolio, pf.portfolio, sec.security_id, sec.strike, sec.derivative_type,
                       tr.id AS trader_id, tr.name AS trader_name,
                       cp.id AS counterparty_id, cp.counterparty_parent,
                       SUM(pos.total_active_lots) AS total_active_lots
                FROM position_opera.position_loader pos
                JOIN position_opera.sub_portfolio sp ON pos.sub_portfolio_id = sp.id
                JOIN position_opera.portfolio pf ON sp.portfolio_id = pf.id
                JOIN position_opera.counterparty cp ON pos.counterparty_id = cp.id
                JOIN position_opera.trader tr ON pos.trader_id = tr.id
                JOIN position_opera.security sec ON pos.risk_security_id = sec.id
                WHERE pos.opera_product = :opera_product
                  AND pos.cob_date = :date
                  AND sp.subportfolio != 'CONSO-CT'
            '''

        # Prepare parameters dictionary
        params = {
            "opera_product": opera_product,
            "date": date,
        }

        # Add optional filters
        if trader_id is not None:
            base_query += " AND tr.id = :trader_id"
            params["trader_id"] = trader_id

        if counterparty_id is not None:
            base_query += " AND cp.id = :counterparty_id"
            params["counterparty_id"] = counterparty_id

        # Add GROUP BY clause
        base_query += '''
                GROUP BY pos.cob_date, sp.subportfolio, pf.portfolio, sec.security_id, sec.strike, sec.derivative_type,
                         tr.id, tr.name, cp.id, cp.counterparty_parent
            '''

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(base_query), conn, params=params)

        return df

    def _load_filtered_cotton_positions(self, cob_date: str) -> pd.DataFrame:
        query = 'SELECT * FROM staging.cotton_physical_positions WHERE quantity != 0'
        print(query)
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        print(df.head())
        # Apply consistent filters
        df = df[df['cob'] == cob_date]
        df = df[df['UNIT'] != 'TOTAL']
        return df

    def load_cotton_phy_position_from_staging(self, cob_date: str) -> pd.DataFrame:
        df = self._load_filtered_cotton_positions(cob_date)
        return df

    def _load_filtered_rubber_positions(self, cob_date: str) -> pd.DataFrame:
        query = """
            SELECT * 
            FROM staging.ors_positions_updated
            """
        print(query)
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        # Apply consistent filters
        df['Delta Quantity'] = pd.to_numeric(df['Delta Quantity'], errors='coerce')
        df = df[df['Delta Quantity'] != 0.0]
        cob_date = cob_date + ' 00:00:00'
        df= df[df['tdate'] == cob_date]
        print(df.head())
        return df

    def load_rubber_phy_position_from_staging(self, cob_date: str) -> pd.DataFrame:
        df = self._load_filtered_rubber_positions(cob_date)
        return df

    @staticmethod
    def map_bbg_tickers(instrument_name: str, terminal_month: datetime, instrument_ref_dict: dict) -> tuple[str | None, str | None]:
        """
        Returns both Bloomberg-style option ticker and underlying ticker.
        Returns: (option_ticker, underlying_ticker)
        """
        try:
            if (not isinstance(instrument_name, str) or instrument_name not in instrument_ref_dict.keys()
                    or not isinstance(terminal_month, datetime)):
                return None, None

            month = terminal_month.month
            year = str(terminal_month.year)[-1]  # last digit of year as string
            month_code = get_month_code(month)
            if month_code is None:
                return None, None

            bbg_ticker = instrument_name + month_code + year + ' Comdty'
            return bbg_ticker, bbg_ticker

        except Exception:
            return None, None

    def assign_bbg_tickers(self, df: pd.DataFrame, instrument_ref_dict: dict) -> pd.DataFrame:
        df = df.copy()
        df[['bbg_ticker', 'underlying_bbg_ticker']] = df.apply(
            lambda row: pd.Series(self.map_bbg_tickers(row['instrument_name'], row['terminal_month'], instrument_ref_dict)), axis=1)

        for col in ['bbg_ticker', 'underlying_bbg_ticker']:
            invalid = df[df[col].isna()]
            if not invalid.empty:
                print(f"Warning: Found {len(invalid)} invalid instrument_name or terminal_month for column '{col}':")
                print(invalid[['instrument_name', 'terminal_month']])
            else:
                print(f"All instrument_name and terminal_month successfully mapped to '{col}'.")
        return df

    @staticmethod
    def map_generic_curve_with_fallback(contract, contract_to_curve_map):
        if contract in contract_to_curve_map:
            return contract_to_curve_map[contract]

        sorted_contracts = sorted(contract_to_curve_map.keys(), key=custom_monthly_contract_sort_key)

        for c in sorted_contracts:
            if custom_monthly_contract_sort_key(c) > custom_monthly_contract_sort_key(contract):
                return contract_to_curve_map[c]

        # No later contract found — use the last available
        print(contract, sorted_contracts)
        if len(sorted_contracts) > 0:
            last_contract = sorted_contracts[-1]
            return contract_to_curve_map[last_contract]
        else:
            return None

    @staticmethod
    def map_generic_curve(row, instrument_dict):
        underlying = str(row.get('underlying_bbg_ticker', ''))
        contract = underlying.replace(' Comdty', '').upper()

        instrument_name = str(row.get('instrument_name', '')).upper()
        instrument_info = instrument_dict.get(instrument_name)
        if not instrument_info or not isinstance(instrument_info, dict):
            print(f"[WARN] Instrument '{instrument_name}' not found or invalid in instrument_dict.")
            return None

        curve_map = instrument_info.get('contract_to_curve_map', {})

        if contract in curve_map:
            return curve_map[contract]

        # Fallback: next available or last
        fallback = PhysicalPositionLoader.map_generic_curve_with_fallback(contract, curve_map)
        if fallback:
            print(f"[Fallback] {contract} mapped to {fallback} under instrument {instrument_name} for {row}")
            return fallback

        print(f"[WARN] No mapping or fallback found for contract {contract} under instrument {instrument_name}")
        return None

    def assign_generic_curves(self, df: pd.DataFrame, instrument_dict: dict) -> pd.DataFrame:
        df = df.copy()
        df['generic_curve'] = df.apply(lambda row: self.map_generic_curve(row, instrument_dict), axis=1)

        invalid = df[df['generic_curve'].isna()]
        if not invalid.empty:
            print(f"Warning: {len(invalid)} positions could not be mapped to a generic curve.")
            print(invalid[['region', 'subportfolio', 'underlying_bbg_ticker']])
        else:
            print("All positions successfully mapped to a generic curve.")
        return df

    @staticmethod
    def map_basis_series(row, mapping_dict):
        region = str(row.get('region', ''))
        exposure = str(row.get('exposure', ''))
        if exposure == 'BASIS (NET PHYS)':
            result = mapping_dict.get(region)
            if result is None:
                result = 'A Index'
            return result
        return None

    def assign_basis_series(self, df: pd.DataFrame, dict) -> pd.DataFrame:
        df = df.copy()
        df['basis_series'] = df.apply(lambda row: self.map_basis_series(row, fy24_unit_to_cotlook_basis_origin_dict), axis=1)
        missing = df[df['basis_series'].isna()]
        if not missing.empty:
            print(f"Warning: {len(missing)} rows have no basis_series mapping.")
            print(missing[['region', 'exposure']])
        else:
            print("All rows successfully mapped to basis_series.")

        return df
