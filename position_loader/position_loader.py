from abc import ABC
import pandas as pd


class PositionLoader(ABC):

    def __init__(self, source, params=None):
        self.source = source
        self.params = params or {}

    @staticmethod
    def map_linear_risk_factor(row):
        # Priority: basis_series > generic_curve > instrument_name
        if pd.notna(row.get('basis_series')) and row.get('basis_series') != '' and row.get('return_type') == 'absolute':
            return row['basis_series'] + '_abs'
        elif 'generic_curve' in row and pd.notna(row.get('generic_curve')) and row.get('generic_curve') != '':
            return row['generic_curve']
        else:
            return row.get('instrument_name')

    def assign_linear_risk_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['risk_factor'] = df.apply(self.map_linear_risk_factor, axis=1)

        # Optional: show warnings or success
        missing = df[df['risk_factor'].isna() | (df['risk_factor'] == '')]
        if not missing.empty:
            print(f"Warning: Found {len(missing)} rows with empty 'risk_factor'.")
        else:
            print("All rows successfully mapped to 'risk_factor'.")

        return df

    @staticmethod
    def map_risk_factor(row) -> str:
        """
        Map a single row to a monte_carlo_var_risk_factor.
        """
        exposure = row.get('exposure')
        position_type = row.get('position_type')
        region = row.get('region')
        generic_curve = row.get('generic_curve')
        basis_series = row.get('basis_series')

        if exposure == 'OUTRIGHT':
            if position_type in ['FIXED PHYS', 'DIFF PHYS'] and region == 'INDIA':
                return 'EX GIN S6'
            return generic_curve

        elif exposure == 'BASIS (NET PHYS)':
            return basis_series  # COTLOOK leg by default

        return None

    def assign_risk_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign the monte_carlo_var_risk_factor column to the DataFrame.
        """
        df = df.copy()
        df['risk_factor'] = df.apply(self.map_risk_factor, axis=1)
        return df

    def duplicate_basis_and_assign_ct1(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Select BASIS rows to duplicate
        basis_rows = df[df['exposure'] == 'BASIS (NET PHYS)'].copy()
        if basis_rows.empty:
            print("No BASIS rows to duplicate.")
            return df

        # Reassign risk factor to CT1 in duplicate
        ct_leg = basis_rows.copy()
        ct_leg['risk_factor'] = 'CT1'
        ct_leg['delta'] = -basis_rows['delta']
        ct_leg['delta_exposure'] = 0

        # Mark cotlook/ct legs for indexing
        basis_rows['leg_type'] = 'COTLOOK'
        ct_leg['leg_type'] = 'CT'

        # Combine back
        df = df[~((df['exposure'] == 'BASIS (NET PHYS)'))]  # remove original basis
        combined_df = pd.concat([df, basis_rows, ct_leg], ignore_index=True)

        print(f"Duplicated {len(basis_rows)} BASIS rows into CT legs.")
        return combined_df

    @staticmethod
    def map_rubber_unit(row):
        portfolio = row['portfolio']

        if not isinstance(portfolio, str):
            return 'NULL'

    def assign_cob_date_price(self, position_df: pd.DataFrame, price_df: pd.DataFrame, cob_date: str) -> pd.DataFrame:
        """
        Assign cob_date_price to each position row for a single cob_date.

        Args:
            position_df: DataFrame with columns ['generic_curve', ...].
            price_df: pivot-like DataFrame, index=cob_date, columns=risk_factor.
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
        position_df['cob_date_price'] = position_df['risk_factor'].map(prices_for_date)

        # Detect missing prices
        missing = position_df['cob_date_price'].isna()
        if missing.any():
            print(f"[WARN] {missing.sum()} positions missing cob_date_price:")
            print(position_df[missing][['risk_factor']])
        else:
            print("All positions mapped to cob_date_price.")

        return position_df