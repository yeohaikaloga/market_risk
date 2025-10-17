from abc import ABC
import pandas as pd


class PositionLoader(ABC):

    def __init__(self, source, params=None):
        self.source = source
        self.params = params or {}

    @staticmethod
    def map_linear_var(row):
        # Priority: basis_series > generic_curve > instrument_name
        if pd.notna(row.get('basis_series')) and row.get('basis_series') != '':
            return row['basis_series']
        elif 'generic_curve' in row and pd.notna(row.get('generic_curve')) and row.get('generic_curve') != '':
            return row['generic_curve']
        else:
            return row.get('instrument_name')

    def assign_linear_var_map(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['linear_var_map'] = df.apply(self.map_linear_var, axis=1)

        # Optional: show warnings or success
        missing = df[df['linear_var_map'].isna() | (df['linear_var_map'] == '')]
        if not missing.empty:
            print(f"Warning: Found {len(missing)} rows with empty 'linear_var_map'.")
        else:
            print("All rows successfully mapped to 'linear_var_map'.")

        return df

    @staticmethod
    def map_monte_carlo_var_risk_factor(row) -> str:
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

    def assign_monte_carlo_var_risk_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign the monte_carlo_var_risk_factor column to the DataFrame.
        """
        df = df.copy()
        df['monte_carlo_var_risk_factor'] = df.apply(self.map_monte_carlo_var_risk_factor, axis=1)
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
        ct_leg['monte_carlo_var_risk_factor'] = 'CT1'
        ct_leg['delta'] = -basis_rows['delta']

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

