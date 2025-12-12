from price_series_generator.price_series_generator import PriceSeriesGenerator
import pandas as pd
from typing import List
from utils.file_utils import load_from_pickle_in_dir



class SimulatedReturnsSeriesGenerator(PriceSeriesGenerator):

    def __init__(self, returns_df: pd.DataFrame, cob_date: str):
        super().__init__(returns_df)
        self.source = 'Grains_DB'
        self.cob_date = cob_date

    @staticmethod
    def filter_relevant_risk_factors(simulated_returns_df: pd.DataFrame, relevant_risk_factors: List[str]) -> pd.DataFrame:
        """
        Filters a DataFrame by selecting columns whose prefix (part before the first '_')
        is present in the list of relevant_tickers.

        Args:
            simulated_returns_df: The DataFrame of simulated returns (e.g., dict2["simulatedRet_df_ld"]).
            relevant_tickers: A list of base risk factor codes (e.g., ['CT', 'VV']).

        Returns:
            A new DataFrame containing only the filtered columns.
        """

        # Use a set for faster lookup time complexity (O(1) instead of O(N))
        risk_factor_set = set(relevant_risk_factors)

        # 1. Identify matching column names using list comprehension
        # col.split('_')[0] extracts the prefix (e.g., 'CT' from 'CT_H26_USD')
        filtered_cols = [
            col for col in simulated_returns_df.columns
            if col.split('_')[0] in risk_factor_set
        ]

        # 2. Return the filtered DataFrame
        print(f"Original columns: {simulated_returns_df.columns}")
        print(f"Filtered columns: {filtered_cols}")
        return simulated_returns_df[filtered_cols]

    @staticmethod
    def rename_simulated_columns(simulated_returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns in the filtered DataFrame sequentially based on their ticker prefix.
        Example: 'CT_H26_USD' -> 'CT1', 'CT_K26_USD' -> 'CT2', etc.

        Args:
            simulated_returns_df: The DataFrame containing returns of risk factor.

        Returns:
            The DataFrame with the new, sequenced column names.
        """
        new_names_map = {}

        # Group columns by prefix (e.g., 'CT', 'VV', 'JN') while maintaining order
        # Note: The order of columns in the DataFrame must be preserved for sequencing (CT1, CT2, ...)
        grouped_columns = {}
        for col in simulated_returns_df.columns:
            prefix = col.split('_')[0]
            if len(prefix) == 1:
                prefix = prefix + ' '
            if prefix not in grouped_columns:
                grouped_columns[prefix] = []
            grouped_columns[prefix].append(col)

        # Generate the new sequential names for each group
        for prefix, cols in grouped_columns.items():
            # cols is already in the correct order for sequencing
            for i, original_col_name in enumerate(cols, 1):
                new_name = f"{prefix}{i}"
                new_names_map[original_col_name] = new_name

        # Apply the renaming
        renamed_df = simulated_returns_df.rename(columns=new_names_map)
        grains_db_to_grid_map = {'AIndex1': 'A Index', 'MeOrTe1': 'Memphis/Orleans/Texas',
                                 'IvCoMa1': 'Ivory Coast Manbo/s', 'BuFaBo1': 'Burkina Faso Bola/s',
                                 'BrCo1': 'Brazilian', 'Shankar61': 'EX GIN S6', 'GaSu1': 'GARMMZ SUGAR',
                                 'MaizeUP1': 'MAIZE UP', 'SawnAvg1': 'WOOD AVG'}
        renamed_df = renamed_df.rename(columns=grains_db_to_grid_map)
        print(f"Columns renamed to sequential format: {list(new_names_map.values())[:10]}...")

        return renamed_df

    @classmethod
    def load_relevant_simulated_returns(cls, cob_date: str, filename: str, relevant_risk_factors: List[str]):
        """
        Class method (factory) to load simulated data using the file_utils helper,
        filter, rename columns, and return a new instance of SimulatedReturnsSeriesGenerator.

        Args:
            cob_date (str): The COB date in 'YYYY-MM-DD' format.
            filename (str): The filename of the pickle file (e.g., 'daily_simulated_matrix_20251210.pickle').
            relevant_risk_factors (List[str]): The list of base risk factors to filter by (e.g., ['CT', 'VV']).
        """
        dict_file = load_from_pickle_in_dir(cob_date, filename)
        returns_df = dict_file["simulatedRet_df_ld"]
        returns_df = cls.filter_relevant_risk_factors(returns_df, relevant_risk_factors)
        returns_df = cls.rename_simulated_columns(returns_df)
        return cls(returns_df, cob_date)

