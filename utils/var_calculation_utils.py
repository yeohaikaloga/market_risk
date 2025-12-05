import pandas as pd
import numpy as np
from typing import List, Dict
from financial_calculations.var_calculator import VaRCalculator
from pnl_analyzer.pnl_analyzer import PnLAnalyzer


def _get_pnl_pivots(analyzer: PnLAnalyzer, values: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Pivots the PnL data once outside the main aggregation loop for efficiency.
    Returns a dictionary of pivoted DataFrames keyed by the value column name.
    """
    if not analyzer or analyzer._pos_and_pnl_df.empty:
        return {v: pd.DataFrame() for v in values}

    pivots = {}

    # We create a single, efficient pivot on the entire dataset
    full_pnl_df = analyzer._pos_and_pnl_df.pivot(
        index='pnl_date',
        columns='position_index',
        values=values
    )

    # The resulting column names are a MultiIndex (value, position_index). We flatten this.
    # Example: ('lookback_pnl', 1234), ('lookback_pnl', 5678)
    # We rename columns for easier slicing, e.g., 'lookback_pnl_1234'

    for v in values:
        pivots[v] = full_pnl_df[v]

    return pivots


def calculate_var_for_regions(var_data_df: pd.DataFrame, analyzer: PnLAnalyzer, simulation_method: str,
                              calculation_method: str, cob_date: str, window: int, percentiles: List[int] = [95, 99]) \
        -> pd.DataFrame:
    var_calc = VaRCalculator()
    pos_and_pnl_df = analyzer._pos_and_pnl_df
    pos_df = analyzer.position_df

    # Step 1: Pivot for all PnL vectors
    pnl_pivots = _get_pnl_pivots(analyzer, ['lookback_pnl', 'inverse_pnl'])
    full_lookback = pnl_pivots.get('lookback_pnl', pd.DataFrame())
    full_inverse = pnl_pivots.get('inverse_pnl', pd.DataFrame())

    # Handle the case where the full PnL matrix might be empty
    if full_lookback.empty or full_inverse.empty:
        print("[WARNING]: Full PnL matrix is empty. Cannot calculate VaR.")
        return var_data_df

    # Add placeholder columns for var percentiles and positions
    for p in percentiles:
        var_data_df[f'outright_{p}_var'] = np.nan
        var_data_df[f'basis_{p}_var'] = np.nan
        var_data_df[f'overall_{p}_var'] = np.nan

    var_data_df['outright_pos'] = np.nan
    var_data_df['basis_pos'] = np.nan
    var_data_df['overall_pos'] = np.nan

    for i, row in var_data_df.iterrows():
        outright_positions = row['outright_position_index_list'] or []
        basis_positions = row['basis_position_index_list'] or []
        overall_positions = list(set(outright_positions) | set(basis_positions))

        # Step 2: Position Calculation
        outright_pos = 0
        if outright_positions and len(outright_positions) > 0:
            outright_pos = pos_df[pos_df['position_index'].isin(outright_positions)]['exposure_delta'].sum()

        basis_pos = 0
        if basis_positions and len(basis_positions) > 0:
            basis_pos = pos_df[pos_df['position_index'].isin(basis_positions)]['exposure_delta'].sum()

        overall_pos = 0
        if overall_positions and len(overall_positions) > 0:
            overall_pos = pos_df[pos_df['position_index'].isin(overall_positions)]['exposure_delta'].sum()

        var_data_df.at[i, 'outright_pos'] = int(outright_pos)
        var_data_df.at[i, 'basis_pos'] = int(basis_pos)
        var_data_df.at[i, 'overall_pos'] = int(overall_pos)

        # Step 3: VaR Data Preparation (Vectorized Slicing):
        outright_lookback = full_lookback[outright_positions] if outright_positions else pd.DataFrame()
        outright_inverse = full_inverse[outright_positions] if outright_positions else pd.DataFrame()

        basis_lookback = full_lookback[basis_positions] if basis_positions else pd.DataFrame()
        basis_inverse = full_inverse[basis_positions] if basis_positions else pd.DataFrame()

        overall_lookback = full_lookback[overall_positions] if overall_positions else pd.DataFrame()
        overall_inverse = full_inverse[overall_positions] if overall_positions else pd.DataFrame()

        # Step 4: Calculate VaR per percentile for each exposure
        for p in percentiles:

            outright_var = var_calc.calculate_var(
                outright_lookback, outright_inverse, simulation_method, calculation_method, cob_date, p, window
            ) if not outright_lookback.empty else 0

            basis_var = var_calc.calculate_var(
                basis_lookback, basis_inverse, simulation_method, calculation_method, cob_date, p, window
            ) if not basis_lookback.empty else 0

            overall_var = var_calc.calculate_var(
                overall_lookback, overall_inverse, simulation_method, calculation_method, cob_date, p, window
            ) if not overall_lookback.empty else 0

            var_data_df.at[i, f'outright_{p}_var'] = int(outright_var) if not pd.isna(outright_var) else None
            var_data_df.at[i, f'basis_{p}_var'] = int(basis_var) if not pd.isna(basis_var) else None
            var_data_df.at[i, f'overall_{p}_var'] = int(overall_var) if not pd.isna(overall_var) else None

    return var_data_df
