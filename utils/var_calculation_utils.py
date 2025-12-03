import pandas as pd
import numpy as np
from typing import List
from financial_calculations.var_calculator import VaRCalculator
from pnl_analyzer.pnl_analyzer import PnLAnalyzer


def calculate_var_for_regions(
    var_data_df: pd.DataFrame,
    analyzer: PnLAnalyzer,
    cob_date: str,
    window: int,
    percentiles: List[int] = [95, 99]
) -> pd.DataFrame:
    var_calc = VaRCalculator()

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

        # Combine position indexes for overall calculation (union or concat)
        overall_positions = list(set(outright_positions) | set(basis_positions))

        # Filter analyzers for outright, basis, and overall
        outright_analyzer = analyzer.filter(position_index=outright_positions)
        basis_analyzer = analyzer.filter(position_index=basis_positions)
        overall_analyzer = analyzer.filter(position_index=overall_positions)

        def safe_pivot(_analyzer, value):
            if _analyzer is not None:
                return _analyzer.pivot(index='pnl_date', columns='position_index', values=value)
            return pd.DataFrame()

        outright_lookback = safe_pivot(outright_analyzer, 'lookback_pnl')
        outright_inverse = safe_pivot(outright_analyzer, 'inverse_pnl')

        basis_lookback = safe_pivot(basis_analyzer, 'lookback_pnl')
        basis_inverse = safe_pivot(basis_analyzer, 'inverse_pnl')

        overall_lookback = safe_pivot(overall_analyzer, 'lookback_pnl')
        overall_inverse = safe_pivot(overall_analyzer, 'inverse_pnl')

        def safe_position_sum(_analyzer, positions):
            if _analyzer is not None and _analyzer.position_df is not None and not _analyzer.position_df.empty:
                return _analyzer.position_df[_analyzer.position_df['position_index'].isin(positions)]['delta'].sum()
            return 0

        outright_pos = safe_position_sum(outright_analyzer, outright_positions)
        basis_pos = safe_position_sum(basis_analyzer, basis_positions)
        overall_pos = safe_position_sum(overall_analyzer, overall_positions)

        var_data_df.at[i, 'outright_pos'] = int(outright_pos)
        var_data_df.at[i, 'basis_pos'] = int(basis_pos)
        var_data_df.at[i, 'overall_pos'] = int(overall_pos)


        # Calculate VaR per percentile for each exposure
        for p in percentiles:
            outright_var = var_calc.calculate_historical_var(
                outright_lookback, outright_inverse, cob_date, p, window
            ) if outright_positions else 0

            basis_var = var_calc.calculate_historical_var(
                basis_lookback, basis_inverse, cob_date, p, window
            ) if basis_positions else 0

            overall_var = var_calc.calculate_historical_var(
                overall_lookback, overall_inverse, cob_date, p, window
            ) if overall_positions else 0
            var_data_df.at[i, f'outright_{p}_var'] = int(outright_var) if not pd.isna(outright_var) else None
            var_data_df.at[i, f'basis_{p}_var'] = int(basis_var) if not pd.isna(basis_var) else None
            var_data_df.at[i, f'overall_{p}_var'] = int(overall_var) if not pd.isna(overall_var) else None

    return var_data_df
