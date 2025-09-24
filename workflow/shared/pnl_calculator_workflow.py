"""
PnL CALCULATION WORKFLOW
========================

Generates and analyzes PnL vectors from positions and market returns.

Uses PnLAnalyzer class internally for safe merging, filtering, and pivoting.

Returns dict with precomputed views for VaR, Stress, Attribution workflows.
"""

import pandas as pd
from typing import Dict, Any
import os

from pnl_analyzer.pnl_analyzer import PnLAnalyzer

def generate_pnl_vectors(
    combined_pos_df: pd.DataFrame,
    instrument_dict: Dict[str, Any],
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Generate daily PnL vectors for each position.

    For linear method:
        PnL = delta * $returns * to_USD_conversion

    Args:
        combined_pos_df: Must have: position_index, delta, instrument_name, generic_curve, to_USD_conversion
        instrument_dict: From data_preparation — contains relative_returns_$_df per instrument
        method: 'linear' (default), 'non-linear' (TBC)

    Returns:
        Long-format DataFrame with columns:
        - pnl_date
        - position_index
        - lookback_pnl
        - inverse_pnl ( = -lookback_pnl )
        - cob_date
        - method
    """

    # Add position_index if not exists
    if 'position_index' not in combined_pos_df.columns:
        combined_pos_df['position_index'] = (
                combined_pos_df['product'].str[:3] + '_L_' +
                combined_pos_df['cob_date'].astype(str) + '_' +
                combined_pos_df.index.map(lambda i: str(i).zfill(4))
        )
    pnl_dfs = []

    for idx, row in combined_pos_df.iterrows():
        instrument_name = row['instrument_name']
        #generic_curve = row['generic_curve']
        #basis_series = row['basis_series']
        if method == 'linear':
            linear_var_map = row['linear_var_map']
        elif method == 'non-linear (monte carlo)':
            monte_carlo_risk_factor = row['monte_carlo_var_risk_factor']
        position_index = row['position_index']
        delta = row['delta']
        exposure = row['exposure']
        to_usd = row['to_USD_conversion']

        if exposure == 'OUTRIGHT':
            if instrument_name in instrument_dict.keys():
                if instrument_name == 'VV':
                    if method == 'linear':
                        returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][linear_var_map]
                        print(exposure, position_index, instrument_name, linear_var_map, returns_series.tail())
                    elif method == 'non-linear (monte carlo)':
                        returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][monte_carlo_risk_factor]

                    returns_series = returns_series / 7.1675
                    print(exposure, instrument_name, returns_series.tail())

                elif instrument_name == 'CCL':
                    if method == 'linear':
                        returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][linear_var_map]
                        print(exposure, position_index, instrument_name, linear_var_map, returns_series.tail())
                    elif method == 'non-linear (monte carlo)':
                        returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][monte_carlo_risk_factor]

                    returns_series = returns_series / 87.5275
                    print(exposure, instrument_name, returns_series.tail())

                else:
                    # Get $ returns for this generic curve
                    print(exposure, instrument_name, position_index)
                    if method == 'linear':
                        returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][linear_var_map]
                    elif method == 'non-linear (monte carlo)':
                        returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][monte_carlo_risk_factor]

            else:
                returns_series = instrument_dict['PHYS'][instrument_name]['relative_returns_$']
                print(exposure, position_index, instrument_name, returns_series.tail())
                returns_series = returns_series / 87.5275
                print(exposure, instrument_name, returns_series.tail())

            # Calculate PnL
            pnl_series = delta * returns_series * to_usd
            df = pnl_series.to_frame(name='lookback_pnl')

        elif exposure == 'BASIS (NET PHYS)':
            if method == 'linear':
                basis_returns_series = instrument_dict['BASIS']['abs_returns_$_df'][linear_var_map]
                pnl_series = delta * basis_returns_series * to_usd
                df = pnl_series.to_frame(name='lookback_pnl')
            elif method == 'non-linear (monte carlo)':
                if monte_carlo_risk_factor == 'CT1':
                    returns_series = instrument_dict['CT']['relative_returns_$_df'][monte_carlo_risk_factor]
                else:
                    returns_series = instrument_dict['PHYS']['COTLOOK'][monte_carlo_risk_factor][monte_carlo_risk_factor]
                pnl_series = delta * returns_series * to_usd
                df = pnl_series.to_frame(name='lookback_pnl')

        else:
            print(exposure, position_index, '[WARNING] Exposure is not Outright or Basis (Net Phys)')

        # Build DataFrame for this position

        df['inverse_pnl'] = -df['lookback_pnl']
        df['position_index'] = position_index
        df['cob_date'] = row.get('cob_date', None)
        df['method'] = method
        df['pnl_date'] = df.index

        pnl_dfs.append(df)

    if not pnl_dfs:
        raise ValueError("No PnL vectors generated.")

    long_pnl_df = pd.concat(pnl_dfs, ignore_index=True)
    return long_pnl_df



def analyze_and_export_unit_pnl(
    long_pnl_df: pd.DataFrame,
    combined_pos_df: pd.DataFrame,
    position_index_list: list,
    filename: str,
    write_to_excel: bool
) -> Dict[str, Any]:
    # Initialize analyzer — handles merge + validation internally
    analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)

    # Filter by exposure
    outright_analyzer = analyzer.filter(exposure='OUTRIGHT', position_index=position_index_list)
    basis_analyzer = analyzer.filter(exposure='BASIS (NET PHYS)', position_index=position_index_list)

    unit_outright_lookback = outright_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'], values='lookback_pnl')
    unit_outright_inverse = outright_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'], values='inverse_pnl')
    unit_basis_lookback = basis_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'], values='lookback_pnl')
    #unit_basis_inverse = basis_analyzer.pivot(index='pnl_date', columns=['region', 'position_index'], values='inverse_pnl')

    if write_to_excel:
        if os.path.exists(filename):
            mode = 'a'
            if_sheet_exists = 'replace'
        else:
            mode = 'w'
            if_sheet_exists = None
        with pd.ExcelWriter(filename, mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            combined_pos_df.to_excel(writer, sheet_name='pos', index=True)
            unit_outright_lookback.sort_index(ascending=False).to_excel(writer, sheet_name='outright_lookback', index=True)
            unit_outright_inverse.sort_index(ascending=False).to_excel(writer, sheet_name='outright_inverse', index=True)
            unit_basis_lookback.sort_index(ascending=False).to_excel(writer, sheet_name='basis_lookback', index=True)
            #unit_basis_inverse.sort_index(ascending=False).to_excel(writer, sheet_name='basis_inverse', index=True)

    print(f"Export for unit pnl vectors completed: f'{filename}'")
