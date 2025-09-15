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
    basis_abs_ret_df: pd.DataFrame,
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
    if method != 'linear':
        raise NotImplementedError(f"Method '{method}' not yet supported.")

    # Add position_index if not exists
    if 'position_index' not in combined_pos_df.columns:
        product = combined_pos_df['product'].iloc[0] if 'product' in combined_pos_df.columns else 'UNK'
        cob_date = combined_pos_df['cob_date'].iloc[0] if 'cob_date' in combined_pos_df.columns else 'UNK'
        combined_pos_df['position_index'] = (
            product[:3] + '_L_' + str(cob_date) + '_' +
            combined_pos_df.index.map(lambda i: str(i).zfill(4))
        )

    pnl_dfs = []

    for idx, row in combined_pos_df.iterrows():
        instrument_name = row['instrument_name']
        generic_curve = row['generic_curve']
        basis_series = row['basis_series']
        position_index = row['position_index']
        delta = row['delta']
        exposure = row['exposure']
        to_usd = row['to_USD_conversion']

        if exposure == 'OUTRIGHT':
            # Get $ returns for this generic curve
            returns_series = instrument_dict[instrument_name]['relative_returns_$_df'][generic_curve]

            # Calculate PnL
            pnl_series = delta * returns_series * to_usd

        else:
            basis_returns_series = basis_abs_ret_df[basis_series]

            # Calculate PnL
            pnl_series = delta * basis_returns_series * to_usd


        # Build DataFrame for this position
        df = pnl_series.to_frame(name='lookback_pnl')
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

    unit_outright_lookback = outright_analyzer.pivot(index='pnl_date', columns=['region','position_index'], values='lookback_pnl')
    unit_outright_inverse = outright_analyzer.pivot(index='pnl_date', columns=['region','position_index'], values='inverse_pnl')
    unit_basis_lookback = basis_analyzer.pivot(index='pnl_date', columns=['region','position_index'], values='lookback_pnl')

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

    print(f"Export for unit pnl vectors completed: f'{filename}'")
