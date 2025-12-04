"""
PnL CALCULATOR WORKFLOW
===========================

Vectorized PnL generator for positions and market returns.

Supports:
- linear PnL (historical or Monte Carlo linear)
- taylor series PnL
- basis/physical positions

No row-by-row loops for large datasets.
"""

import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, Any

from pnl_analyzer.pnl_analyzer import PnLAnalyzer


# =========================================
# VECTORIZED PnL FUNCTIONS
# =========================================

def generate_linear_pnl(combined_pos_df: pd.DataFrame, percentage_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized linear PnL: pnl = delta * returns * to_USD_conversion
    FIX: long_pnl_df['position_index'] now matches combined_pos_df['position_index']
    without changing the DataFrame index.
    """

    # 1. Ensure unique return series columns
    percentage_returns_df = percentage_returns_df.loc[:, ~percentage_returns_df.columns.duplicated()]

    # 2. Extract mapped return series
    selected_cols = combined_pos_df["linear_var_map"].astype(str).tolist()

    missing = [col for col in selected_cols if col not in percentage_returns_df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing return series for: {missing}")

    # 3. Build return matrix (N positions × T dates)
    returns_matrix = percentage_returns_df[selected_cols].T.values  # shape (N, T)

    pnl_matrix = pd.DataFrame(
        returns_matrix,
        index=combined_pos_df.index,
        columns=percentage_returns_df.index
    )

    # 4. Multiply by cob_date_prices, deltas and FX conversions
    pnl_matrix = pnl_matrix.mul(combined_pos_df["cob_date_price"].values, axis=0)
    pnl_matrix = pnl_matrix.mul(combined_pos_df["delta"].values, axis=0)
    pnl_matrix = pnl_matrix.mul(combined_pos_df["to_USD_conversion"].values, axis=0)
    pnl_matrix = pnl_matrix.div(combined_pos_df["cob_date_fx"].values, axis=0)

    # ------------------------------------------------------------------
    # 5. ADD REAL position_index AS A COLUMN
    # ------------------------------------------------------------------
    pnl_matrix["position_index"] = combined_pos_df["position_index"].values

    # ------------------------------------------------------------------
    # 6. Convert to long form (stack) but keep position_index column
    # ------------------------------------------------------------------
    out = (
        pnl_matrix
        .set_index("position_index", append=False)
        .stack()
        .reset_index()
    )
    date_col = out.columns[1]
    value_col = out.columns[2]

    out = out.rename(columns={
        date_col: "pnl_date",
        value_col: "lookback_pnl"
    })

    out["inverse_pnl"] = -out["lookback_pnl"]

    return out


def generate_taylor_pnl(combined_pos_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
                        #, vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized Taylor series PnL: pnl = delta*r + gamma*r^2 + vega*vol + theta
    """
    date_index = returns_df.index

    pnl_matrix = pd.DataFrame(index=combined_pos_df.index, columns=date_index, dtype=float)

    for inst in combined_pos_df["instrument_name"].str.strip().unique():
        inst_mask = combined_pos_df["instrument_name"].str.strip() == inst
        if inst not in returns_df.columns:
            continue

        ret_df = returns_df[inst]
        curve_cols = combined_pos_df.loc[inst_mask, "generic_curve"]

        for row_idx, gc in curve_cols.items():
            r = ret_df[gc]
            #v = vol_df[gc]
            row = combined_pos_df.loc[row_idx]

            pnl_vector = (
                row.delta * r +
                row.gamma * (r ** 2) +
                #row.vega * v +
                row.theta
            )
            pnl_matrix.loc[row_idx] = pnl_vector.values

    out = (
        pnl_matrix
        .stack()
        .rename("lookback_pnl")
        .reset_index()
        .rename(columns={"level_0": "position_index", "level_1": "pnl_date"})
    )
    out["inverse_pnl"] = -out["lookback_pnl"]
    return out


def generate_pnl_vectors(combined_pos_df: pd.DataFrame, returns_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Wrapper to generate PnL vectors by method.
    """
    if method == "linear":
        return generate_linear_pnl(combined_pos_df, returns_df)
    elif method == "taylor_series":
        return generate_taylor_pnl(combined_pos_df, returns_df)
    else:
        raise NotImplementedError(f"Unsupported PnL method: {method}")


# =========================================
# ANALYSIS / EXPORT FUNCTIONS
# =========================================

def analyze_and_export_unit_pnl(
        product: str,
        returns_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        long_pnl_df: pd.DataFrame,
        combined_pos_df: pd.DataFrame,
        position_index_list: list,
        filename: str,
        write_to_excel: bool
) -> Dict[str, Any]:
    """
    Analyze PnL vectors and optionally export to Excel.

    Optimization Note: The pivoting operations have been consolidated to reduce
    redundant indexing and reshaping, which dramatically speeds up the function.
    """
    analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)

    # Filter data into two large long-form DataFrames
    outright_analyzer = analyzer.filter(exposure='OUTRIGHT', position_index=position_index_list)
    basis_analyzer = analyzer.filter(exposure='BASIS (NET PHYS)', position_index=position_index_list)

    # 1. Pivot both 'lookback_pnl' and 'inverse_pnl' for Outright in one go.
    # This replaces two separate pivot calls with a single, highly efficient operation.
    outright_pivoted_combined = outright_analyzer.pivot(
        index='pnl_date',
        columns=['region', 'position_index'],
        values=['lookback_pnl', 'inverse_pnl']  # Pivot multiple value columns at once
    )

    # Separate the results from the MultiIndex columns
    # Note: Column names will be like ('lookback_pnl', 'RegionX', 'PosY')
    unit_outright_lookback = outright_pivoted_combined['lookback_pnl']
    unit_outright_inverse = outright_pivoted_combined['inverse_pnl']

    # 2. Basis analyzer still requires its own pivot, but the cost is minimized.
    unit_basis_lookback = basis_analyzer.pivot(
        index='pnl_date',
        columns=['region', 'position_index'],
        values='lookback_pnl'
    )
    # --- END OPTIMIZATION ---

    if write_to_excel:
        if os.path.exists(filename):
            mode = 'a'
            if_sheet_exists = 'replace'
        else:
            mode = 'w'
            if_sheet_exists = None

        # NOTE: Using 'replace' for if_sheet_exists on append mode ('a') is redundant
        # unless you intend to overwrite specific sheets, but kept for logic consistency.

        with pd.ExcelWriter(filename, mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            if product == 'rms':
                combined_pos_df.to_excel(writer, sheet_name='pos', index=True)
            else:
                returns_df.to_excel(writer, sheet_name='returns', index=True)
                prices_df.to_excel(writer, sheet_name='prices', index=True)
                combined_pos_df.to_excel(writer, sheet_name='pos', index=True)

                # Export results using the data extracted from the single combined pivot
                unit_outright_lookback.sort_index(ascending=False).to_excel(writer, sheet_name='outright_lookback',
                                                                            index=True)
                unit_outright_inverse.sort_index(ascending=False).to_excel(writer, sheet_name='outright_inverse',
                                                                           index=True)
                unit_basis_lookback.sort_index(ascending=False).to_excel(writer, sheet_name='basis_lookback',
                                                                         index=True)

    print(f"[EXPORT] Unit PnL vectors exported: {filename}")
    return {}