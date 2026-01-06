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
from typing import Dict, Any
from utils.log_utils import get_logger
from pnl_analyzer.pnl_analyzer import PnLAnalyzer
from db.db_connection import get_engine
from sensitivity_matrix_loader.sensitivity_matrix_loader import SensitivityMatrixLoader


# =========================================
# VECTORIZED PnL FUNCTIONS
# =========================================

def generate_linear_pnl(combined_pos_df: pd.DataFrame, percentage_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized linear PnL: pnl = delta * returns * to_USD_conversion
    FIX: long_pnl_df['position_index'] now matches combined_pos_df['position_index']
    without changing the DataFrame index.
    """
    logger = get_logger(__name__)
    # 1. Ensure unique return series columns
    percentage_returns_df = percentage_returns_df.loc[:, ~percentage_returns_df.columns.duplicated()]

    # 2. Extract mapped return series
    selected_cols = combined_pos_df["risk_factor"].astype(str).tolist()

    missing_cols = [col for col in selected_cols if col not in percentage_returns_df.columns]
    if missing_cols:
        unique_missing = list(set(missing_cols))
        percentage_returns_df[unique_missing] = 0.0
        logger.warning(f"Risk factors missing from returns data. Defaulting to zero returns for: "
                       f"{unique_missing}")

    # Crate temporary None mapping as rubber basis VaR not being calculated
    returns_matrix = percentage_returns_df[selected_cols].T.values  # shape (N, T)
    logger.info('STEP 3-1: Returns matrix prepared')
    pnl_matrix = pd.DataFrame(
        returns_matrix,
        index=combined_pos_df.index,
        columns=percentage_returns_df.index
    )
    logger.info('STEP 3-2: PnL matrix calculated')
    # 4. Multiply by cob_date_prices, deltas and FX conversions
    cob_date_price_multiplier_condition = np.where(combined_pos_df["return_type"] == 'relative',
                                                   combined_pos_df["cob_date_price"], 1.0)
    pnl_matrix = pnl_matrix.mul(cob_date_price_multiplier_condition, axis=0)
    pnl_matrix = pnl_matrix.mul(combined_pos_df["delta"].values, axis=0)
    pnl_matrix = pnl_matrix.mul(combined_pos_df["to_USD_conversion"].values, axis=0)
    pnl_matrix = pnl_matrix.div(combined_pos_df["cob_date_fx"].values, axis=0)
    logger.info('STEP 3-3: Conversions completed')
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


def generate_taylor_series_pnl(combined_pos_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Refactored Taylor series PnL calculation.
    Expected returns_df shape: Index = Dates, Columns = Generic Curves (or Risk Factors)
    """
    # 1. Ensure instrument names and curve names are clean
    pos_df = combined_pos_df.copy()
    pos_df["generic_curve"] = pos_df["generic_curve"].str.strip()

    # 2. Filter for curves that actually exist in the returns data
    available_curves = pos_df["generic_curve"].unique()
    valid_curves = [c for c in available_curves if c in returns_df.columns]

    if not valid_curves:
        # Return empty DataFrame with expected schema if no matches
        return pd.DataFrame(columns=["position_index", "pnl_date", "lookback_pnl", "inverse_pnl"])

    # 3. Vectorized Calculation over the whole returns matrix for relevant curves
    # R: [Dates x Valid Curves]
    R = returns_df[valid_curves]
    R2 = R ** 2

    all_pnl_parts = []

    # We iterate over the positions once, but the PnL calculation for each row is a
    # vectorized operation across the entire date index of R.
    for idx, row in pos_df.iterrows():
        gc = row["generic_curve"]
        position_index = row["position_index"]

        if gc not in R.columns:
            continue

        # Extract the returns vector for this specific generic curve
        r_vector = R[gc]
        r2_vector = R2[gc]

        # pnl = delta * r + gamma * r^2 + theta
        # Note: theta is usually daily, so we add it to every scenario date
        #abs return: pnl_series = (row.delta * r_vector) + (row.gamma * r2_vector)
        delta_times_price = row.delta * row.cob_date_price
        gamma_times_price_sq = 0.5 * row.gamma * row.cob_date_price ** 2
        pnl_series = (
                delta_times_price * r_vector
                + gamma_times_price_sq * r2_vector
        )

        # Create a temporary DF to collect results
        tmp = pd.DataFrame({
            "position_index": position_index,
            "pnl_date": pnl_series.index,
            "lookback_pnl": pnl_series.values
        })
        all_pnl_parts.append(tmp)

    # 4. Combine and finalize
    if not all_pnl_parts:
        return pd.DataFrame(columns=["position_index", "pnl_date", "lookback_pnl", "inverse_pnl"])

    out = pd.concat(all_pnl_parts, ignore_index=True)
    out["inverse_pnl"] = -out["lookback_pnl"]

    return out


def generate_sensitivity_repricing_pnl(combined_pos_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates lookback and inverse lookback PnL for combined linear and non-linear
    positions, using sensitivity repricing for options, and ensures all results
    are concatenated in the final long (stacked) format.
    """
    logger = get_logger(__name__)
    all_pnl_dfs = []  # Collect all PnL components in long format here

    # --- 0. Initial Setup & Data Segregation ---
    product = combined_pos_df['product'].unique()[0] if len(combined_pos_df['product'].unique()) == 1 else None
    cob_date = combined_pos_df['cob_date'].unique()[0] if len(combined_pos_df['cob_date'].unique()) == 1 else None
    simulation_method = combined_pos_df['simulation_method'].unique()[0] if (
            len(combined_pos_df['simulation_method'].unique()) == 1) else None
    calculation_method = combined_pos_df['calculation_method'].unique()[0] if (
            len(combined_pos_df['calculation_method'].unique()) == 1) else None

    if product is None:
        logger.warning(f"More than one product in combined_pos_df: {combined_pos_df['product'].unique()}")
    if cob_date is None:
        logger.warning(f"More than one cob_date in combined_pos_df: {combined_pos_df['cob_date'].unique()}")

    is_non_linear = combined_pos_df["derivative_type"].isin(['VANILLA_CALL', 'VANILLA_PUT'])
    linear_pos_df = combined_pos_df[~is_non_linear].copy()
    options_pos_df = combined_pos_df[is_non_linear].copy()

    # 1. Calculate Linear PnL (for non-derivatives) (LONG format)
    linear_pnl_df = generate_linear_pnl(linear_pos_df, returns_df)
    all_pnl_dfs.append(linear_pnl_df)

    # --- Edge Case: No Options Positions ---
    if options_pos_df.empty:
        logger.info("No options positions found. Returning only linear PnL.")
        return pd.concat(all_pnl_dfs, ignore_index=True)

    # --- Load Sensitivity Matrix ---
    uat_engine = get_engine('uat')
    sensitivity_matrix = SensitivityMatrixLoader(cob_date, product, uat_engine)
    sensitivity_df = sensitivity_matrix.load_sensitivity_matrix()

    # --- Edge Case: Missing Sensitivity Matrix Fallback ---
    if sensitivity_df is None or sensitivity_df.empty:
        logger.warning("Options positions detected, but no sensitivity matrix. "
                       "Calculating linear PnL as fallback for all derivatives.")
        options_pnl_df = generate_linear_pnl(options_pos_df, returns_df)  # Still LONG format
        all_pnl_dfs.append(options_pnl_df)
        return pd.concat(all_pnl_dfs, ignore_index=True)

    # --- 2. Match Positions to Sensitivity Curves ---
    options_pos_df['match_key'] = (options_pos_df['portfolio'].astype(str) + '|' +
                                   options_pos_df['strike'].astype(str) + '|' +
                                   options_pos_df['product_code'].astype(str) + '|' +
                                   options_pos_df['contract_month'].astype(str) + '|' +
                                   options_pos_df['derivative_type'].astype(str) + '|' +
                                   options_pos_df['total_active_lots'].astype(str)
                                   )
    valid_sensitivity_keys = set(sensitivity_df['match_key'].unique())
    is_matched_to_sensitivity = options_pos_df['match_key'].isin(valid_sensitivity_keys)

    options_matched_pos_df = options_pos_df[is_matched_to_sensitivity].copy()
    options_unmatched_pos_df = options_pos_df[~is_matched_to_sensitivity].copy()

    logger.info(f"Split positions: {len(options_matched_pos_df)} matched, {len(options_unmatched_pos_df)} unmatched "
                f"to sensitivity report.")

    # 3. Calculate Unmatched Options PnL (LONG format, linear fallback)
    options_unmatched_pnl_df = generate_linear_pnl(options_unmatched_pos_df, returns_df)
    all_pnl_dfs.append(options_unmatched_pnl_df)

    # --- Edge Case: No Positions Matched to Sensitivity ---
    if options_matched_pos_df.empty:
        logger.info("No options positions matched sensitivity curves. Returning combined PnL.")
        return pd.concat(all_pnl_dfs, ignore_index=True)

    # --- 4. Sensitivity Repricing PnL Calculation (Matched Positions) ---

    # Pre-cache sensitivity curves for fast lookup
    curve_lookups: Dict[str, Dict[str, np.ndarray]] = {}
    for curve_key, group_df in sensitivity_df.groupby('match_key'):
        sorted_curve = group_df.sort_values(by='pricemove_in_percentage')
        curve_lookups[curve_key] = {
            'xp': sorted_curve['pricemove_in_percentage'].values,
            'yp': sorted_curve['diff_m2m_usd'].values
        }
    logger.info(f'Pre-cached {len(curve_lookups)} unique sensitivity curves into NumPy arrays.')

    # Prepare input data for interpolation
    selected_cols = options_matched_pos_df["risk_factor"].astype(str).tolist()
    # Rows = Positions, Columns = Dates
    matched_returns_matrix = returns_df[selected_cols].T.values

    N_matched = len(options_matched_pos_df)
    T_dates = len(returns_df)
    lookback_pnl_array = np.zeros((N_matched, T_dates))
    inverse_pnl_array = np.zeros((N_matched, T_dates))

    key_to_indices = options_matched_pos_df.groupby('match_key').groups

    for curve_key, pos_indices in key_to_indices.items():
        curve_data = curve_lookups[curve_key]
        xp = curve_data['xp']
        yp = curve_data['yp']

        # Get the zero-based integer row indices in the returns matrix
        position_row_indices = options_matched_pos_df.index.get_indexer(pos_indices)
        curve_returns = matched_returns_matrix[position_row_indices, :]

        # Vectorized Interpolation
        lookback_pnl = np.interp(curve_returns, xp, yp)
        if simulation_method == 'hist_sim':
            inverse_pnl = np.interp(-curve_returns, xp, yp)
        elif simulation_method == 'mc_sim':
            inverse_pnl = -lookback_pnl
            # No need for lookback pnl for mc_sim as simulation already normalised distribution and includes both tails
        else:
            raise ValueError(
                f"Invalid simulation_method: '{simulation_method}'. "
                "Expected 'hist_sim' or 'mc_sim'."
            )

        # Store results back into the final arrays
        lookback_pnl_array[position_row_indices, :] = lookback_pnl
        inverse_pnl_array[position_row_indices, :] = inverse_pnl

    logger.info('Non-linear PnL calculation complete.')

    # --- 5. Convert Matched PnL to LONG Format ---

    # 5a. Create WIDE PnL DataFrames (Index=Date, Columns=TempIndex)
    options_lookback_matched_pnl_df = pd.DataFrame(
        lookback_pnl_array.T,
        index=returns_df.index,
        columns=options_matched_pos_df.index
    ).rename_axis(columns=None).rename_axis(index='pnl_date')  # Use pnl_date for stack

    options_inverse_matched_pnl_df = pd.DataFrame(
        inverse_pnl_array.T,
        index=returns_df.index,
        columns=options_matched_pos_df.index
    ).rename_axis(columns=None).rename_axis(index='pnl_date')

    # 5b. Stack to LONG format
    matched_lookback_long = options_lookback_matched_pnl_df.stack().reset_index()
    matched_lookback_long.columns = ['pnl_date', 'temp_index', 'lookback_pnl']

    matched_inverse_long = options_inverse_matched_pnl_df.stack().reset_index()
    matched_inverse_long.columns = ['pnl_date', 'temp_index', 'inverse_pnl']

    # 5c. Map the temporary index back to the real position_index
    index_map = options_matched_pos_df['position_index'].to_dict()
    matched_lookback_long['position_index'] = matched_lookback_long['temp_index'].map(index_map)
    matched_inverse_long['position_index'] = matched_inverse_long['temp_index'].map(index_map)

    # 5d. Merge the matched results (LONG)
    matched_pnl_long_df = pd.merge(
        matched_lookback_long[['pnl_date', 'position_index', 'lookback_pnl']],
        matched_inverse_long[['pnl_date', 'position_index', 'inverse_pnl']],
        on=['pnl_date', 'position_index'],
        how='outer'
    )
    all_pnl_dfs.append(matched_pnl_long_df)

    # 6. Final Concatenation (All PnL is now in LONG format)
    final_pnl_df = pd.concat(all_pnl_dfs, ignore_index=True)

    return final_pnl_df


def generate_pnl_vectors(combined_pos_df: pd.DataFrame, returns_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Wrapper to generate PnL vectors by method.
    """
    if method == "linear":
        return generate_linear_pnl(combined_pos_df, returns_df)
    elif method == "taylor_series":
        return generate_taylor_series_pnl(combined_pos_df, returns_df)
    elif method == "sensitivity_matrix":
        return generate_sensitivity_repricing_pnl(combined_pos_df, returns_df)
    elif method == "repricing":
        pass
        # return generate_full_repricing_pnl(combined_pos_df, returns_df)
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
        full_path: str,
        write_to_excel: bool,
        is_truncated: bool,
        write_to_feather_for_oga_level_var: bool
) -> Dict[str, Any]:
    """
    Analyze PnL vectors and optionally export to Excel and/or .feather.

    Optimization Note: The pivoting operations have been consolidated to reduce
    redundant indexing and reshaping, which dramatically speeds up the function.
    """
    logger = get_logger(__name__)
    analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)

    # Filter data into two large long-form DataFrames
    outright_analyzer = analyzer.filter_position_metadata(exposure='OUTRIGHT')
    logger.info('STEP 3A-1: Outright positions analysed')
    if product == 'cotton':
        basis_analyzer = analyzer.filter_position_metadata(exposure='BASIS (NET PHYS)')
    logger.info('STEP 3A-2: Basis positions analysed')
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
    logger.info('STEP 3A-3: Outright lookback PnL prepared')
    unit_outright_inverse = outright_pivoted_combined['inverse_pnl']
    logger.info('STEP 3A-4: Outright inverse PnL prepared')
    # 2. Basis analyzer still requires its own pivot, but the cost is minimized.
    if product == 'cotton':
        unit_basis_lookback = basis_analyzer.pivot(
            index='pnl_date',
            columns=['region', 'position_index'],
            values='lookback_pnl'
        )
    logger.info('STEP 3A-5: Basis PnL prepared')

    if write_to_excel:
        writer_kwargs = {'mode': 'w'}
        if os.path.exists(full_path):
            writer_kwargs['mode'] = 'a'
            writer_kwargs['if_sheet_exists'] = 'replace'

        with (pd.ExcelWriter(full_path, **writer_kwargs) as writer):
            combined_pos_df.to_excel(writer, sheet_name='pos', index=True)
            if is_truncated:
                returns_df.sort_index(ascending=False).head().to_excel(writer, sheet_name='returns', index=True)
                prices_df.sort_index(ascending=False).head().to_excel(writer, sheet_name='prices', index=True)

                # Export results using the data extracted from the single combined pivot
                unit_outright_lookback.sort_index(ascending=False).head(100).to_excel(writer,
                                                                                   sheet_name='outright_lookback',
                                                                                   index=True)
                unit_outright_inverse.sort_index(ascending=False).head(100).to_excel(writer,
                                                                                  sheet_name='outright_inverse',
                                                                                  index=True)
                if product == 'cotton':
                    unit_basis_lookback.sort_index(ascending=False).head(100).to_excel(writer,
                                                                                    sheet_name='basis_lookback',
                                                                                    index=True)

            else:
                returns_df.sort_index(ascending=False).to_excel(writer, sheet_name='returns', index=True)
                prices_df.sort_index(ascending=False).to_excel(writer, sheet_name='prices', index=True)

                # Export results using the data extracted from the single combined pivot
                unit_outright_lookback.sort_index(ascending=False).to_excel(writer, sheet_name='outright_lookback',
                                                                            index=True)
                unit_outright_inverse.sort_index(ascending=False).to_excel(writer, sheet_name='outright_inverse',
                                                                           index=True)
                if product == 'cotton':
                    unit_basis_lookback.sort_index(ascending=False).to_excel(writer, sheet_name='basis_lookback',
                                                                             index=True)
            logger.info(f'STEP 3A-6: Unit PnL vectors exported to excel: {full_path}')

    if write_to_feather_for_oga_level_var:
        # 1. Aggregate PnL vectors into arrays, ensuring sorting by pnl_date
        # We only need the PnL values, grouped by position_index.
        def aggregate_pnl_vectors(group):
            # Ensure chronological order by pnl_date before converting to numpy array
            group = group.sort_values(by='pnl_date', ascending=True)
            return pd.Series({
                'lookback_pnl_vector': group['lookback_pnl'].to_numpy(),
                'inverse_pnl_vector': group['inverse_pnl'].to_numpy()
            })

        # Group by position_index and apply the aggregation function
        aggregated_pnl = analyzer.pnl_df.groupby('position_index').apply(
            aggregate_pnl_vectors
        ).reset_index()

        # Rename columns to match the requested output
        aggregated_pnl.rename(columns={
            'lookback_pnl_vector': 'lookback_pnl',
            'inverse_pnl_vector': 'inverse_pnl'
        }, inplace=True)

        # Merge
        feather_df = pd.merge(
            analyzer.position_df,
            aggregated_pnl,
            on='position_index',
            how='inner'
        )

        feather_df['level 15'] = product.capitalize()
        if product == 'cotton':
            # only consider Sum Cotton (ex JS, US Eq OR)
            logger.info(f"{product} has total {len(feather_df)} positions")
            feather_df = feather_df[~((feather_df['region'] == 'USA EQUITY') & (feather_df['exposure'] == 'OUTRIGHT'))]
            logger.info(f"{product} only includes {len(feather_df)} Sum Cotton (ex JS, US Eq OR) positions")

        elif product == 'rubber':
            condition_ivc = (feather_df['region'] == 'IVC MANUFACTURING')
            feather_df.loc[condition_ivc, 'level 14'] = 'Midstream'
            feather_df.loc[~condition_ivc, 'level 14'] = 'Trading & Supply Chain'

        elif product == 'biocane':
            feather_df['level 15'] = 'Project Bio-Cane'

        elif product == 'rms':
            feather_df['level 15'] = 'RMS'

            # 3. Define the feather path and export
        feather_path = full_path.replace('.xlsx', '.feather')
        feather_df.to_feather(feather_path)
        logger.info(f'STEP 3A-7: {len(feather_df)} Position and PnL vectors exported to Feather: {feather_path}')

    return {}
