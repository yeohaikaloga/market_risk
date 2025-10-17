from workflow.var.var_calculation_utils import get_cotton_region_aggregates, get_rubber_region_aggregates
from pnl_analyzer.pnl_analyzer import PnLAnalyzer
from workflow.var.var_calculation_utils import build_position_index_df, calculate_var_for_units
from utils.contract_utils import instrument_ref_dict

import pandas as pd
import os


def generate_var(product, combined_pos_df, long_pnl_df, cob_date, window) -> pd.DataFrame:

    # Step 1: Build analyzer
    pnl_analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)
    # Step 2: Build position index DataFrame
    if product == 'cotton':
        region_aggregate_map = get_cotton_region_aggregates(combined_pos_df['region'].unique())
    elif product == 'rubber':
        region_aggregate_map = get_rubber_region_aggregates(combined_pos_df['region'].unique())

    var_data_df = build_position_index_df(pnl_analyzer, region_aggregate_map)
    # Step 3: Run VaR calculation
    var_data_df = calculate_var_for_units(
        var_data_df=var_data_df,
        analyzer=pnl_analyzer,
        cob_date=cob_date,
        window=window
    )

    return var_data_df


def build_var_report(var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic report builder supporting region, prop, agg levels.
    """
    columns = ['outright_pos', 'basis_pos', 'outright_95_var', 'basis_95_var', 'overall_95_var',
               'outright_99_var', 'basis_99_var', 'overall_99_var']

    # --- Region-level report ---
    unique_regions = sorted(var_df[var_df['level'] == 'region']['region_agg'].unique())
    region_report = pd.DataFrame(columns=columns, index=unique_regions)

    for unit in unique_regions:
        filtered_row = var_df[(var_df['level'] == 'region') & (var_df['region_agg'] == unit) & (var_df['instrument_name'] == 'ALL')]
        if filtered_row.empty:
            continue
        if len(filtered_row) == 1:
            row = filtered_row.iloc[0]
            region_report.loc[unit] = row[columns]

    # --- Prop-level report ---
    prop_df = var_df[var_df['level'] == 'prop']
    unique_props = (prop_df['region_agg'] + '_' + prop_df['instrument_name']).unique()
    prop_report = pd.DataFrame(columns=columns, index=unique_props)

    for unit in unique_props:
        region_agg, instr_name = unit.split('_', 1)
        filtered_row = var_df[(var_df['level'] == 'prop') &
                             (var_df['region_agg'] == region_agg) &
                             (var_df['instrument_name'] == instr_name)]
        if filtered_row.empty:
            continue
        if len(filtered_row) == 1:
            row = filtered_row.iloc[0]
            prop_report.loc[unit] = row[columns]

    # --- Aggregate-level report ---
    agg_df = var_df[var_df['level'] == 'agg']
    unique_aggs = agg_df['region_agg'].str.replace(r'\[O\]', '', regex=True).str.replace(r'\[B\]', '', regex=True).unique()
    agg_report = pd.DataFrame(columns=columns, index=unique_aggs)

    for unit in unique_aggs:
        filtered_row = agg_df[agg_df['region_agg'].str.contains(unit)]
        # Find 'ALL' instrument_name row or first match
        filtered_row_all = filtered_row[filtered_row['instrument_name'] == 'ALL']
        row = filtered_row_all.iloc[0] if not filtered_row_all.empty else filtered_row.iloc[0] if not filtered_row.empty else None
        if row is not None:
            agg_report.loc[unit] = row[columns]

    # Combine all into one DataFrame - either concat or append
    report_df = pd.concat([region_report, prop_report, agg_report])
    return report_df

def build_cotton_var_report_exceptions(long_pnl_df: pd.DataFrame,
                                       combined_pos_df: pd.DataFrame,
                                       report_df: pd.DataFrame,
                                       var_df: pd.DataFrame,
                                       cob_date: str,
                                       window: int) -> pd.DataFrame:
    """
    Apply cotton-specific overrides:
    - For CENTRAL x_ALL, replace outright_pos with that from CENTRAL x_CT
    - CENTRAL positions are cotton positions only, but VaR is for all positions
    - Rename units with product suffixes based on instrument_ref_dict
    - Apply the relevant definitions for the aggregates with exclusions (ex. CP/JS/US EQ)
    """

    report_df = report_df.copy()

    # Step 1: Overwrite CENTRAL x_ALL's outright_pos using CENTRAL x_CT
    for i in range(2, 5):
        all_key = f'CENTRAL {i}_ALL'
        ct_key = f'CENTRAL {i}_CT'
        if all_key in report_df.index and ct_key in report_df.index:
            report_df.loc[all_key, 'outright_pos'] = report_df.loc[ct_key, 'outright_pos']

    # Step 2: Create SUM CENTRAL from CENTRAL x_CT (cotton positions only, but VaR for all positions, similarly for SUM COTTON)
    central_ct_keys = [f'CENTRAL {i}_CT' for i in range(2, 10)]
    existing_ct_keys = [k for k in central_ct_keys if k in report_df.index]
    if existing_ct_keys:
        report_df.loc['SUM CENTRAL', 'outright_pos'] = report_df.loc[existing_ct_keys, 'outright_pos'].sum()
    report_df.loc['SUM COTTON', 'outright_pos'] = (report_df.loc['SUM CENTRAL', 'outright_pos']
                                                    + report_df.loc['SUM ORIGIN', 'outright_pos'])

    # Step 3: Rename suffixes using instrument_ref_dict and replace underscores
    def replace_suffix(unit):
        for instrument in instrument_ref_dict.keys():
            product_name = instrument_ref_dict[instrument].get('product_name')

            if unit.endswith('_' + instrument):
                # Replace instrument with product name, e.g. 'CENTRAL 2_CT' -> 'CENTRAL 2_COTTON ICE'
                return unit.rsplit('_', 1)[0] + ' ' + product_name
        return unit.replace('_', ' ').replace('ALL', ' ')
    new_index = [replace_suffix(unit) for unit in report_df.index]
    report_df.index = new_index

    # Step 4: Apply the relevant definitions for the aggregates with exclusions (ex. CP/JS/US EQ)
    sum_origin_o = var_df[(var_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ[O]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'outright_pos'] = var_df.loc[sum_origin_o, 'outright_pos']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'outright_95_var'] = var_df.loc[sum_origin_o, 'outright_95_var']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'outright_99_var'] = var_df.loc[sum_origin_o, 'outright_99_var']
    sum_origin_b = var_df[(var_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ[B]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'basis_pos'] = var_df.loc[sum_origin_b, 'basis_pos']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'basis_95_var'] = var_df.loc[sum_origin_b, 'basis_95_var']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'basis_99_var'] = var_df.loc[sum_origin_b, 'basis_99_var']

    sum_cotton_o = var_df[(var_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ[O]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'outright_pos'] = var_df.loc[sum_cotton_o, 'outright_pos']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'outright_95_var'] = var_df.loc[sum_cotton_o, 'outright_95_var']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'outright_99_var'] = var_df.loc[sum_cotton_o, 'outright_99_var']
    sum_cotton_b = var_df[(var_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ[B]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'basis_pos'] = var_df.loc[sum_cotton_b, 'basis_pos']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'basis_95_var'] = var_df.loc[sum_cotton_b, 'basis_95_var']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'basis_99_var'] = var_df.loc[sum_cotton_b, 'basis_99_var']

    exception_cotton_var_data_df = pd.DataFrame(columns=var_df.columns)
    sum_origin_ex_row = {
        'region_agg': 'SUM ORIGIN EX CP/JS/US EQ',
        'level': 'agg',
        'instrument_name': 'ALL',
        'outright_position_index_list': var_df.loc[sum_origin_o, 'outright_position_index_list'],
        'no_of_outright_positions': len(var_df.loc[sum_origin_o, 'outright_position_index_list']),
        'basis_position_index_list': var_df.loc[sum_origin_b, 'basis_position_index_list'],
        'no_of_basis_positions': len(var_df.loc[sum_origin_b, 'basis_position_index_list']),
    }
    sum_cotton_ex_row = {
        'region_agg': 'SUM COTTON EX CP/JS/US EQ',
        'level': 'agg',
        'instrument_name': 'ALL',
        'outright_position_index_list': var_df.loc[sum_cotton_o, 'outright_position_index_list'],
        'no_of_outright_positions': len(var_df.loc[sum_cotton_o, 'outright_position_index_list']),
        'basis_position_index_list': var_df.loc[sum_cotton_b, 'basis_position_index_list'],
        'no_of_basis_positions': len(var_df.loc[sum_cotton_b, 'basis_position_index_list']),
    }

    exception_cotton_var_data_df = pd.concat([exception_cotton_var_data_df, pd.DataFrame([sum_origin_ex_row, sum_cotton_ex_row])], ignore_index=True)
    pnl_analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)
    exception_cotton_var_data_df = calculate_var_for_units(
        var_data_df=exception_cotton_var_data_df,
        analyzer=pnl_analyzer,
        cob_date=cob_date,
        window=window
    )
    sum_origin_ov = exception_cotton_var_data_df[
        exception_cotton_var_data_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ'
        ].index[0]
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'overall_95_var'] = exception_cotton_var_data_df.loc[sum_origin_ov, 'overall_95_var']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'overall_99_var'] = exception_cotton_var_data_df.loc[sum_origin_ov, 'overall_99_var']
    sum_cotton_ov = exception_cotton_var_data_df[
        exception_cotton_var_data_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ'
        ].index[0]
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'overall_95_var'] = exception_cotton_var_data_df.loc[sum_cotton_ov, 'overall_95_var']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'overall_99_var'] = exception_cotton_var_data_df.loc[sum_cotton_ov, 'overall_99_var']

    return report_df

def build_cotton_price_var_report_exceptions(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cotton-specific price overrides:
    - Only show outright position and VaR
    """

    # Step 1: Only show outright position and VaR
    report_df = report_df.copy()
    report_df = report_df[['outright_pos', 'outright_95_var', 'outright_99_var']]

    # Step 2: Overwrite CENTRAL x_ALL's outright_pos using CENTRAL x_CT
    for i in range(2, 5):
        all_key = f'CENTRAL {i}_ALL'
        ct_key = f'CENTRAL {i}_CT'
        if all_key in report_df.index and ct_key in report_df.index:
            report_df.loc[all_key, 'outright_pos'] = report_df.loc[ct_key, 'outright_pos']

    # Step 3: Create SUM CENTRAL from CENTRAL x_CT (cotton positions only, but VaR for all positions, similarly for SUM COTTON)
    central_ct_keys = [f'CENTRAL {i}_CT' for i in range(2, 10)]
    existing_ct_keys = [k for k in central_ct_keys if k in report_df.index]
    if existing_ct_keys:
        report_df.loc['SUM CENTRAL', 'outright_pos'] = report_df.loc[existing_ct_keys, 'outright_pos'].sum()
    report_df.loc['SUM COTTON', 'outright_pos'] = (report_df.loc['SUM CENTRAL', 'outright_pos']
                                                    + report_df.loc['SUM ORIGIN', 'outright_pos'])

    # Step 4: Only retain SUM CENTRAL and drop all other CENTRAL rows
    report_df = report_df[~((report_df.index.str.contains('CENTRAL')) & (report_df.index != 'SUM CENTRAL'))]

    # Step 5: Drop all rows with EX CP/JS/US EQ
    report_df = report_df[~report_df.index.str.contains("EX CP/JS/US EQ")]

    # Step 6: For aggregates (with SUM), prefix with PRICE
    report_df.index = report_df.index.map(
        lambda x: "PRICE " + x if x.startswith("SUM") else x
    )

    return report_df

def build_rubber_var_report_exceptions(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rubber-specific overrides:
    - Only show outright position, basis position and outright VaR
    """

    # Step 1: Only show outright position, basis position and outright VaR
    report_df = report_df.copy()
    report_df = report_df[['outright_pos', 'basis_pos', 'outright_95_var', 'outright_99_var']]
    return report_df


