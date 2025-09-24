import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from financial_calculations.var_calculator import VaRCalculator
from pnl_analyzer.pnl_analyzer import PnLAnalyzer


def get_cotton_region_aggregates(region_list: list[str]) -> dict[str, list[str]]:
    """
    Define cotton-specific region aggregates for VaR reporting.
    """
    sum_africa = ['W. AFRICA', 'TOGO', 'CHAD', 'E. AFRICA', 'SECO']
    sum_usa = ['USA', 'USA EQUITY']
    sum_usa_mex = sum_usa + ['MEXICO']
    sum_americas = sum_usa_mex + ['BRAZIL']
    sum_aus_china = ['AUSTRALIA', 'CHINA']
    sum_aus_china_ind = sum_aus_china + ['INDIA']
    sum_med = ['TURKEY+', 'GREECE+']
    sum_central = [x for x in region_list if 'CENTRAL ' in x]
    sum_origin = [x for x in region_list if x not in sum_central]
    sum_non_usa = [x for x in sum_origin if x not in sum_usa]

    sum_origin_ex_cp_js_useq_outright = [x for x in sum_origin if x not in ['CP', 'JESS SMITH', 'USA EQUITY']]
    sum_origin_ex_cp_js_useq_basis = [x for x in sum_origin if x not in ['JESS SMITH']]

    sum_cotton = sum_origin + sum_central
    sum_cotton_ex_cp_js_useq_outright = sum_origin_ex_cp_js_useq_outright + sum_central
    sum_cotton_ex_cp_js_useq_basis = sum_origin_ex_cp_js_useq_basis + sum_central

    return {
        'SUM AFRICA': sum_africa,
        'SUM USA': sum_usa,
        'SUM USA & MEXICO': sum_usa_mex,
        'SUM AMERICAS': sum_americas,
        'SUM AUSTRALIA/CHINA': sum_aus_china,
        'SUM AUSTRALIA/CHINA/INDIA': sum_aus_china_ind,
        'SUM MED REGION': sum_med,
        'SUM CENTRAL': sum_central,
        'SUM ORIGIN': sum_origin,
        'SUM NON-USA': sum_non_usa,
        'SUM ORIGIN EX CP/JS/US EQ[O]': sum_origin_ex_cp_js_useq_outright,
        'SUM ORIGIN EX CP/JS/US EQ[B]': sum_origin_ex_cp_js_useq_basis,
        'SUM COTTON': sum_cotton,
        'SUM COTTON EX CP/JS/US EQ[O]': sum_cotton_ex_cp_js_useq_outright,
        'SUM COTTON EX CP/JS/US EQ[B]': sum_cotton_ex_cp_js_useq_basis,
    }

def build_position_index_df(
    pnl_analyzer: PnLAnalyzer,
    region_aggregate_map: dict
) -> pd.DataFrame:
    records = []

    def get_position_data(analyzer):
        if not analyzer:
            return {
                'outright_position_index_list': [],
                'no_of_outright_positions': 0,
                'basis_position_index_list': [],
                'no_of_basis_positions': 0
            }

        outright_analyzer = analyzer.filter(exposure='OUTRIGHT')
        basis_analyzer = analyzer.filter(exposure='BASIS (NET PHYS)')

        outright_positions = outright_analyzer.get_position_index() if outright_analyzer else []
        basis_positions = basis_analyzer.get_position_index() if basis_analyzer else []

        return {
            'outright_position_index_list': outright_positions,
            'no_of_outright_positions': len(outright_positions),
            'basis_position_index_list': basis_positions,
            'no_of_basis_positions': len(basis_positions)
        }

    # === Per-region entries ===
    region_list = pnl_analyzer._pos_and_pnl_df['region'].unique()
    for region in region_list:
        if pnl_analyzer:
            region_analyzer = pnl_analyzer.filter(region=region)
            instruments = region_analyzer._pos_and_pnl_df['instrument_name'].unique()

            level = 'prop' if 'CENTRAL ' in region else 'region'

            # ALL positions in region
            pos_data = get_position_data(region_analyzer)
            records.append({
                'region_agg': region,
                'level': level,
                'instrument_name': 'ALL',
                **pos_data
            })

        if 'CENTRAL ' in region:
            # NON-COTTON only for 'CENTRAL ' regions
            if region_analyzer:
                non_cotton_analyzer = region_analyzer.filter(
                    instrument_name=lambda c: c not in ['CT', 'VV', 'CCL']
                )
                pos_data = get_position_data(non_cotton_analyzer)
                records.append({
                    'region_agg': region,
                    'level': level,
                    'instrument_name': 'NON-COTTON',
                    **pos_data
                })

            # Per-instrument entries
            for instrument_name in instruments:
                if region_analyzer:
                    instrument_analyzer = region_analyzer.filter(instrument_name=instrument_name)
                    pos_data = get_position_data(instrument_analyzer)
                    records.append({
                        'region_agg': region,
                        'level': level,
                        'instrument_name': instrument_name,
                        **pos_data
                    })

    # === Per-aggregate entries ===
    for agg_name, regions in region_aggregate_map.items():
        if pnl_analyzer:
            agg_analyzer = pnl_analyzer.filter(region=regions)

            level = 'agg'

            # ALL in aggregate
            pos_data = get_position_data(agg_analyzer)
            records.append({
                'region_agg': agg_name,
                'level': level,
                'instrument_name': 'ALL',
                **pos_data
            })

        # COTTON only for aggregates
        if agg_analyzer:
            cotton_analyzer = agg_analyzer.filter(
                instrument_name=lambda c: c in ['CT', 'VV', 'CCL', 'EX GIN S6']
            )
            print(f"pos_data: {pos_data}, type: {type(pos_data)}")
            pos_data = get_position_data(cotton_analyzer)
            records.append({
                'region_agg': agg_name,
                'level': level,
                'instrument_name': 'COTTON',
                **pos_data
            })

    return pd.DataFrame.from_records(records)

def calculate_var_for_units(
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

        def safe_pivot(analyzer, value):
            if analyzer is not None:
                return analyzer.pivot(index='pnl_date', columns='position_index', values=value)
            return pd.DataFrame()

        outright_lookback = safe_pivot(outright_analyzer, 'lookback_pnl')
        outright_inverse = safe_pivot(outright_analyzer, 'inverse_pnl')

        basis_lookback = safe_pivot(basis_analyzer, 'lookback_pnl')
        basis_inverse = safe_pivot(basis_analyzer, 'inverse_pnl')

        overall_lookback = safe_pivot(overall_analyzer, 'lookback_pnl')
        overall_inverse = safe_pivot(overall_analyzer, 'inverse_pnl')

        def safe_position_sum(analyzer, positions):
            if analyzer is not None and analyzer.position_df is not None and not analyzer.position_df.empty:
                return analyzer.position_df[analyzer.position_df['position_index'].isin(positions)]['delta'].sum()
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