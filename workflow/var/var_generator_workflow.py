from pnl_analyzer.pnl_analyzer import PnLAnalyzer
from utils.var_calculation_utils import calculate_var_for_regions
from utils.contract_utils import load_instrument_ref_dict

import pandas as pd


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

def get_rubber_region_aggregates(region_list: list[str]) -> dict[str, list[str]]:
    """
    Define rubber-specific region aggregates for VaR reporting.
    """
    sum_prop_trading = ['SINGAPORE PROP 1', 'SINGAPORE PROP 2']
    sum_sg_desk = ['MALAY', 'THAILAND 1', 'THAILAND 2'] + sum_prop_trading
    sum_indoviet = ['INDO', 'VIETNAM']
    sum_trading_supply_chain = sum_sg_desk + sum_indoviet + ['AFRICA', 'CHINA']
    sum_midstream = ['IVC MANUFACTURING']
    sum_rubber = sum_trading_supply_chain + sum_midstream

    return {
        'SG DESK': sum_sg_desk,
        'INDOVIET': sum_indoviet,
        'TRADING & SUPPLY CHAIN': sum_trading_supply_chain,
        'MIDSTREAM': sum_midstream,
        'RUBBER TOTAL': sum_rubber,
        'PROP TRADING': sum_prop_trading
    }

def get_rms_aggregates(region_list: list[str]) -> dict[str, list[str]]:
    sum_rms = region_list
    return {
        'SUM RMS': sum_rms
    }

def get_biocane_aggregates(region_list: list[str]) -> dict[str, list[str]]:
    sum_biocane = region_list
    return {
        'SUM BIOCANE': sum_biocane
    }

def get_wood_aggregates(region_list: list[str]) -> dict[str, list[str]]:
    sum_wood = region_list
    return {
        'SUM WOOD': sum_wood
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

        outright_analyzer = analyzer.filter_position_metadata(exposure='OUTRIGHT')
        basis_analyzer = analyzer.filter_position_metadata(exposure='BASIS (NET PHYS)')

        outright_positions = outright_analyzer.get_position_index() if outright_analyzer else []
        basis_positions = basis_analyzer.get_position_index() if basis_analyzer else []

        return {
            'outright_position_index_list': list(outright_positions),
            'no_of_outright_positions': len(outright_positions),
            'basis_position_index_list': list(basis_positions),
            'no_of_basis_positions': len(basis_positions)
        }

    # === Per-region entries ===
    region_list = pnl_analyzer.position_df['region'].unique()
    for region in region_list:
        print(region)
        if pnl_analyzer:
            region_analyzer = pnl_analyzer.filter_position_metadata(region=region)
            instruments = region_analyzer.position_df['instrument_name'].unique()
            print(instruments)

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
                non_cotton_analyzer = region_analyzer.filter_position_metadata(
                    instrument_name=lambda c: c not in ['CT', 'VV', 'CCL', 'AVY']
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
                print(instrument_name)
                if region_analyzer:
                    instrument_analyzer = region_analyzer.filter_position_metadata(instrument_name=instrument_name)
                    pos_data = get_position_data(instrument_analyzer)
                    records.append({
                        'region_agg': region,
                        'level': level,
                        'instrument_name': instrument_name,
                        **pos_data
                    })

    # === Per-aggregate entries ===
    for agg_name, regions in region_aggregate_map.items():
        print(agg_name, regions)
        if pnl_analyzer:
            agg_analyzer = pnl_analyzer.filter_position_metadata(region=regions)

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
            cotton_analyzer = agg_analyzer.filter_position_metadata(
                instrument_name=lambda c: c in ['CT', 'VV', 'CCL', 'AVY', 'EX GIN S6']
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

def generate_var(product, combined_pos_df, long_pnl_df, simulation_method, calculation_method, cob_date, window) \
        -> pd.DataFrame:

    # Step 1: Build analyzer
    pnl_analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)
    # Step 2: Build position index DataFrame
    if product == 'cotton':
        region_aggregate_map = get_cotton_region_aggregates(combined_pos_df['region'].unique())
    elif product == 'rubber':
        region_aggregate_map = get_rubber_region_aggregates(combined_pos_df['region'].unique())
    elif product == 'rms':
        region_aggregate_map = get_rms_aggregates(combined_pos_df['region'].unique())
    elif product == 'biocane':
        region_aggregate_map = get_biocane_aggregates(combined_pos_df['region'].unique())
    elif product == 'wood':
        region_aggregate_map = get_wood_aggregates(combined_pos_df['region'].unique())

    var_data_df = build_position_index_df(pnl_analyzer, region_aggregate_map)

    # Step 3: Run VaR calculation
    if product == 'rms' or simulation_method == 'mc_sim':
        is_two_tail = False
    else:
        is_two_tail = True
    var_data_df = calculate_var_for_regions(
        var_data_df=var_data_df,
        analyzer=pnl_analyzer,
        simulation_method=simulation_method,
        calculation_method=calculation_method,
        cob_date=cob_date,
        window=window,
        is_two_tail=is_two_tail
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

def build_cotton_var_report_exceptions(long_pnl_df: pd.DataFrame, combined_pos_df: pd.DataFrame, simulation_method: str,
                                       calculation_method: str, report_df: pd.DataFrame, var_df: pd.DataFrame,
                                       cob_date: str, window: int) -> pd.DataFrame:
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
        instrument_ref_dict = load_instrument_ref_dict('uat')
        for instrument in instrument_ref_dict.keys():
            product_name = instrument_ref_dict[instrument].get('product_name')

            if unit.endswith('_' + instrument):
                # Replace instrument with product name, e.g. 'CENTRAL 2_CT' -> 'CENTRAL 2_COTTON ICE'
                return unit.rsplit('_', 1)[0] + ' ' + product_name
        return unit.replace('_', ' ').replace('ALL', ' ')
    new_index = [replace_suffix(unit) for unit in report_df.index]
    report_df.index = new_index

    # Step 4: Apply the relevant definitions for the aggregates with exclusions (ex. CP/JS/US EQ OR)
    sum_origin_o_c = var_df[(var_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ[O]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    sum_origin_o_all = var_df[(var_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ[O]') & (var_df['instrument_name'] == 'ALL')].index[0]
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'outright_pos'] = var_df.loc[sum_origin_o_c, 'outright_pos']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'outright_95_var'] = var_df.loc[sum_origin_o_all, 'outright_95_var']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'outright_99_var'] = var_df.loc[sum_origin_o_all, 'outright_99_var']
    sum_origin_b_c = var_df[(var_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ[B]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    sum_origin_b_all = var_df[(var_df['region_agg'] == 'SUM ORIGIN EX CP/JS/US EQ[B]') & (var_df['instrument_name'] == 'ALL')].index[0]
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'basis_pos'] = var_df.loc[sum_origin_b_c, 'basis_pos']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'basis_95_var'] = var_df.loc[sum_origin_b_all, 'basis_95_var']
    report_df.loc['SUM ORIGIN EX CP/JS/US EQ', 'basis_99_var'] = var_df.loc[sum_origin_b_all, 'basis_99_var']

    sum_cotton_o_c = var_df[(var_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ[O]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    sum_cotton_o_all = var_df[(var_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ[O]') & (var_df['instrument_name'] == 'ALL')].index[0]
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'outright_pos'] = var_df.loc[sum_cotton_o_c, 'outright_pos']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'outright_95_var'] = var_df.loc[sum_cotton_o_all, 'outright_95_var']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'outright_99_var'] = var_df.loc[sum_cotton_o_all, 'outright_99_var']
    sum_cotton_b_c = var_df[(var_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ[B]') & (var_df['instrument_name'] == 'COTTON')].index[0]
    sum_cotton_b_all = var_df[(var_df['region_agg'] == 'SUM COTTON EX CP/JS/US EQ[B]') & (var_df['instrument_name'] == 'ALL')].index[0]
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'basis_pos'] = var_df.loc[sum_cotton_b_c, 'basis_pos']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'basis_95_var'] = var_df.loc[sum_cotton_b_all, 'basis_95_var']
    report_df.loc['SUM COTTON EX CP/JS/US EQ', 'basis_99_var'] = var_df.loc[sum_cotton_b_all, 'basis_99_var']

    exception_cotton_var_data_df = pd.DataFrame(columns=var_df.columns)
    sum_origin_ex_row = {
        'region_agg': 'SUM ORIGIN EX CP/JS/US EQ',
        'level': 'agg',
        'instrument_name': 'ALL',
        'outright_position_index_list': var_df.loc[sum_origin_o_all, 'outright_position_index_list'],
        'no_of_outright_positions': len(var_df.loc[sum_origin_o_all, 'outright_position_index_list']),
        'basis_position_index_list': var_df.loc[sum_origin_b_all, 'basis_position_index_list'],
        'no_of_basis_positions': len(var_df.loc[sum_origin_b_all, 'basis_position_index_list']),
    }
    sum_cotton_ex_row = {
        'region_agg': 'SUM COTTON EX CP/JS/US EQ',
        'level': 'agg',
        'instrument_name': 'ALL',
        'outright_position_index_list': var_df.loc[sum_cotton_o_all, 'outright_position_index_list'],
        'no_of_outright_positions': len(var_df.loc[sum_cotton_o_all, 'outright_position_index_list']),
        'basis_position_index_list': var_df.loc[sum_cotton_b_all, 'basis_position_index_list'],
        'no_of_basis_positions': len(var_df.loc[sum_cotton_b_all, 'basis_position_index_list']),
    }

    exception_cotton_var_data_df = pd.concat([exception_cotton_var_data_df, pd.DataFrame([sum_origin_ex_row, sum_cotton_ex_row])], ignore_index=True)
    pnl_analyzer = PnLAnalyzer(long_pnl_df, combined_pos_df)
    exception_cotton_var_data_df = calculate_var_for_regions(
        var_data_df=exception_cotton_var_data_df,
        analyzer=pnl_analyzer,
        simulation_method=simulation_method,
        calculation_method=calculation_method,
        cob_date=cob_date,
        window=window,
        is_two_tail=True
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


