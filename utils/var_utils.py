import pandas as pd

def get_cotton_region_aggregates(region_list: list[str]) -> dict[str, list[str]]:
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
    sum_origin_ex_cp_js_useq_ov = []  # If needed in future

    sum_cotton = sum_origin + sum_central
    sum_cotton_ex_cp_js_useq_outright = sum_origin_ex_cp_js_useq_outright + sum_central
    sum_cotton_ex_cp_js_useq_basis = sum_origin_ex_cp_js_useq_basis + sum_central
    sum_cotton_ex_cp_js_useq_ov = []  # If needed in future

    return {'SUM AFRICA': sum_africa, 'SUM USA': sum_usa,'SUM USA & MEXICO': sum_usa_mex, 'SUM AMERICAS': sum_americas,
            'SUM AUSTRALIA/CHINA': sum_aus_china, 'SUM AUSTRALIA/CHINA/INDIA': sum_aus_china_ind,
            'SUM MED REGION': sum_med, 'SUM CENTRAL': sum_central, 'SUM ORIGIN': sum_origin, 'SUM NON-USA': sum_non_usa,
            'SUM ORIGIN EX CP/JS/US EQ[O]': sum_origin_ex_cp_js_useq_outright,
            'SUM ORIGIN EX CP/JS/US EQ[B]': sum_origin_ex_cp_js_useq_basis,
            'SUM ORIGIN EX CP/JS/US EQ[OV]': sum_origin_ex_cp_js_useq_ov,
            'SUM COTTON': sum_cotton,
            'SUM COTTON EX CP/JS/US EQ[O]': sum_cotton_ex_cp_js_useq_outright,
            'SUM COTTON EX CP/JS/US EQ[B]': sum_cotton_ex_cp_js_useq_basis,
            'SUM COTTON EX CP/JS/US EQ[OV]': sum_cotton_ex_cp_js_useq_ov}

def calculate_unit_and_aggregate_var(var_calc, aggregate_unit_dict, cob_date, percentiles, window,
                                     unit_outright_lookback_pnl_df, unit_outright_inverse_pnl_df,
                                     unit_basis_lookback_pnl_df=None, unit_basis_inverse_pnl_df=None):
    unit_var_dict = {}
    aggregate_var_dict = {}

    # If basis data missing, create empty dfs to avoid errors
    if unit_basis_lookback_pnl_df is None:
        unit_basis_lookback_pnl_df = pd.DataFrame(index=unit_outright_lookback_pnl_df.index)
    if unit_basis_inverse_pnl_df is None:
        unit_basis_inverse_pnl_df = pd.DataFrame(index=unit_outright_inverse_pnl_df.index)

    # Construct overall pnl dfs by summing outright + basis
    unit_overall_lookback_pnl_df = unit_outright_lookback_pnl_df.add(unit_basis_lookback_pnl_df, fill_value=0)
    unit_overall_inverse_pnl_df = unit_outright_inverse_pnl_df.add(unit_basis_inverse_pnl_df, fill_value=0)

    for pct in percentiles:
        unit_var_dict[pct] = {}
        aggregate_var_dict[pct] = {}

        # Unit-level VaR
        unit_var_dict[pct]['outright'] = {unit: var_calc.calculate_historical_var(
            lookback_df=unit_outright_lookback_pnl_df[[unit]], inverse_df=unit_outright_inverse_pnl_df[[unit]],
            cob_date=cob_date, percentile=pct, window=window) for unit in unit_outright_lookback_pnl_df.columns}

        unit_var_dict[pct]['basis'] = {unit: var_calc.calculate_historical_var(
            lookback_df=unit_basis_lookback_pnl_df[[unit]], inverse_df=unit_basis_inverse_pnl_df[[unit]],
            cob_date=cob_date, percentile=pct, window=window) for unit in unit_basis_lookback_pnl_df.columns}

        unit_var_dict[pct]['overall'] = {unit: var_calc.calculate_historical_var(
            lookback_df=unit_overall_lookback_pnl_df[[unit]], inverse_df=unit_overall_inverse_pnl_df[[unit]],
            cob_date=cob_date, percentile=pct, window=window) for unit in unit_overall_lookback_pnl_df.columns}

        # Aggregate VaR
        for aggregate, units in aggregate_unit_dict.items():
            aggregate_var_dict[pct].setdefault('outright', {})
            aggregate_var_dict[pct].setdefault('basis', {})
            aggregate_var_dict[pct].setdefault('overall', {})

            # Helper function to sum and calc VaR if valid units exist
            def calc_agg_var(unit_list, lookback_df, inverse_df, category):
                valid_units = [u for u in unit_list if u in lookback_df.columns]
                missing_units = [u for u in unit_list if u not in lookback_df.columns]
                if missing_units:
                    print(f"[Warning] {category} - Missing units for aggregate '{aggregate}': {missing_units}")
                if valid_units:
                    agg_lookback = lookback_df[valid_units].sum(axis=1)
                    agg_inverse = inverse_df[valid_units].sum(axis=1)
                    aggregate_var_dict[pct][category][aggregate] = var_calc.calculate_historical_var(
                        lookback_df=agg_lookback, inverse_df=agg_inverse,
                        cob_date=cob_date, percentile=pct, window=window
                    )

            calc_agg_var(units, unit_outright_lookback_pnl_df, unit_outright_inverse_pnl_df, 'outright')
            calc_agg_var(units, unit_basis_lookback_pnl_df, unit_basis_inverse_pnl_df, 'basis')
            calc_agg_var(units, unit_overall_lookback_pnl_df, unit_overall_inverse_pnl_df, 'overall')

    return unit_var_dict, aggregate_var_dict

def format_var_results(unit_var_dict: dict, aggregate_var_dict: dict, cob_date: str, method: str,
                       exposure_override: str | None = None) -> pd.DataFrame:
    """
    Convert VaR dictionaries into a single long-format DataFrame.

    Args:
        unit_var_dict (dict): Standardized unit-level VaR results.
        aggregate_var_dict (dict): Standardized aggregate-level VaR results.
        cob_date (str): COB date for the VaR report.
        method (str): Methodology used (e.g., 'linear').
        exposure_override (str, optional): If provided, override the exposure label with this value.


    Returns:
        pd.DataFrame: Formatted long DataFrame with columns:
                      ['cob_date', 'level', 'unit_or_aggregate', 'exposure', 'percentile', 'var', 'method']
    """
    rows = []
    for level, var_dict in [('unit', unit_var_dict), ('aggregate', aggregate_var_dict)]:
        for percentile, exposure_dict in var_dict.items():
            for exposure, entity_dict in exposure_dict.items():
                for name, var_val in entity_dict.items():
                    rows.append({'cob_date': cob_date, 'level': level, 'unit_or_aggregate': name,
                                 'exposure': exposure_override.upper() if exposure_override else exposure.upper(),
                                 'percentile': percentile, 'var': round(var_val, 2), 'method': method})
    return pd.DataFrame(rows)
