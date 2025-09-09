import pandas as pd
import numpy as np

def build_var_report(product: str, books: str, cob_date: str, pos_df: pd.DataFrame, var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the product type and output from generate_var_workflow(),
    format and write a custom report.
    """
    columns = ['unit', 'outright_pos', 'basis_pos', 'outright_95_VaR', 'basis_95_VaR', 'overall_95_VaR',
               'outright_99_VaR', 'basis_99_VaR', 'overall_99_VaR']

    unique_units = pos_df['region'].unique()
    report_df = pd.DataFrame(columns=columns, index=unique_units)
    report_df['region'] = unique_units

    for unit in unique_units:
        # Extract position info
        # TODO: To convert to position to MT eventually
        outright_pos = pos_df[(pos_df['exposure'] == 'OUTRIGHT') & (pos_df['region'] == unit)]['delta'].sum()
        basis_pos = pos_df[(pos_df['exposure'] == 'BASIS (NET PHYS)') & (pos_df['region'] == unit)]['delta'].sum()

        def get_var(exposure, percentile):
            result = var_df[(var_df['unit_or_aggregate'] == unit) & (var_df['exposure'] == exposure)
                            & (var_df['percentile'] == percentile)]['var']
            return result.values[0] if not result.empty else np.nan

        report_df.loc[unit, 'outright_pos'] = outright_pos
        report_df.loc[unit, 'basis_pos'] = basis_pos
        report_df.loc[unit, 'outright_95_VaR'] = get_var(exposure='OUTRIGHT', percentile=95)
        report_df.loc[unit, 'basis_95_VaR'] = get_var(exposure='BASIS (NET PHYS)', percentile=95)
        report_df.loc[unit, 'overall_95_VaR'] = get_var(exposure='OVERALL', percentile=95)
        report_df.loc[unit, 'outright_99_VaR'] = get_var(exposure='OUTRIGHT', percentile=99)
        report_df.loc[unit, 'basis_99_VaR'] = get_var(exposure='BASIS (NET PHYS)', percentile=99)
        report_df.loc[unit, 'overall_99_VaR'] = get_var(exposure='OVERALL', percentile=99)
    print(report_df)

    filename = f"{cob_date}_var_output.xlsx"
    with pd.ExcelWriter(filename, mode='a', if_sheet_exists='replace') as writer:
        report_df.to_excel(writer, sheet_name=books)

    return report_df
