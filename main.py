from workflow.cotton_var_report_workflow import cotton_var_report_workflow
from workflow.options_repricing_workflow import options_workflow
import pandas as pd


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # cotton_var_report_workflow('linear')
    options_workflow()
