from workflow.cotton_var_report_workflow import cotton_var_report_workflow
from workflow.options_repricing_workflow import options_workflow
import pandas as pd
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from utils.contract_utils import custom_monthly_contract_sort_key
from datetime import datetime
import math
from scipy.stats import norm

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    cob_date = '2025-08-05'
    cotton_var_report_workflow(method='linear', cob_date=cob_date)
    # options_workflow()
