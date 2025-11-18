import pandas as pd

from workflow.var.main_var_workflow import main_var_workflow
from workflow.shared.data_preparation_workflow import generate_ex_gin_s6_returns_df, generate_cotlook_returns_df
from position_loader.physical_position_loader import PhysicalPositionLoader
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from price_series_loader.vol_series_loader import VolLoader
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # # TODO: No rubber/cotton OPERA derivative positions on 2025-10-31
    cob_date = '2025-11-06' # TODO: Friday positions 2025-08-22 is x3. Need to fix.
    # Note: Greeks in DB only start from 20 Aug 2025
    product = 'cotton'
    method = 'linear'
    #method = 'non-linear_monte_carlo'
    window = 260
    main_var_workflow(cob_date=cob_date, product=product, method=method, window=window, with_price_var=True,
                      write_to_excel=True)

    # cob_date = '2025-11-06' # Only have ORS positions for 2025-10-31 and 2025-11-05, but no derivs for 2025-10-31.
    # product = 'rubber'
    # method = 'linear'
    # window = 260
    #
    # uat_engine = get_engine('uat')
    # derivatives_loader = DerivativesPositionLoader(date=cob_date, source=uat_engine)
    # deriv_pos_df = derivatives_loader.load_position(
    #     date=cob_date,
    #     trader_id=None,
    #     counterparty_id=None,
    #     product=product,
    #     book=None
    # )
    # deriv_pos_df = deriv_pos_df[~deriv_pos_df['security_id'].astype(str).str.startswith('CR')]



    # main_var_workflow(cob_date=cob_date, product=product, method=method, window=window, with_price_var=False,
    #                   write_to_excel=True)

    # product = 'rms'
    # cob_date = '2025-11-07'
    # method = 'taylor_series'
    # window = 260
    #
    # main_var_workflow(cob_date=cob_date, product=product, method=method, window=window, with_price_var=False,
    #                  write_to_excel=True)



