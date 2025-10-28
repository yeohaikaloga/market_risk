import pandas as pd

from workflow.var.main_var_workflow import main_var_workflow
from workflow.shared.data_preparation_workflow import generate_ex_gin_s6_returns_df, generate_cotlook_returns_df, test
from position_loader.physical_position_loader import PhysicalPositionLoader
from db.db_connection import get_engine
if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # cob_date = '2025-08-22'
    # # Note: Greeks in DB only start from 20 Aug 2025
    # product = 'cotton'
    # method = 'linear'
    # #method = 'non-linear_monte_carlo'
    # window = 260
    # main_var_workflow(cob_date=cob_date, product=product, method=method, window=window, with_price_var=True, write_to_excel=True)

    # cob_date = '2025-10-14'
    # product = 'rubber'
    # method = 'linear'
    # window = 260
    # main_var_workflow(cob_date=cob_date, product=product, method=method, window=window, with_price_var=False,
    #                   write_to_excel=True)

    #NOTE rms VaR will continue to use Taylor series expansion
    product = 'rms'
    cob_date = '2025-10-16'
    method = 'taylor_series'
    window = 260

    test()


    main_var_workflow(cob_date=cob_date, product=product, method=method, window=window, with_price_var=False,
                      write_to_excel=True)
    #generate_rms_var_dict = generate_var_workflow(product=product, method=method, cob_date=cob_date, window=window)



