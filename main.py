import pandas as pd
import traceback
from workflow.var.historical_var_workflow import historical_var_workflow

from position_loader.physical_position_loader import PhysicalPositionLoader
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from price_series_loader.vol_series_loader import VolLoader
from position_loader.derivatives_position_loader import DerivativesPositionLoader
from utils.date_utils import get_prev_biz_days_list
from db.db_connection import get_engine
from workflow.shared.forex_workflow import load_forex
from monte_carlo_simulations.simulator import simulate_ret
from utils.contract_utils import load_instrument_ref_dict
from utils.date_utils import load_opera_market_calendar
from price_series_generator.generic_curve_generator import GenericCurveGenerator
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from workflow.shared.market_data.market_data_workflow import (build_cotlook_relative_returns,
                                                              build_average_wood_returns, build_garmmz_sugar_returns,
                                                              build_maize_up_returns, build_biocane_returns)


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    try:

        # cob_date = '2025-11-21'
        # instrument_list = ['OR']
        # window = 260
        # usd_conversion_mode = 'pre'
        # market_calendar = load_opera_market_calendar(instrument_list)
        # instrument_ref_dict = load_instrument_ref_dict('uat')

        # instrument_name = 'OR'

        # cob_date = '2025-11-21'
        # window = 260
        # prod_engine = get_engine('prod')
        # fx_spot_df = load_forex(cob_date=cob_date, window=window)
        # cotton_dict = build_cotlook_relative_returns(cob_date, window)
        # print('cotton done')
        # wood_dict = build_average_wood_returns(cob_date, window)
        # biocane_dict = build_biocane_returns(cob_date, window)
        # print('done')

        #
        # ### THIS IS FOR MC SIMULATIONS OF RETURNS
        # ret_df = pd.read_excel('ref_df_sample.xlsx', index_col=0)
        # ld = 0.94
        # output = simulate_ret(ret_df, ld, no_of_observations=30000, lookback=260, std_dev_estimation_method="exponential",
        #              is_random_seed=True, asarray=True, random_type="uniform", fill_random=None, seed=None)
        # print(output)
        # output[1].to_csv('output.csv')


        #### THIS IS FOR COTTON VAR
        # cob_date = '2025-11-06'
        # # Sensitivity report not cleaned up for 2025-11-06
        # # Note: Greeks in DB only start from 20 Aug 2025
        # product = 'cotton'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                             calculation_method=calculation_method, window=window,
        #                             with_price_var=True, write_to_excel=True)

        #### THIS IS FOR RUBBER VAR
        # cob_date = '2025-11-21' # Only have ORS positions for 2025-10-31 and 2025-11-05, but no derivs for 2025-10-31.
        # # Sensitivity report not cleaned up for 2025-11-06 double entry opera sensitivity; no entry for 2025-11-07
        # product = 'rubber'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                         calculation_method=calculation_method, window=window,
        #                         with_price_var=False, write_to_excel=True)

        #### THIS IS FOR RMS VAR
        product = 'rms'
        cob_date = '2025-11-12'
        simulation_method = 'hist_sim'
        calculation_method = 'taylor_series'
        window = 260

        historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
                                    calculation_method=calculation_method, window=window,
                                    with_price_var=False, write_to_excel=True)
        ## THIS IS FOR BIOCANE VAR
        # product = 'biocane'
        # cob_date = '2025-11-21'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        #
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                         calculation_method=calculation_method, window=window,
        #                         with_price_var=False, write_to_excel=True)

        # ### THIS IS FOR WOOD VAR
        # product = 'wood'
        # cob_date = '2025-11-21'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        #
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                         calculation_method=calculation_method, window=window,
        #                         with_price_var=False, write_to_excel=True)
    except:
         traceback.print_exc()
         raise
