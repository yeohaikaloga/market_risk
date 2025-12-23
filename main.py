import pandas as pd
import traceback
from workflow.var.historical_var_workflow import historical_var_workflow
from workflow.var.monte_carlo_var_workflow import monte_carlo_var_workflow
from utils.log_utils import setup_logging, get_logger
from utils.file_utils import get_full_path, load_from_feather_in_dir
from typing import List, Dict

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
from price_series_generator.simulated_returns_series_generator import SimulatedReturnsSeriesGenerator
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader
from workflow.shared.market_data.market_data_workflow import (build_cotlook_relative_returns,
                                                              build_average_wood_returns, build_garmmz_sugar_returns,
                                                              build_maize_up_returns, build_biocane_returns)
from workflow.var.all_product_var_workflow import all_product_var_workflow
from sensitivity_matrix_loader.sensitivity_matrix_loader import SensitivityMatrixLoader


if __name__ == '__main__':

    logger = get_logger(__name__)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    try:
        logger.info("BEGIN RUN")

        # cob_date = '2025-12-08'
        # date_formatted = cob_date.replace('-', '')
        # base_name = f'eo_simulation_vector_{date_formatted}'
        # filename_with_ext = f"{base_name}.feather"
        # try:
        #     data_frame = load_from_feather_in_dir(cob_date, filename_with_ext)
        # except Exception as e:
        #     print(f"ERROR: Could not load EO simulation vector for {cob_date}.")
        #     raise
        # print("done")
        app_name = 'VaR_Calculator_2026'
        cob_date = '2025-12-12'
        log_file = 'summary_logs_' + cob_date + '.txt'
        setup_logging(app_name=app_name, log_filename=log_file)
        logger = get_logger(__name__)

        all_product_var_workflow(cob_date, product=['rubber'], simulation_method=['hist_sim', 'mc_sim'],
                                 calculation_method=['linear', 'sensitivity_matrix'])
        # all_product_var_workflow(
        #     cob_date,
        #     product='rms',
        #     calculation_method=['taylor_series'],
        #     simulation_method=['hist_sim']
        # )
        # TODO save feather file for cotton to only include outright ex JS/US EQ
        # TODO rollback to old position for wood and biocane when there is no position
        # TODO print logger, especially warning messages for checking
        # all_product_var_workflow(
        #     cob_date,
        #     product=['biocane', 'wood'],
        #     calculation_method=['linear'],
        #     simulation_method=['hist_sim']
        # )

        # all_product_var_workflow(
        #     cob_date,
        #     product='rubber',
        #     calculation_method=['linear', 'sensitivity_matrix'],
        #     simulation_method=['hist_sim']
        # )

        # relevant_risk_factors = ['CT', 'VV', 'AVY', 'OR', 'JN', 'SRB', 'BDR', 'RG', 'RT', 'AIndex', 'MeOrTe', 'IvCoMa', 'BuFaBo', 'BrCo', 'Shankar6', 'GaSu', 'MaizeUP', 'SawnAvg']
        # filename = 'daily_simulated_matrix_20251201'
        # mc_returns_generator = SimulatedReturnsSeriesGenerator.load_relevant_simulated_returns(filename, relevant_risk_factors)
        # print(mc_returns_generator.price_df.head())
        # print(mc_returns_generator.contracts)
        # print('done')


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
        #ret_df = pd.read_excel('ref_df_sample.xlsx', index_col=0)

        #output_df = pd.read_csv('output.csv', index_col=0)
        # ret_df = pd.read_csv('ref_df_20251124.csv', index_col=0)
        # ld = 0.95
        # output = simulate_ret(ret_df, ld, no_of_observations=30000, lookback=260, std_dev_estimation_method="exponential",
        #              is_random_seed=True, asarray=True, random_type="uniform", fill_random=None, seed=1416349404)
        # print(output)
        # output[1].to_csv('output.csv')


        #### THIS IS FOR COTTON VAR
        # cob_date = '2025-12-01'
        # product = 'cotton'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                             calculation_method=calculation_method, window=window,
        #                             with_price_var=True, write_to_excel=True)
        #
        # simulation_method = 'mc_sim'
        # monte_carlo_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                         calculation_method=calculation_method, window=window,
        #                         with_price_var=True, write_to_excel=True)
        #
        # ### THIS IS FOR RUBBER VAR
        # cob_date = '2025-12-01'
        # product = 'rubber'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                         calculation_method=calculation_method, window=window,
        #                         with_price_var=False, write_to_excel=True)
        #
        # simulation_method = 'mc_sim'
        # monte_carlo_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                          calculation_method=calculation_method, window=window,
        #                          with_price_var=False, write_to_excel=True)

        #### THIS IS FOR RMS VAR
        # product = 'rms'
        # cob_date = '2025-11-28'
        # simulation_method = 'hist_sim'
        # calculation_method = 'taylor_series'
        # window = 260
        #
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                             calculation_method=calculation_method, window=window,
        #                             with_price_var=False, write_to_excel=True)
        # THIS IS FOR BIOCANE VAR
        # product = 'biocane'
        # cob_date = '2025-11-21'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        #
        # # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        # #                         calculation_method=calculation_method, window=window,
        # #                         with_price_var=False, write_to_excel=True)
        #
        # simulation_method = 'mc_sim'
        # monte_carlo_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                          calculation_method=calculation_method, window=window,
        #                          with_price_var=False, write_to_excel=True)
        # #
        # # # ### THIS IS FOR WOOD VAR
        # product = 'wood'
        # cob_date = '2025-11-21'
        # simulation_method = 'hist_sim'
        # calculation_method = 'linear'
        # window = 260
        #
        # historical_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                         calculation_method=calculation_method, window=window,
        #                         with_price_var=False, write_to_excel=True)
        #
        # simulation_method = 'mc_sim'
        # monte_carlo_var_workflow(cob_date=cob_date, product=product, simulation_method=simulation_method,
        #                          calculation_method=calculation_method, window=window,
        #                          with_price_var=False, write_to_excel=True)
        logger.info("END RUN")
    except:
        traceback.print_exc()