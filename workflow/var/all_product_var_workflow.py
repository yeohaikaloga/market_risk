from workflow.shared.positions.positions_workflow import build_combined_position, prepare_positions_data_for_var
from workflow.shared.market_data.market_data_workflow import build_product_prices_returns_dfs_for_mc_sim
from workflow.shared.pnl_calculator_workflow import generate_pnl_vectors, analyze_and_export_unit_pnl
from workflow.var.var_generator_workflow import (generate_var, build_var_report, build_cotton_var_report_exceptions,
                                                 build_cotton_price_var_report_exceptions,
                                                 build_rubber_var_report_exceptions)
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow
from price_series_generator.simulated_returns_series_generator import SimulatedReturnsSeriesGenerator
from sensitivity_matrix_loader.sensitivity_matrix_loader import SensitivityMatrixLoader
import pandas as pd
import os
from utils.log_utils import get_logger
from utils.file_utils import (create_output_directory, get_full_path, save_to_pickle_in_dir, load_from_pickle_in_dir,
                              save_to_csv_in_dir, load_from_csv_in_dir)
from utils.contract_utils import product_specifications
from db.db_connection import get_engine
import shutil
from typing import List, Optional, Union, Dict, Any

logger = get_logger(__name__)
def all_product_var_workflow(cob_date: str, product: Optional[Union[str, List[str]]] = None,
                             calculation_method: Optional[Union[str, List[str]]] = None,
                             simulation_method: Optional[Union[str, List[str]]] = None):
    """
    Performs VaR workflow, optionally filtered by specific product(s),
    calculation method(s), or simulation method(s).
    """
    product_dict = {'cotton': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear', 'sensitivity_matrix']},
                    'rubber': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear', 'sensitivity_matrix']},
                    'biocane': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear']},
                    'wood': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear']},
                    'rms': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['taylor_series']}}
    create_output_directory(cob_date)
    required_files = [
        f'prices_{cob_date}.pkl',
        f'hist_relative_returns_{cob_date}.pkl',
        f'hist_w_abs_returns_{cob_date}.pkl',
        f'fx_spot_{cob_date}.pkl',
        f'instrument_dict_{cob_date}.pkl'
    ]
    files_exist = all(os.path.exists(get_full_path(cob_date, filename)) for filename in required_files)
    if not files_exist:
        historical_market_data_preparation_workflow(cob_date, save_to_pickle=True, save_to_csv=True)
    else:
        logger.info(f"Skipping market_data_preparation_workflow for {cob_date}: required files already exist.")

    # --- INPUT NORMALIZATION & FILTERING ---

    # 1. Normalize user product input
    user_products_list = [product] if isinstance(product, str) else (product or [])

    # Determine which products to iterate over
    if user_products_list:
        products_to_run = [p for p in user_products_list if p in product_dict]
        if not products_to_run:
            logger.error(f"None of the specified products {user_products_list} are valid. Aborting run.")
            return
    else:
        products_to_run = list(product_dict.keys())

    # 2. Normalize user method inputs
    user_calcs_list = [calculation_method] if isinstance(calculation_method, str) else (calculation_method or [])
    user_sims_list = [simulation_method] if isinstance(simulation_method, str) else (simulation_method or [])

    # --- EXECUTION LOOP ---

    for current_product in products_to_run:
        product_config = product_dict[current_product]

        # Determine valid calculation methods for this product, filtered by user input
        all_calcs = product_config.get('calculation_method', [])
        if user_calcs_list:
            # Intersection of user request and product configuration
            calcs_to_run = [c for c in user_calcs_list if c in all_calcs]
        else:
            calcs_to_run = all_calcs

        if not calcs_to_run:
            logger.warning(
                f"Skipping {current_product}: No valid calculation_methods found between requested ({user_calcs_list}) "
                f"and available ({all_calcs})."
            )
            continue

        # Determine valid simulation methods for this product, filtered by user input
        all_sims = product_config.get('simulation_method', [])
        if user_sims_list:
            # Intersection of user request and product configuration
            sims_to_run = [s for s in user_sims_list if s in all_sims]
        else:
            sims_to_run = all_sims

        if not sims_to_run:
            logger.warning(
                f"Skipping {current_product}: No valid simulation_methods found between requested ({user_sims_list}) "
                f"and available ({all_sims})."
            )
            continue

        # Main execution
        for current_calc_method in calcs_to_run:
            for current_sim_method in sims_to_run:
                product_position_workflow(cob_date, current_product, current_calc_method, current_sim_method)
                pnl_and_var_workflow(cob_date, current_product, current_calc_method, current_sim_method)


def historical_market_data_preparation_workflow(cob_date: str, save_to_pickle: bool, save_to_csv: bool):
    product = 'cotton' #TODO Need to look into this! If simulation_method == 'mc_sim', then product is irrelevant.
    window = 260
    simulation_method = 'mc_sim'
    # STEP 1: Market data preparation
    prices_df, returns_df, fx_spot_df, instrument_dict = build_product_prices_returns_dfs_for_mc_sim(cob_date, product,
                                                                                                     window,
                                                                                                     simulation_method)
    save_to_pickle_in_dir(returns_df, cob_date, f'hist_relative_returns_{cob_date}.pkl')
    basis_df = fy24_cotton_basis_workflow(
        cob_date=cob_date,
        window=window,
        write_to_excel=True,
        apply_smoothing=True
    )
    basis_df = basis_df.reindex(returns_df.index)
    absolute_returns_df = basis_df
    absolute_returns_df.columns = absolute_returns_df.columns.str.replace(' final AR series', '', regex=False)
    absolute_returns_df.columns = absolute_returns_df.columns.str.replace(' final AR (sm) series', '', regex=False)
    absolute_returns_df.columns = absolute_returns_df.columns + '_abs'
    instrument_dict['BASIS'] = {}
    instrument_dict['BASIS']['abs_returns_$_df'] = absolute_returns_df
    returns_df = pd.concat([returns_df, absolute_returns_df], axis=1)

    if save_to_pickle:
        save_to_pickle_in_dir(prices_df, cob_date, f'prices_{cob_date}.pkl')
        save_to_pickle_in_dir(returns_df, cob_date, f'hist_w_abs_returns_{cob_date}.pkl')
        save_to_pickle_in_dir(fx_spot_df, cob_date, f'fx_spot_{cob_date}.pkl')
        save_to_pickle_in_dir(instrument_dict, cob_date, f'instrument_dict_{cob_date}.pkl')

    if save_to_csv:
        save_to_csv_in_dir(prices_df, cob_date, f'prices_{cob_date}.csv')
        save_to_csv_in_dir(returns_df, cob_date, f'hist_w_abs_returns_{cob_date}.csv')
        save_to_csv_in_dir(fx_spot_df, cob_date, f'fx_spot_{cob_date}.csv')
        save_to_csv_in_dir(instrument_dict, cob_date, f'instrument_dict_{cob_date}.csv')


def product_position_workflow(cob_date: str, product: str, calculation_method: str, simulation_method: str):
    prices_df = load_from_pickle_in_dir(cob_date, f'prices_{cob_date}.pkl')
    fx_spot_df = load_from_pickle_in_dir(cob_date, f'fx_spot_{cob_date}.pkl')
    instrument_dict = load_from_pickle_in_dir(cob_date, f'instrument_dict_{cob_date}.pkl')

    # STEP 2: Product-specific position preparation
    combined_pos_df = build_combined_position(cob_date, product, instrument_dict, prices_df, fx_spot_df)
    combined_pos_df = prepare_positions_data_for_var(
        combined_pos_df=combined_pos_df,
        price_df=prices_df,
        cob_date=cob_date,
        simulation_method=simulation_method,
        calculation_method=calculation_method,
        trader=False,
        counterparty=False)
    save_to_pickle_in_dir(combined_pos_df, cob_date, f'{product}_combined_pos_{cob_date}.pkl')
    if product == 'cotton':
        combined_price_pos_df = combined_pos_df[combined_pos_df['book'] == 'PRICE']
        save_to_pickle_in_dir(combined_price_pos_df, cob_date, f'combined_price_pos_{cob_date}.pkl')
        save_to_csv_in_dir(combined_price_pos_df, cob_date, f'combined_price_pos_{cob_date}.csv')

def monte_carlo_market_data_preparation_workflow(cob_date: str):
    simulated_returns_filename = 'daily_simulated_matrix_cotton_' + cob_date.replace('-', '') + '.pickle'
    destination_path = get_full_path(cob_date, simulated_returns_filename)
    file_exist = os.path.exists(destination_path)
    if not file_exist:
        dir_date = cob_date.replace('-', '')
        backup_dir = r"C:\Users\haikal.yeo\OneDrive - Olam Global Agri Pte Ltd\OGA_MR_GOO - Daily Time Series - Universe"
        dir = os.path.join(backup_dir, dir_date)
        source_path = os.path.join(dir, simulated_returns_filename)
        if os.path.exists(source_path):
            try:
                # Copy the file from the backup location to the output directory
                shutil.copy2(source_path, destination_path)
                logger.info(f"Successfully copied missing simulated returns matrix from backup: {source_path}")
            except Exception as e:
                logger.error(f"Error copying file {source_path} to {destination_path}: {e}")
                raise RuntimeError(f"Failed to copy required simulation file for {cob_date}.")
        else:
            error_msg = (f"Error: Required simulated returns file '{simulated_returns_filename}' is "
                         f"missing in the output directory and backup location ({dir}). "
                         f"Cannot proceed with 'mc_sim'.")
            logger.error(error_msg)
    else:
        logger.info(f"Simulated return matrix for {cob_date} already in folder.")
    #returns_df = load_from_csv_in_dir(cob_date, f'simulatedRet_df_ld_20260102.csv')
    relevant_risk_factors = (['CT', 'VV', 'AVY', 'OR', 'JN', 'SRB', 'BDR', 'RG', 'RT', 'C', 'W', 'S', 'AIndex',
                             'MeOrTe', 'IvCoMa', 'BuFaBo', 'BrCo', 'Shankar6', 'GaSu', 'MaizeUP', 'SawnAvg'] +
                             product_specifications['rms']['instrument_list'])
    mc_returns_generator = (SimulatedReturnsSeriesGenerator.
                            load_relevant_simulated_returns(cob_date, simulated_returns_filename,
                                                            relevant_risk_factors))
    returns_df = mc_returns_generator.price_df
    #returns_df = load_from_pickle_in_dir(cob_date, f'hist_relative_returns_{cob_date}.pkl')

    return returns_df

def sensitivity_matrix_preparation_workflow(cob_date: str, product: str):
    uat_engine = get_engine('uat')
    if product in ['cotton', 'rubber']:
        sensitivity_matrix = SensitivityMatrixLoader(cob_date, product, uat_engine)
        sensitivity_matrix.load_sensitivity_matrix(product)
        return sensitivity_matrix

def pnl_and_var_workflow(cob_date: str, product: str, calculation_method: str, simulation_method: str):
    excel_filename = f'{calculation_method}_{simulation_method}_{product}_var_{cob_date}.xlsx'
    window = 260
    full_path_to_excel = get_full_path(cob_date, excel_filename)

    # STEP 3: Load MC sim returns
    if simulation_method == 'mc_sim':
        #returns_df = load_from_pickle_in_dir(cob_date, f'hist_relative_returns_{cob_date}.pkl')
        returns_df = monte_carlo_market_data_preparation_workflow(cob_date)
    else:
        returns_df = load_from_pickle_in_dir(cob_date, f'hist_w_abs_returns_{cob_date}.pkl')

    prices_df = load_from_pickle_in_dir(cob_date, f'prices_{cob_date}.pkl')
    combined_pos_df = load_from_pickle_in_dir(cob_date, f'{product}_combined_pos_{cob_date}.pkl')

    if len(combined_pos_df) > 0:
        # STEP 4: Calculate PnL
        if (calculation_method == 'linear' or calculation_method == 'sensitivity_matrix'
                or calculation_method == 'taylor_series'):
            long_pnl_df = generate_pnl_vectors(
                combined_pos_df=combined_pos_df,
                returns_df=returns_df,
                method=calculation_method
            )
            logger.info('Main PnL shape: ', str(combined_pos_df.shape))
            if product == 'cotton':
                combined_price_pos_df = load_from_pickle_in_dir(cob_date, f'combined_price_pos_{cob_date}.pkl')
                long_price_pnl_df = generate_pnl_vectors(
                    combined_pos_df=combined_price_pos_df,
                    returns_df=returns_df,
                    method=calculation_method
                )
                logger.info('Price PnL shape: ', str(combined_price_pos_df.shape))
            logger.info("STEP 3: PnL prepared")
        else:
            logger.error(f"Method '{calculation_method}' not supported yet.")

        if simulation_method == 'hist_sim':
            is_truncated = False
        elif simulation_method == 'mc_sim':
            is_truncated = True

        analyze_and_export_unit_pnl(
            product=product,
            returns_df=returns_df,
            prices_df=prices_df,
            long_pnl_df=long_pnl_df,
            combined_pos_df=combined_pos_df,
            full_path=full_path_to_excel,
            write_to_excel=True,
            is_truncated=is_truncated,
            write_to_feather_for_oga_level_var=True)

        var_data_df = generate_var(
            product=product,
            combined_pos_df=combined_pos_df,
            long_pnl_df=long_pnl_df,
            simulation_method=simulation_method,
            calculation_method=calculation_method,
            cob_date=cob_date,
            window=window
        )

        if product == 'cotton':
            price_var_data_df = generate_var(
                product=product,
                combined_pos_df=combined_price_pos_df,
                long_pnl_df=long_price_pnl_df,
                simulation_method=simulation_method,
                calculation_method=calculation_method,
                cob_date=cob_date,
                window=window
            )
            logger.info(f"STEP 4: PnL calculated")

        # Step 5: Calculate VaR
        var_report_df = build_var_report(var_df=var_data_df)
        if product == 'cotton':
            price_var_report_df = build_var_report(var_df=price_var_data_df)

        # Apply product-specific exceptions or overrides
        if product == 'cotton':
            var_report_df = build_cotton_var_report_exceptions(long_pnl_df=long_pnl_df,
                                                               combined_pos_df=combined_pos_df,
                                                               simulation_method=simulation_method,
                                                               calculation_method=calculation_method,
                                                               report_df=var_report_df,
                                                               var_df=var_data_df,
                                                               cob_date=cob_date,
                                                               window=window)
            price_var_report_df = build_cotton_price_var_report_exceptions(report_df=price_var_report_df)
            logger.info("STEP 5: Cotton VaR report done")
        elif product == 'rubber':
            var_report_df = build_rubber_var_report_exceptions(report_df=var_report_df)
            logger.info("STEP 5: Rubber VaR report done")

        full_path_to_excel = get_full_path(cob_date, excel_filename)
        writer_kwargs = {'mode': 'w'}  # Default to write mode (for new file)

        if os.path.exists(full_path_to_excel):
            writer_kwargs['mode'] = 'a'
            writer_kwargs['if_sheet_exists'] = 'replace'

        # Use the full path for the Excel writer, passing arguments dynamically
        with pd.ExcelWriter(full_path_to_excel, **writer_kwargs) as writer:
            var_report_df.to_excel(writer, sheet_name='var', index=True)
            if product == 'cotton':
                price_var_report_df.to_excel(writer, sheet_name='price_var', index=True)

        logger.info(f"STEP 5: VaR Report saved successfully to {full_path_to_excel}")

    else:
        logger.warning(f'No positions for {product}. PnL and VaR not calculated.')
