from workflow.shared.positions.positions_workflow import build_combined_position, prepare_positions_data_for_var
from workflow.shared.market_data.market_data_workflow import build_product_prices_returns_dfs_for_mc_sim
from workflow.shared.pnl_calculator_workflow import generate_pnl_vectors, analyze_and_export_unit_pnl
from workflow.var.var_generator_workflow import (generate_var, build_var_report, build_cotton_var_report_exceptions,
                                                 build_cotton_price_var_report_exceptions,
                                                 build_rubber_var_report_exceptions)
from workflow.cotton_basis_calculator_workflow import fy24_cotton_basis_workflow
from price_series_generator.simulated_returns_series_generator import SimulatedReturnsSeriesGenerator
import pandas as pd
import os
from utils.log_utils import get_logger
from utils.file_utils import create_output_directory, get_full_path, save_to_pickle_in_dir, load_from_pickle_in_dir
import shutil

def all_product_var_workflow(cob_date: str):
    logger = get_logger(__name__)
    product_dict = {'cotton': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear']},
                    'rubber': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear']},
                    'biocane': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear']},
                    'wood': {'simulation_method': ['hist_sim', 'mc_sim'], 'calculation_method': ['linear']}}
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
        market_data_preparation_workflow(cob_date)
    else:
        logger.info(f"Skipping market_data_preparation_workflow for {cob_date}: required files already exist.")

    for product in product_dict.keys():
        calculation_method = product_dict[product]['calculation_method'][0]
        for simulation_method in product_dict[product]['simulation_method']:
            product_position_workflow(cob_date, product, simulation_method)
            pnl_and_var_workflow(cob_date, product, calculation_method, simulation_method)


def market_data_preparation_workflow(cob_date: str):
    logger = get_logger(__name__)
    product = 'cotton' #TODO Need to look into this!
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

    save_to_pickle_in_dir(prices_df, cob_date, f'prices_{cob_date}.pkl')
    save_to_pickle_in_dir(returns_df, cob_date, f'hist_w_abs_returns_{cob_date}.pkl')
    save_to_pickle_in_dir(fx_spot_df, cob_date, f'fx_spot_{cob_date}.pkl')
    save_to_pickle_in_dir(instrument_dict, cob_date, f'instrument_dict_{cob_date}.pkl')


def product_position_workflow(cob_date: str, product: str, simulation_method:str):
    logger = get_logger(__name__)
    prices_df = load_from_pickle_in_dir(cob_date, f'prices_{cob_date}.pkl')
    fx_spot_df = load_from_pickle_in_dir(cob_date, f'fx_spot_{cob_date}.pkl')
    instrument_dict = load_from_pickle_in_dir(cob_date, f'instrument_dict_{cob_date}.pkl')

    # STEP 2: Product-specific position preparation
    combined_pos_df = build_combined_position(cob_date, product, instrument_dict, prices_df, fx_spot_df)
    calculation_method = 'linear'  # TODO Change to sensitivity report when ready
    combined_pos_df = prepare_positions_data_for_var(
        product=product,
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


def pnl_and_var_workflow(cob_date: str, product: str, calculation_method: str, simulation_method: str):
    logger = get_logger(__name__)
    excel_filename = f'{calculation_method}_{simulation_method}_{product}_var_{cob_date}.xlsx'
    window = 260
    full_path_to_excel = get_full_path(cob_date, excel_filename)

    # STEP 3: Load MC sim returns
    if simulation_method == 'mc_sim':

        simulated_returns_filename = 'daily_simulated_matrix_' + cob_date.replace('-', '') + '.pickle'
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
                raise FileNotFoundError(error_msg)
        else:
            logger.info(f"Simulated return matrix for {cob_date} already in folder.")

        relevant_risk_factors = ['CT', 'VV', 'AVY', 'OR', 'JN', 'SRB', 'BDR', 'RG', 'RT', 'C', 'W', 'S', 'AIndex',
                                 'MeOrTe', 'IvCoMa', 'BuFaBo', 'BrCo', 'Shankar6', 'GaSu', 'MaizeUP', 'SawnAvg']
        mc_returns_generator = (SimulatedReturnsSeriesGenerator.
                                load_relevant_simulated_returns(cob_date, simulated_returns_filename,
                                                                relevant_risk_factors))
        returns_df = mc_returns_generator.price_df.head() #TODO To remove .head() when going into production!
        # returns_df = load_from_pickle_in_dir(cob_date, f'hist_relative_returns_{cob_date}.pkl')
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
            print('Main PnL shape: ', str(combined_pos_df.shape))
            if product == 'cotton':
                combined_price_pos_df = load_from_pickle_in_dir(cob_date, f'combined_price_pos_{cob_date}.pkl')
                long_price_pnl_df = generate_pnl_vectors(
                    combined_pos_df=combined_price_pos_df,
                    returns_df=returns_df,
                    method=calculation_method
                )
                print('Price PnL shape: ', str(combined_price_pos_df.shape))
            logger.info("STEP 3: PnL prepared")
        else:
            raise NotImplementedError(f"Method '{calculation_method}' not supported yet.")


        analyze_and_export_unit_pnl(
            product=product,
            returns_df=returns_df,
            prices_df=prices_df,
            long_pnl_df=long_pnl_df,
            combined_pos_df=combined_pos_df,
            position_index_list=[],
            full_path_to_excel=full_path_to_excel,
            write_to_excel=True)

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
        logger.info(f'No positions for {product}. PnL and VaR not calculated.')
