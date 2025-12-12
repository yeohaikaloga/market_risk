from workflow.shared.positions.positions_workflow import build_combined_position, prepare_positions_data_for_var
from workflow.shared.market_data.market_data_workflow import build_product_prices_returns_dfs_for_hist_sim
from workflow.shared.pnl_calculator_workflow import generate_pnl_vectors, analyze_and_export_unit_pnl
from workflow.var.var_generator_workflow import (generate_var, build_var_report, build_cotton_var_report_exceptions,
                                                 build_cotton_price_var_report_exceptions,
                                                 build_rubber_var_report_exceptions)
import pandas as pd
import os
from datetime import datetime
from utils.log_utils import get_logger, save_to_pickle, load_from_pickle

def historical_var_workflow(cob_date: str, product: str, simulation_method: str, calculation_method: str, window: int,
                            with_price_var: bool, write_to_excel: bool):
    logger = get_logger(__name__)
    logger.info(f"Running historical VaR workflow for product: {product}, COB: {cob_date}, "
                f"Simulation Method: {simulation_method}, Calculation Method: {calculation_method}, Window: {window}")

    filename = f"{cob_date}_{product[:3]}_{simulation_method}_{calculation_method}_var_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # === STEP 1: Prepare Market Data ===
    # prices_df, returns_df, fx_spot_df, instrument_dict = build_product_prices_returns_dfs_for_hist_sim(cob_date, product, window,
    #                                                                                                     simulation_method)
    # save_to_pickle(prices_df, f'prices_{product}_{simulation_method}_{cob_date}.pkl')
    # save_to_pickle(returns_df, f'returns_{product}_{simulation_method}_{cob_date}.pkl')
    # save_to_pickle(fx_spot_df, f'fx_spot_{product}_{simulation_method}_{cob_date}.pkl')
    # save_to_pickle(instrument_dict, f'instrument_dict_{product}_{simulation_method}_{cob_date}.pkl')
    prices_df = load_from_pickle(f'prices_{product}_{simulation_method}_{cob_date}.pkl')
    returns_df = load_from_pickle(f'returns_{product}_{simulation_method}_{cob_date}.pkl')
    fx_spot_df = load_from_pickle(f'fx_spot_{product}_{simulation_method}_{cob_date}.pkl')
    instrument_dict = load_from_pickle(f'instrument_dict_{product}_{simulation_method}_{cob_date}.pkl')
    logger.info("STEP 1: Market data prepared")

    # === STEP 2: Prepare Positions Data ===
    combined_pos_df = build_combined_position(cob_date, product, instrument_dict, prices_df, fx_spot_df)
    logger.info("STEP 2: Position data prepared")

    combined_pos_df = prepare_positions_data_for_var(
        product=product,
        combined_pos_df=combined_pos_df,
        price_df=prices_df,
        cob_date=cob_date,
        simulation_method=simulation_method,
        calculation_method=calculation_method,
        trader=False,
        counterparty=False)
    logger.info("STEP 2-1: Main position data prepared")

    if with_price_var:
        combined_price_pos_df = combined_pos_df[combined_pos_df['book'] == 'PRICE']
        save_to_pickle(combined_price_pos_df, 'combined_price_pos.pkl')
        combined_price_pos_df = load_from_pickle('combined_price_pos.pkl')

    save_to_pickle(combined_pos_df, 'combined_pos.pkl')
    combined_pos_df = load_from_pickle('combined_pos.pkl')
    logger.info("STEP 2-2: Price position data prepared")


    # === STEP 3: Generate PnL Vectors ===
    if (calculation_method == 'linear' or calculation_method == 'sensitivity_matrix'
            or calculation_method == 'taylor_series'):
        long_pnl_df = generate_pnl_vectors(
            combined_pos_df=combined_pos_df,
            returns_df=returns_df,
            method=calculation_method
        )
        print('Main PnL shape: ', str(combined_pos_df.shape))
        if with_price_var:
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
        filename=filename,
        write_to_excel=write_to_excel)

    # === STEP 4: Calculate VaR ===
    var_data_df = generate_var(
        product=product,
        combined_pos_df=combined_pos_df,
        long_pnl_df=long_pnl_df,
        simulation_method=simulation_method,
        calculation_method=calculation_method,
        cob_date=cob_date,
        window=window
    )
    logger.info("STEP 4-1: Main VaR calculated")
    if with_price_var:
        price_var_data_df = generate_var(
            product=product,
            combined_pos_df=combined_price_pos_df,
            long_pnl_df=long_price_pnl_df,
            simulation_method=simulation_method,
            calculation_method=calculation_method,
            cob_date=cob_date,
            window=window
        )
        logger.info("STEP 4-2: Price VaR calculated")

    logger.info("STEP 4: VaR calculated")
    # var_data_df.to_pickle('var_data_df.pkl')
    # var_data_df = pd.read_pickle('var_data_df.pkl')

    # === STEP 5: Build VaR Report ===
    var_report_df = build_var_report(var_df=var_data_df)
    if with_price_var:
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

    # === STEP 6: Export to Excel ===
    if write_to_excel:
        if os.path.exists(filename):
            mode = 'a'
            if_sheet_exists = 'replace'
        else:
            mode = 'w'
            if_sheet_exists = None
        with pd.ExcelWriter(filename, mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            var_report_df.to_excel(writer, sheet_name='var', index=True)
            if with_price_var:
                price_var_report_df.to_excel(writer, sheet_name='price_var', index=True)

    # TODO to write report formatter function, and subsequently, create email with to/cc/bcc list.
    logger.info(f"STEP 6: {simulation_method} VaR report exported")
