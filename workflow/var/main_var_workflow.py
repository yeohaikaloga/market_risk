from workflow.shared.data_preparation_workflow import (prepare_returns_and_positions_data, prepare_pos_data_for_var,
                                                       load_raw_cotton_deriv_position, load_raw_rubber_deriv_position,
                                                       load_raw_rms_deriv_position,
                                                       generate_product_code_list_for_generic_curve)
from workflow.shared.pnl_calculator_workflow import generate_pnl_vectors, analyze_and_export_unit_pnl
from workflow.var.var_generator_workflow import (generate_var, build_var_report, build_cotton_var_report_exceptions,
                                                 build_cotton_price_var_report_exceptions,
                                                 build_rubber_var_report_exceptions)
from utils.contract_utils import instrument_ref_dict
import pickle
import pandas as pd
import os
from datetime import datetime

def main_var_workflow(cob_date: str, product: str, method: str, window: int, with_price_var: bool, write_to_excel: bool):


    print(f"[INFO] Running VaR workflow for product: {product}, COB: {cob_date}, Method: {method}, Window: {window}")

    filename = f"{cob_date}_{product[:3]}_{method}_var_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    if product == 'cotton':
        deriv_pos_df = load_raw_cotton_deriv_position(cob_date)
    elif product == 'rubber':
        deriv_pos_df = load_raw_rubber_deriv_position(cob_date)
    elif product == 'rms':
        deriv_pos_df = load_raw_rms_deriv_position(cob_date)
    print('No. of derivatives positions: ', str(len(deriv_pos_df)))
    if len(deriv_pos_df) != 0:
        product_code_list = generate_product_code_list_for_generic_curve(product, deriv_pos_df)
    else:
        if product == 'cotton':
            product_code_list = ['CM CT', 'CM VV', 'CM AVY', 'CM CCL', 'CM C', 'CM W', 'CM S']
        elif product == 'rubber':
            product_code_list = ['CM OR', 'CM JN', 'CM SRB', 'IM RT', 'CM BDR', 'CM RG']
        elif product == 'rms':
            product_code_list = list(instrument_ref_dict.keys())

    # #=== STEP 1 and 2: Prepare Data ===
    instrument_dict, combined_pos_df = prepare_returns_and_positions_data(
        product=product,
        product_code_list=product_code_list,
        cob_date=cob_date,
        window=window
    )
    combined_pos_df = prepare_pos_data_for_var(
        combined_pos_df=combined_pos_df,
        method=method,
        trader=False,
        counterparty=False)
    print('Main positions done')

    if with_price_var:
        combined_price_pos_df = combined_pos_df[combined_pos_df['book'] == 'PRICE']
        print('Price positions done')
    print("[MAIN VAR WORKFLOW INFO] Step 1 and 2: Data preparation completed.")

    # === STEP 3: Generate PnL Vectors ===
    if method == 'linear' or method == 'non-linear_monte_carlo' or method == 'taylor_series':
        long_pnl_df = generate_pnl_vectors(
            combined_pos_df=combined_pos_df,
            instrument_dict=instrument_dict,
            method=method
        )
        print('Main PnL shape: ', str(combined_pos_df.shape))
        if with_price_var:
            long_price_pnl_df = generate_pnl_vectors(
                combined_pos_df=combined_price_pos_df,
                instrument_dict=instrument_dict,
                method=method
            )
            print('Price PnL shape: ', str(combined_price_pos_df.shape))
        print("[MAIN VAR WORKFLOW INFO] Step 3: PnL vectors generated.")
    else:
        raise NotImplementedError(f"Method '{method}' not supported yet.")
    # #
    combined_pos_df.to_pickle('combined_pos_df.pkl')
    long_pnl_df.to_pickle('long_pnl_df.pkl')
    f = open('instrument_dict.pkl', 'wb')
    pickle.dump(instrument_dict, f)
    f.close()
    f = open('instrument_dict.pkl', 'rb')
    instrument_dict = pickle.load(f)
    f.close()
    combined_pos_df = pd.read_pickle('combined_pos_df.pkl')
    long_pnl_df = pd.read_pickle('long_pnl_df.pkl')
    #

    analyze_and_export_unit_pnl(
        product=product,
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
        cob_date=cob_date,
        window=window
    )
    print('Main VaR done')
    if with_price_var:
        price_var_data_df = generate_var(
            product=product,
            combined_pos_df=combined_price_pos_df,
            long_pnl_df=long_price_pnl_df,
            cob_date=cob_date,
            window=window
        )
        print('Price VaR done')

    print("[MAIN VAR WORKFLOW INFO] Step 4: VaR calculation completed.")
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
                                                           report_df=var_report_df,
                                                           var_df=var_data_df,
                                                           cob_date=cob_date,
                                                           window=window)
        price_var_report_df = build_cotton_price_var_report_exceptions(report_df=price_var_report_df)
        print("[MAIN VAR WORKFLOW INFO] Step 5: Cotton VaR Report built.")
    elif product == 'rubber':
        var_report_df = build_rubber_var_report_exceptions(report_df=var_report_df)
        print("[MAIN VAR WORKFLOW INFO] Step 5: Rubber VaR Report built.")
        pass

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

    #TODO to write report formatter function, and subsequently, create email with to/cc/bcc list.
    print("[MAIN VAR WORKFLOW INFO] Step 6: Report exported to Excel.")


