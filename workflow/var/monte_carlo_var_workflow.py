from workflow.shared.data_preparation_workflow import (prepare_returns_and_positions_data, prepare_pos_data_for_var,
                                                       load_raw_cotton_deriv_position, load_raw_rubber_deriv_position,
                                                       load_raw_rms_deriv_position,
                                                       generate_product_code_list_for_generic_curve)
from workflow.shared.pnl_calculator_workflow import generate_pnl_vectors, analyze_and_export_unit_pnl
from workflow.var.var_generator_workflow import (generate_var, build_var_report, build_cotton_var_report_exceptions,
                                                 build_cotton_price_var_report_exceptions,
                                                 build_rubber_var_report_exceptions)
from utils.contract_utils import load_instrument_ref_dict
import pickle
import pandas as pd
import os
from datetime import datetime

def monte_carlo_var_workflow(cob_date: str, product: str, calculation_method: str, window: int, with_price_var: bool, write_to_excel: bool):

    print(f"[INFO] Running VaR workflow for product: {product}, COB: {cob_date}, Method: {calculation_method}, Window: {window}")

    filename = f"{cob_date}_{product[:3]}_{calculation_method}_var_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    instrument_ref_dict = load_instrument_ref_dict('uat')
    product_code_list = list(instrument_ref_dict.keys())
