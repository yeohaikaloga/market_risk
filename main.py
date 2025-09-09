from workflow.var_generator_workflow import generate_var_workflow
from workflow.var_generator_workflow_v2 import generate_product_var_workflow
from workflow.options_repricer_workflow import options_workflow
from workflow.var_report_builder_workflow import build_var_report
import pandas as pd
from datetime import datetime


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    cob_date = '2025-08-22'
    # Note: Greeks in DB only start from 20 Aug 2025
    product = 'cotton'
    method = 'linear'
    window = 260
    # options_workflow()
    #generate_cotton_var_dict = generate_var_workflow(product=product, method=method, cob_date=cob_date, window=window)
    #cotton_pos_df = generate_cotton_var_dict['delta']
    #cotton_price_pos_df = cotton_pos_df[cotton_pos_df['books'] == 'PRICE']
    #all_books_cotton_var = generate_cotton_var_dict['main']['var']
    #price_book_cotton_var = generate_cotton_var_dict['price']['var']
    #cotton_var_report = build_var_report(product=product, books='all', cob_date=cob_date, pos_df=cotton_pos_df,
    #                                     var_df=all_books_cotton_var)
    #cotton_price_var_report = build_var_report(product=product, books='price', cob_date=cob_date,
    #                                           pos_df=cotton_price_pos_df, var_df=price_book_cotton_var)
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #filename = f"{cob_date}_var_output_{timestamp}.xlsx"
    #with pd.ExcelWriter(filename) as writer:
    #    cotton_var_report.to_excel(writer, sheet_name='var')
    #    cotton_price_var_report.to_excel(writer, sheet_name='price')

    # NOTE rms VaR will continue to use Taylor series expansion
    #product = 'rms'
    #cob_date = '2025-08-29'
    #generate_rms_var_dict = generate_var_workflow(product=product, method=method, cob_date=cob_date, window=window)
    # TODO: Validate cotton linear var reports

    generate_product_var_workflow(product, method, cob_date, window)