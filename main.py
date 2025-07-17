from db.db_connection import get_engine
import pandas as pd
import numpy as np
from utils.date_utils import no_of_days_list
from report.var_report import generate_var_report


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('future.no_silent_downcasting', True)

    prod_engine = get_engine('prod')
    COB_DATE = '2025-07-10'
    days_260_list = no_of_days_list(COB_DATE, 260)
    position = [1 , 1] # To replace with positions once ready

    cotton_instruments_list = ['CT', 'VV', 'CCL']
    #rubber_instruments_list = ['OR', 'SRB', 'RT', 'JN', 'BDR', 'RG']
    #non_cotton_instruments_list = ['IJ', 'SB', 'S ', 'BO', 'QW', 'SM', 'DL', 'C ', 'W ', 'KW']
    #instruments_list = cotton_instruments_list + rubber_instruments_list + non_cotton_instruments_list
    instruments_list = ['CT'] #cotton_instruments_list

    #result = generate_var_report(instruments_list, COB_DATE, days_260_list, position, prod_engine)
    #print(result['price_series'])
    #print(result['returns'])
    #print(result['PnL'])
    #print(result['VaR_95'])
    #print(result['VaR_99'])

    ### NEXT CODE TO WRITE IS TO GENERATE COTTON ORIGIN BASIS

    # Step 1: Import cotton contract prices

    # Step 2: Import cotlook cif prices

    # Step 3: Basis logic: cotlook switch + contract switch + smoothing logic

    # Step 4: Output: pd.Series

