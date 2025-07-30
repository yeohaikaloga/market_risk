from db.db_connection import get_engine
import pandas as pd
from sqlalchemy import text
import numpy as np
from report.var_report import generate_var_report
from utils.date_utils import get_biz_days_between_list
from utils.date_utils import get_prev_biz_days_list
from utils.date_utils import get_weekdays_between_list
from contract.futures_contract import FuturesContract, custom_monthly_contract_sort_key
from contract.futures_contract import tickers_ref_dict
from contract.physical_contract import PhysicalContract
from loaded_price_series.loaded_futures_price import LoadedFuturesPrice
from loaded_price_series.loaded_physical_price import LoadedPhysicalPrice
from generated_price_series.generic_curve import GenericCurveGenerator
from datetime import datetime
from generated_price_series.cotton_basis import CottonBasisGenerator

if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    prod_engine = get_engine('prod')
    COB_DATE = '2025-07-10'
    WINDOW = 260
    biz_days = get_prev_biz_days_list(date=COB_DATE, no_of_days=WINDOW + 20)
    START_DATE = biz_days[0]
    position = [1 , 1] # To replace with positions once ready

    cotton_instruments_list = ['CT', 'VV', 'CCL']
    #rubber_instruments_list = ['OR', 'SRB', 'RT', 'JN', 'BDR', 'RG']
    #non_cotton_instruments_list = ['IJ', 'SB', 'S ', 'BO', 'QW', 'SM', 'DL', 'C ', 'W ', 'KW']
    #instruments_list = cotton_instruments_list + rubber_instruments_list + non_cotton_instruments_list


    ### NEXT CODE TO WRITE IS TO GENERATE COTTON ORIGIN BASIS

    # Step 1: Import cotton contract prices
    instrument = 'CT'
    source = prod_engine
    futures_contract = FuturesContract(instrument_id=instrument, source=prod_engine)
    selected_months = {'H', 'K', 'N', 'Z'}
    futures_contract.load_ref_data()
    print('futures load ref data')
    print(futures_contract.load_ref_data())
    contracts = futures_contract.load_contracts(relevant_months=selected_months)
    print('contracts:', contracts)
    contract_start_dates = futures_contract.load_start_dates(relevant_months=selected_months)
    print(futures_contract.start_dates)
    contract_expiry_dates = futures_contract.load_expiry_dates(relevant_months=selected_months)
    print(futures_contract.expiry_dates)
    roll_days = 14
    contract_roll_dates = {k: v - pd.Timedelta(days=roll_days) for k, v in contract_expiry_dates.items()}
    print(contract_roll_dates)

    # Step 2: Load loaded_price_series data for these contracts
    futures_price = LoadedFuturesPrice(instrument_id=futures_contract.instrument_id, source=prod_engine)
    futures_price_df = futures_price.load_prices(start_date=START_DATE,
                                                 end_date=COB_DATE,
                                                 selected_contracts=contracts,
                                                 reindex_dates=biz_days,
                                                 instrument_id=instrument)
    print(futures_price_df.head())
    print(futures_price_df.tail())

    # Step 2A: Import cotlook data
    cif_crops = ['Burkina Faso Bola/s', 'Brazilian', 'Ivory Coast Manbo/s', 'Mali Juli/s', 'Memphis/Orleans/Texas',
                 'A Index']
    crop_dict = {'Brazilian': {'grade': 'M_1-1/8"_std', 'type': None},
                 'Burkina Faso Bola/s': {'grade': 'SM_1-1/8"_h', 'type': 'Bola/s'},
                 'Ivory Coast Manbo/s': {'grade': 'SM_1-1/8"_h', 'type': 'Manbo/s'},
                 'Mali Juli/s': {'grade': 'SM_1-1/8"_h', 'type': 'Juli/s'},
                 'Memphis/Orleans/Texas': {'grade': 'M_1-1/8"_std', 'type': 'MOT'},
                 'A Index': {'grade': 'M_1-1/8"_std', 'type': 'A Index'}}

    filename = f"basis_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(filename) as writer:
        basis_df = pd.DataFrame()
        for crop_name in cif_crops:
            print(f"\nProcessing {crop_name}...")

            crop_params = crop_dict.get(crop_name, {})
            crop_params['data_source'] = 'cotlook'
            physical_contract = PhysicalContract(instrument_id=crop_name, source=prod_engine, params=crop_params)
            crop_ref_df = physical_contract.load_ref_data()
            print('load ref data:', crop_ref_df.head())

            # Step 2B: Import cotlook cif prices
            crop_price = LoadedPhysicalPrice(instrument_id=physical_contract.instrument_id, source=prod_engine,
                                             params=crop_params)
            crop_price_df = crop_price.load_prices(start_date=START_DATE, end_date=COB_DATE)
            print(crop_price_df.head())

            crop_price_df_by_year = crop_price_df.pivot(index='tdate', columns='crop_year', values='px_settle')
            print(crop_price_df_by_year.head())
            print(crop_price_df_by_year.tail())

            # Step 3: Basis logic: cotlook switch + contract switch + smoothing gic
            valid_basis_columns = []
            basis_index = pd.to_datetime(get_weekdays_between_list(START_DATE, COB_DATE))
            crop_reindexed = crop_price_df_by_year.reindex(basis_index)
            futures_reindexed = futures_price_df.reindex(basis_index)
            crop_basis_df = pd.concat([crop_reindexed, futures_reindexed], axis=1)
            crop_basis_df.index.name = 'tdate'
            print(crop_basis_df.head())
            print(crop_basis_df.tail())

            crop_year_ar_switch_dates = []

            for contract in futures_price_df.columns:
                contract_start = contract_start_dates[contract]
                contract_expiry = contract_expiry_dates[contract]

                for crop_year in crop_price_df_by_year.columns:
                    crop_prices = crop_price_df_by_year[crop_year]
                    if physical_contract.crop_year_type == 'cross':
                        ref_year, ref_next_year = map(int, crop_year.split("/"))
                    elif physical_contract.crop_year_type == 'straight':
                        ref_year = int(crop_year)
                        ref_next_year = ref_year + 1
                    crop_year_start = pd.Timestamp(f"{ref_year}-08-01")
                    crop_year_end = pd.Timestamp(f"{ref_next_year}-07-31")
                    crop_year_ar_switch = pd.Timestamp(f"{ref_year}-07-31")
                    # next crop year's abs ret starts from last biz day in Jul (based on next crop year's price in first biz day in Aug)
                    crop_year_ar_switch_dates.append(crop_year_ar_switch)

                    # This is to allow Z contract to be taken against current and next crop year
                    contract_year = 2020 + int(contract[-1])
                    is_ctz_exception = (contract[2] == 'Z' and (contract_year == ref_next_year or
                                                                contract_year == ref_year))

                    if crop_year_start <= contract_expiry <= crop_year_end or is_ctz_exception:
                        col_name = f"{crop_year.replace('20','')} vs {contract}"
                        crop_basis_df[col_name] = crop_basis_df[crop_year].shift(-1) - crop_basis_df[contract]
                        crop_basis_df[col_name] = crop_basis_df[col_name].interpolate(method='linear', limit_area='inside')
                        crop_basis_df[col_name + ' (sm)'] = CottonBasisGenerator.smooth_basis(crop_basis_df[col_name], crop_name)[0]
                        crop_basis_df[col_name + ' (sm) w'] = CottonBasisGenerator.smooth_basis(crop_basis_df[col_name], crop_name)[1]
                        valid_basis_columns.append(col_name)

            for col_name in valid_basis_columns:
                crop_basis_df[col_name + ' AR'] = crop_basis_df[col_name] - crop_basis_df[col_name].shift(1)
                crop_basis_df[col_name + ' AR (sm)'] = crop_basis_df[col_name + ' (sm)'] - crop_basis_df[col_name + ' (sm)'].shift(1)

            print(crop_basis_df.head())
            print(basis_index)

            sorted_all_switch_dates = (sorted(dates for dates in set(contract_roll_dates.values()).union(crop_year_ar_switch_dates)
                                             if pd.Timestamp(START_DATE) <= dates <= pd.Timestamp(COB_DATE))
                                       + [pd.Timestamp(COB_DATE)])
            print(sorted_all_switch_dates)

            abs_ret_conditions = ([basis_index < sorted_all_switch_dates[0]] +
                                  [((basis_index >= sorted_all_switch_dates[i]) &
                                    (basis_index < sorted_all_switch_dates[i + 1]))
                                   for i in range(len(sorted_all_switch_dates) - 1)])
            abs_ret_cols = [col for col in crop_basis_df.columns if col.endswith(' AR')]
            smooth_abs_ret_cols = [col for col in crop_basis_df.columns if col.endswith(' AR (sm)')]
            abs_ret_choices = [crop_basis_df[col] for col in abs_ret_cols][:len(abs_ret_conditions)]
            smooth_abs_ret_choices = [crop_basis_df[col] for col in smooth_abs_ret_cols][:len(abs_ret_conditions)]
            crop_basis_df['final AR series'] = np.select(abs_ret_conditions, abs_ret_choices, default=0)
            crop_basis_df['final AR (sm) series'] = np.select(abs_ret_conditions, smooth_abs_ret_choices, default=0)
            crop_basis_df.to_excel(writer, sheet_name=crop_name.replace("/", "-")[:31])

            summary_cols = crop_basis_df[['final AR (sm) series']].copy()
            summary_cols.columns = [f"{crop_name} {col}" for col in summary_cols.columns]
            basis_df = pd.concat([basis_df, summary_cols], axis=1)

        basis_df.to_excel(writer, sheet_name="Summary AR Series")

        return basis_df



