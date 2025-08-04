import pandas as pd
from db.db_connection import get_engine
from utils.date_utils import get_prev_biz_days_list
from datetime import datetime
from contract.futures_contract import FuturesContract
from loaded_price_series.loaded_futures_price import LoadedFuturesPrice
from loaded_price_series.loaded_physical_price import LoadedPhysicalPrice
from generated_price_series.cotton_basis import CottonBasisGenerator
from generated_price_series.cotton_basis import crop_dict
from contract.physical_contract import PhysicalContract


def fy24_cotton_basis_workflow(write_to_excel: bool = True, apply_smoothing: bool = True):
    prod_engine = get_engine('prod')
    cob_date = '2025-07-10'
    window = 260
    trailing_days_before_start = 20
    no_of_days = window + trailing_days_before_start
    biz_days = get_prev_biz_days_list(date=cob_date, no_of_days=no_of_days)
    start_date = biz_days[0]

    # Step 1: Import cotton contract prices
    instrument = 'CT'
    futures_contract = FuturesContract(instrument_id=instrument, source=prod_engine)
    selected_months = {'H', 'K', 'N', 'Z'}
    futures_contract.load_ref_data()
    contracts = futures_contract.load_contracts(relevant_months=selected_months)
    contract_expiry_dates = futures_contract.load_expiry_dates(relevant_months=selected_months)
    roll_days = 14
    contract_roll_dates = {k: v - pd.Timedelta(days=roll_days) for k, v in contract_expiry_dates.items()}

    # Step 2: Load loaded_price_series data for these contracts
    futures_price = LoadedFuturesPrice(instrument_id=futures_contract.instrument_id, source=prod_engine)
    futures_price_df = futures_price.load_prices(start_date=start_date,
                                                 end_date=cob_date,
                                                 selected_contracts=contracts,
                                                 reindex_dates=biz_days,
                                                 instrument_id=instrument)

    # Step 2A: Import cotlook data
    cif_crops = ['Burkina Faso Bola/s', 'Brazilian', 'Ivory Coast Manbo/s', 'Mali Juli/s', 'Memphis/Orleans/Texas',
                 'A Index']

    cotton_basis_generator = CottonBasisGenerator(futures_price_df=futures_price_df,
                                                  contract_expiry_dates=contract_expiry_dates,
                                                  contract_roll_dates=contract_roll_dates,
                                                  cob_date=cob_date, window=window,
                                                  trailing_days_before_start=trailing_days_before_start,
                                                  source=prod_engine)

    # Pre-load all physical contracts and price data
    physical_contracts_and_prices = []
    for crop_name in cif_crops:
        crop_params = crop_dict.get(crop_name, {})
        crop_params['data_source'] = 'cotlook'
        physical_contract = PhysicalContract(instrument_id=crop_name, source=prod_engine, params=crop_params)
        physical_contract.load_ref_data()
        crop_price = LoadedPhysicalPrice(instrument_id=physical_contract.instrument_id, source=prod_engine,
                                         params=crop_params)
        crop_price_df = crop_price.load_prices(start_date=start_date, end_date=cob_date)
        crop_price_df_by_year = crop_price_df.pivot(index='tdate', columns='crop_year', values='px_settle')

        physical_contracts_and_prices.append((physical_contract, crop_price_df_by_year))

    # Generate summary basis
    basis_df = cotton_basis_generator.generate_all_crop_basis_return_series(physical_contracts_and_prices,
                                                                            apply_smoothing=apply_smoothing)

    if write_to_excel:
        filename = f"basis_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(filename) as writer:
            for physical_contract, crop_price_df_by_year in physical_contracts_and_prices:
                crop_name = physical_contract.instrument_id

                # Step 3: Generate crop basis
                crop_basis_df = cotton_basis_generator.generate_crop_basis(
                    physical_contract=physical_contract,
                    crop_price_df_by_year=crop_price_df_by_year
                )
                crop_basis_df.to_excel(writer, sheet_name=crop_name.replace("/", "-")[:31])

            # Write the summary DataFrame last
            basis_df.to_excel(writer, sheet_name="Summary AR Series")
