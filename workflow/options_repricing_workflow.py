from db.db_connection import get_engine
from utils.date_utils import get_prev_biz_days_list
from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from price_series_loader.derivatives_price_loader import DerivativesPriceLoader


def options_workflow():
    prod_engine = get_engine('prod')
    cob_date = '2025-08-02'
    days_list = get_prev_biz_days_list(cob_date, 261)

    instruments_list = ['CT']
    for instrument in instruments_list:
        derivatives_contract = DerivativesContractRefLoader(instrument_name=instrument, source=prod_engine)
        futures_contracts = derivatives_contract.load_contracts(mode='futures', relevant_months=('H', 'K'),
                                                                relevant_years=('4', '5'), relevant_options=None,
                                                                relevant_strikes=None)
        print(derivatives_contract.load_underlying_futures_start_dates(mode='futures', contracts=futures_contracts))
        print(derivatives_contract.load_underlying_futures_expiry_dates(mode='futures', contracts=futures_contracts))
        # print(futures_contracts)


        futures_price = DerivativesPriceLoader(instrument_name=instrument, mode='futures', source=prod_engine)
        futures_price_df = futures_price.load_prices(start_date=days_list[0],
                                                     end_date=cob_date,
                                                     contracts=futures_contracts,
                                                     reindex_dates=days_list,
                                                     instrument_name=instrument)
        print(futures_price_df.head())
        # print(vars(futures_price))
        # print('end')

        options_contracts = derivatives_contract.load_contracts(mode='options', relevant_months=('H', 'K'),
                                                                relevant_years=('4', '5'), relevant_options=('C'),
                                                                relevant_strikes=[100, 101])
        print(derivatives_contract.load_underlying_futures(contracts=options_contracts))
        print(derivatives_contract.load_options_start_dates(contracts=options_contracts))
        print(derivatives_contract.load_options_expiry_dates(contracts=options_contracts))
        # print(options_contracts)
        # print(vars(derivatives_contract))

        options_price = DerivativesPriceLoader(instrument_name=instrument, mode='options', source=prod_engine)
        options_price_df = options_price.load_prices(start_date=days_list[0],
                                                     end_date=cob_date,
                                                     contracts=options_contracts,
                                                     reindex_dates=days_list,
                                                     instrument_name=instrument)
        print(options_price_df.head())
        # print(vars(options_price))
        print('end')
