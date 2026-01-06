from price_series_loader.forex_price_loader import ForexPriceLoader
from contract_ref_loader.forex_ref_loader import ForexRefLoader
from db.db_connection import get_engine
import pandas as pd
from utils.date_utils import get_prev_biz_days_list
from utils.forex_utils import invert_selected_fx

def load_forex(cob_date, window, type):
    uat_engine = get_engine('uat')
    days_list = get_prev_biz_days_list(cob_date, window + 1)
    fx = ForexRefLoader(source=uat_engine)
    usd_ccy = ['USD']
    quote_ccy = ['BRL', 'MYR', 'MXN', 'COP', 'CNH', 'CNY', 'THB', 'IDR', 'JPY', 'INR', 'CAD']
    base_ccy = ['EUR', 'GBP', 'AUD', 'NZD']
    fwd_mon = 'NULL'

    #USD as base
    fx.load_metadata(
        type=type,
        base_ccy=usd_ccy,
        quote_ccy=quote_ccy,
        fwd_mon=fwd_mon,
    )

    fx_usd_base_price_loader = ForexPriceLoader(source=uat_engine, ref_loader=fx)
    fx_usd_base_df = fx_usd_base_price_loader.load_prices(
        start_date=days_list[0],
        end_date=cob_date,
        type=type
    )

    #USD as quote
    fx.load_metadata(
        type=type,
        base_ccy=base_ccy,
        quote_ccy=usd_ccy,
        fwd_mon=fwd_mon,
    )

    fx_usd_quote_price_loader = ForexPriceLoader(source=uat_engine, ref_loader=fx)
    fx_usd_quote_df = fx_usd_quote_price_loader.load_prices(
        start_date=days_list[0],
        end_date=cob_date
    )
    fx_usd_quote_df = invert_selected_fx(fx_usd_quote_df,
                                         cols=['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD'],
                                         rename=True)

    fx_spot_df = pd.concat([fx_usd_base_df, fx_usd_quote_df], axis=1)

    usdusd_series = pd.Series(1.0, index=fx_spot_df.index, name='USDUSD')
    fx_spot_df['USDUSD'] = usdusd_series

    fx_spot_df = fx_spot_df.sort_index(axis=1)
    print(fx_spot_df.tail())
    return fx_spot_df