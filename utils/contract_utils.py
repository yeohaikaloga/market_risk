from sqlalchemy import text
from db.db_connection import get_engine
import pandas as pd
import numpy as np


month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}

product_code_to_bbg_futures_category_mapping = {'CM CT': 'Fibers', 'CM VV': 'Fibers', 'CM CCL': 'Fibers',
                                                'CM AVY': 'Fibers', 'CM OR': 'Industrial Material',
                                                'CM JN': 'Industrial Material', 'CM SRB': 'Industrial Material',
                                                'IM RT': 'Industrial Material', 'CM BDR': 'Industrial Material',
                                                'CM RG': 'Industrial Material', 'CM C': 'Corn', 'CM CRD': 'Corn',
                                                'CM AC': 'Corn', 'CM W': 'Wheat', 'CM KW': 'Wheat', 'CM MW': 'Wheat',
                                                'CM KFP': 'Wheat', 'CM S': 'Soy', 'CM SM': 'Soy', 'CM BO': 'Soy',
                                                'CM AE': 'Soy', 'CM SRS': 'Soy', 'CM AK': 'Soy', 'CM BP': 'Soy',
                                                'CM SH': 'Soy', 'CM DL': 'Refined Products',
                                                'CM QS': 'Refined Products', 'CM THE': 'Refined Products',
                                                'CM HO': 'Refined Products', 'CM SB': 'Foodstuff', 'CM QW': 'Foodstuff',
                                                'CM DF': 'Foodstuff', 'CM CC': 'Foodstuff', 'CM KC': 'Foodstuff',
                                                'CM QC': 'Foodstuff', 'CM AX': 'Foodstuff', 'CM KO': 'Foodstuff',
                                                'CM PAL': 'Foodstuff', 'CM VPC': 'Foodstuff', 'CM MDS': 'Foodstuff',
                                                'CM DA': 'Foodstuff', 'CM IJ': 'Other Grain', 'CM RS': 'Other Grain',
                                                'CM ZRR': 'Other Grain', 'CM LHD': 'Livestock'}

_instrument_ref_cache = {}


def load_instrument_ref_dict(source='uat'):
    global _instrument_ref_cache

    # If already loaded, reuse
    if source in _instrument_ref_cache:
        return _instrument_ref_cache[source]

    if source == 'uat':
        uat_engine = get_engine('uat')
        query = """
            SELECT *
            FROM staging.product_master_data
        """
        with uat_engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        result = {}
        for _, row in df.iterrows():
            code = row['invenio_product_code']
            bbg_futures_category = product_code_to_bbg_futures_category_mapping.get(code)
            result[code] = {
                'bbg_product_code': row['bbg_product_code'],
                'product_name': row['product_name'],
                'exchange_name': row['exchange_name'],
                'currency': row['currency'],
                'lots_to_MT_conversion': row['lot_mult'],
                'price_conv_factor': row['price_conv_factor'],
                'to_USD_conversion': row['constant_curr_mult'],
                'curr_mult_factor': row['curr_mult_factor'],
                'curr_mult_type': row['curr_mult_type'],
                'opera_futures_category': row['asset_class'].capitalize(),
                'bbg_futures_category': bbg_futures_category
            }

        _instrument_ref_cache[source] = result
        return result

    else:
        raise ValueError("Invalid source requested for instrument reference dictionary")

def extract_instrument_from_product_code(product_code: str, ref_dict: dict) -> str:
    if product_code in ref_dict.keys():
        tokens = str(product_code).split()
        if len(tokens[1]) > 1:
            return tokens[1]  # second token
        else:
            # single token, pad with space
            return tokens[1] + ' '
    else:
        return product_code


instrument_ref_dict = load_instrument_ref_dict('uat')
instrument_list = [
    extract_instrument_from_product_code(pc, instrument_ref_dict)
    for pc in instrument_ref_dict.keys()
]
product_specifications = {'cotton': {'instrument_list': ['CT', 'VV', 'AVY', 'C', 'W', 'S', 'BO',
                                                         'SM', 'IJ', 'SB', 'QW'],
                                     'hist_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'cob_date_fx'}, # same as grains 'after_ret'
                                     'mc_sim': {'usd_conversion_mode': 'pre', 'forex_mode': 'cob_date_fx'}}, # for physicals only; generic curves is always  pre, daily_fx
                          'rubber': {'instrument_list': ['OR', 'JN', 'SRB', 'RT', 'BDR', 'RG'],
                                     'usd_conversion_mode': 'pre', 'forex_mode': 'cob_date_fx',
                                     'hist_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'cob_date_fx'}, # same as grains 'before_ret'
                                     'mc_sim': {'usd_conversion_mode': 'pre', 'forex_mode': 'cob_date_fx'}}, # for physicals only; generic curves is always  pre, daily_fx
                          'rms': {'instrument_list': ['KC', 'DF', 'CC', 'QC', 'CT', 'SB', 'QW', 'S', 'SM', 'BO', 'C',
                                                      'W', 'KW', 'MW', 'CA', 'EP', 'IJ', 'RS', 'CRD', 'AX', 'KO', 'OR',
                                                      'AE', 'VV', 'AC', 'LHD', 'SRS', 'PAL', 'RT', 'ZRR', 'AK', 'BP',
                                                      'SH', 'VPC', 'QS', 'THE', 'HO', 'MDS', 'DA', 'BDR', 'KFP', 'LC',
                                                      'FC', 'LS', 'CO', 'CL', 'EN', 'XB', 'NG'],
                                  'hist_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'daily_fx'},
                                  'mc_sim': {'usd_conversion_mode': 'pre', 'forex_mode': 'cob_date_fx'}},
                          'biocane': {'instrument_list': None,
                                      'hist_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'cob_date_fx'},
                                      'mc_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'cob_date_fx'}},
                          'wood': {'instrument_list': None,
                                   'hist_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'cob_date_fx'},
                                   'mc_sim': {'usd_conversion_mode': 'post', 'forex_mode': 'cob_date_fx'}}}

done = ['KC', 'DF', 'CC', 'QC', 'CT', 'SB', 'QW', 'S', 'SM', 'BO', 'C',
                                                      'W', 'KW', 'MW', 'IJ', 'RS', 'CRD', 'AX', 'KO', 'OR',
                                                      'AE', 'VV', 'PAL', 'RT', 'ZRR', 'AK', 'BP',
                                                      'SH', 'VPC', 'DA', 'BDR', 'KFP', ]
problem = ['CA', 'EP', 'AC', 'LHD', 'SRS', 'QS', 'THE', 'HO', 'MDS', 'LC', 'FC', 'LS', 'CO', 'CL', 'EN', 'XB', 'NG']

def custom_monthly_contract_sort_key(contract):
    contract_part = contract.replace(' COMB', '').replace(' Comdty', '')
    if len(contract_part) < 3:
        return contract_part, 0, 0
    # Extract month and year chars
    month_char = contract_part[-2]
    year_digit = contract_part[-1]
    # Instrument name is everything before last two characters (including spaces)
    instrument_name = contract_part[:-2]

    try:
        year = 2000 + int(year_digit)
    except ValueError:
        year = 0

    month = month_codes.get(month_char, 0)

    return instrument_name, year, month


def get_month_code(month: int) -> str | None:
    """
    Given a month number, return the corresponding month code key.
    If no match found, return None.
    """
    for code, num in month_codes.items():
        if num == month:
            return code
    return None


def obtain_product_code_from_instrument_name(instrument_name: str, ref_dict: dict):
    instrument_name = instrument_name.strip().upper()

    for key, val in ref_dict.items():
        if val.get('bbg_product_code', '').upper() == instrument_name:
            return key

    return None


def get_USD_MT_conversion_from_product_code(product_code: str, ref_dict: dict):
    if product_code == 'EX GIN S6':
        return ref_dict.get('CM CCL', {}).get('to_USD_conversion', np.nan)
    return ref_dict.get(product_code, {}).get('to_USD_conversion', np.nan)
