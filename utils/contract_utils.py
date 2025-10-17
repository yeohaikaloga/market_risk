
month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}

#TODO to replace with product_master_table
instrument_ref_dict = {'CT': {'futures_category': 'Fibers', 'to_USD_conversion': 2204.6/100, 'currency': 'USc',
                              'units': 'lbs', 'lots_to_MT_conversion': 22.679851220176, 'product_name': 'COTTON ICE'},
                       'VV': {'futures_category': 'Fibers', 'to_USD_conversion': 1, 'currency': 'CNY', 'units': 'MT',
                              'lots_to_MT_conversion': 5, 'product_name': 'COTTON ZCE'},
                       'CCL': {'futures_category': 'Fibers', 'to_USD_conversion': 1000/355.56, 'currency': 'INR',
                               'units': 'candy', 'lots_to_MT_conversion': 22.679851220176, 'product_name': 'MCX COTTON'},
                       'OR': {'futures_category': 'Industrial Material', 'to_USD_conversion': 1000/100, 'currency': 'USc',
                              'lots_to_MT_conversion': 5},
                       'JN': {'futures_category': 'Industrial Material', 'to_USD_conversion': 1000, 'currency': 'JPY',
                              'lots_to_MT_conversion': 5},
                       'SRB': {'futures_category': 'Industrial Material', 'to_USD_conversion': 1, 'currency': 'CNY',
                               'lots_to_MT_conversion': 10},
                       'RT': {'futures_category': 'Industrial Material', 'to_USD_conversion': 1,
                              'lots_to_MT_conversion': 10},
                       'BDR': {'futures_category': 'Industrial Material', 'to_USD_conversion': 1, 'currency': 'CNY',
                               'lots_to_MT_conversion': 5},
                       'RG': {'futures_category': 'Industrial Material', 'to_USD_conversion': 1000/100,
                              'currency': 'USc', 'lots_to_MT_conversion': 5},
                       'C ': {'futures_category': 'Corn', 'to_USD_conversion': 39.36821/100, 'currency': 'USc',
                              'lots_to_MT_conversion': 127.01, 'product_name': 'CORN CBOT'},
                       'EP': {'futures_category': 'Corn', },
                       'CRD': {'futures_category': 'Corn'},
                       'AC': {'futures_category': 'Corn'},
                       'CA': {'futures_category': 'Wheat'},
                       'W ': {'futures_category': 'Wheat', 'to_USD_conversion': 36.74371/100, 'currency': 'USc',
                              'lots_to_MT_conversion': 136.08, 'product_name': 'WHEAT'},
                       'KW': {'futures_category': 'Wheat', },
                       'MW': {'futures_category': 'Wheat'},
                       'KFP': {'futures_category': 'Wheat'},
                       'S ': {'futures_category': 'Soy', 'to_USD_conversion': 36.74371/100, 'currency': 'USc',
                              'lots_to_MT_conversion': 136.08, 'product_name': 'SOYBEAN'},
                       'SM': {'futures_category': 'Soy'},
                       'BO': {'futures_category': 'Soy'},
                       'AE': {'futures_category': 'Soy'},
                       'SRS': {'futures_category': 'Soy'},
                       'AK': {'futures_category': 'Soy'},
                       'BP': {'futures_category': 'Soy'},
                       'SH': {'futures_category': 'Soy'},
                       'DL': {'futures_category': 'Refined Products'},
                       'QS': {'futures_category': 'Refined Products'},
                       'THE': {'futures_category': 'Refined Products'},
                       'HO': {'futures_category': 'Refined Products'},
                        'SB': {'futures_category': 'Foodstuff', 'to_USD_conversion': 2204.6/100, 'currency': 'USc',
                               'units': 'lbs', 'lots_to_MT_conversion': 22.679851220176, 'product_name': 'NY SUGAR'},
                       'QW': {'futures_category': 'Foodstuff'},
                       'DF': {'futures_category': 'Foodstuff'},
                       'CC': {'futures_category': 'Foodstuff'},
                       'KC': {'futures_category': 'Foodstuff'},
                       'QC': {'futures_category': 'Foodstuff'},
                       'AX': {'futures_category': 'Foodstuff'},
                       'KO': {'futures_category': 'Foodstuff'},
                       'PAL': {'futures_category': 'Foodstuff'},
                       'VPC': {'futures_category': 'Foodstuff'},
                       'MDS': {'futures_category': 'Foodstuff'},
                       'DA': {'futures_category': 'Foodstuff'},
                       'IJ': {'futures_category': 'Other Grain'},
                       'RS': {'futures_category': 'Other Grain'},
                       'ZRR': {'futures_category': 'Other Grain'},
                       'LHD': {'futures_category': 'Livestock'}}

def custom_monthly_contract_sort_key(contract):
    contract_part = contract.replace(' COMB', '').replace(' Comdty', '')
    if len(contract_part) < 3:
        return (contract_part, 0, 0)
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

    return (instrument_name, year, month)

def get_month_code(month: int) -> str | None:
    """
    Given a month number, return the corresponding month code key.
    If no match found, return None.
    """
    for code, num in month_codes.items():
        if num == month:
            return code
    return None

def extract_instrument_name(product_code):
    name = product_code.replace('CM ', '')
    if len(name) == 1:
        name = name + ' '
    return name


