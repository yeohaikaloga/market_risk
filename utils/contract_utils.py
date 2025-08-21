
month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}


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
