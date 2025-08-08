
month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}


def custom_monthly_contract_sort_key(contract, instrument_name):
    instrument_len = len(instrument_name)
    first_leg_len = instrument_len + 2
    first_leg = contract[:first_leg_len]

    if len(first_leg) < first_leg_len:
        return (0, 0)

    month_char = first_leg[instrument_len]
    year_digit = first_leg[instrument_len + 1]

    try:
        year = 2000 + int(year_digit)
    except ValueError:
        year = 0

    month = month_codes.get(month_char, 0)
    return (year, month)
