import pandas as pd


def generate_generic_curve(df: pd.DataFrame, position: int = 1, roll_days: int = 0) -> pd.Series:
    """
    Generate the N-th generic futures curve with optional early rolling.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index and futures contract columns (e.g., CTH4, CTK4).
        position (int): Generic position to compute (1 = front, 2 = second, etc.).
        roll_days (int): Number of business days before a contract's last availability to roll.

    Returns:
        pd.Series: Time series of the N-th generic futures curve.
    """
    df = df.sort_index()
    contracts = df.columns.tolist()
    index = df.index
    generic_curve = pd.Series(index=index, dtype='float64')

    # Build contract active date ranges
    contract_roll_dates = {}
    for col in contracts:
        valid_dates = df[col].dropna().index
        if not valid_dates.empty:
            last_date = valid_dates[-1]
            roll_date_idx = valid_dates.get_loc(last_date) - roll_days
            roll_date = valid_dates[max(0, roll_date_idx)]
            contract_roll_dates[col] = roll_date

    # Rolling logic
    for date in index:
        # Get all active contracts for this date
        row = df.loc[date]
        valid_contracts = row[row.notna()]

        # Filter contracts that are already rolled out due to roll_days logic
        eligible_contracts = []
        for contract in valid_contracts.index:
            roll_date = contract_roll_dates.get(contract, date)
            if date <= roll_date:
                eligible_contracts.append(contract)

        # If enough contracts are eligible, select the Nth
        if len(eligible_contracts) >= position:
            contract_to_use = eligible_contracts[position - 1]
            generic_curve.at[date] = df.at[date, contract_to_use]
        else:
            generic_curve.at[date] = pd.NA

    return generic_curve
