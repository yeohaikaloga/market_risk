import pandas as pd


def invert_fx(fx):
    """
    Invert FX series or DataFrame.

    If fx is a Series:
        returns 1 / fx

    If fx is a DataFrame:
        returns 1 / every column
    """
    if isinstance(fx, pd.Series):
        return 1.0 / fx

    elif isinstance(fx, pd.DataFrame):
        return 1.0 / fx

    else:
        raise TypeError("invert_fx expects a pandas Series or DataFrame")

def invert_selected_fx(fx_df: pd.DataFrame, cols: list, rename=True) -> pd.DataFrame:
    if isinstance(cols, str):
        cols = [cols]

    missing = [c for c in cols if c not in fx_df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    result = fx_df.copy()

    for col in cols:
        result[col] = 1.0 / result[col]

        if rename:
            if len(col) == 6:   # standard XXXYYY format
                base = col[:3]
                quote = col[3:]
                new_col = quote + base
                result = result.rename(columns={col: new_col})
            else:
                raise ValueError(f"Cannot auto-rename non-standard ticker: {col}")

    return result

def cross_fx(base_leg, quote_leg):
    """
    Construct a cross FX pair from two FX legs.

    Examples:
        EURJPY = EURUSD * USDJPY
        EURBRL = EURUSD * USDBRL

    Inputs:
        base_leg  : pd.Series or pd.DataFrame
        quote_leg : pd.Series or pd.DataFrame

    Returns:
        pd.Series or pd.DataFrame depending on input shape.
    """

    if not isinstance(base_leg, (pd.Series, pd.DataFrame)):
        raise TypeError("base_leg must be a Series or DataFrame")

    if not isinstance(quote_leg, (pd.Series, pd.DataFrame)):
        raise TypeError("quote_leg must be a Series or DataFrame")

    # Align dates automatically, forward-fill missing
    base_leg, quote_leg = base_leg.align(quote_leg, join="outer")
    base_leg = base_leg.ffill()
    quote_leg = quote_leg.ffill()

    # Simple multiplication for cross rate
    return base_leg * quote_leg
