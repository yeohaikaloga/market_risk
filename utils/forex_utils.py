import pandas as pd

def convert_series_to_usd(local_series: pd.Series, fx_series: pd.Series) -> pd.Series:
    """
    Convert a daily time series using daily FX rates.
    local_series: Series indexed by date
    fx_series: FX Series indexed by date (e.g. USDINR)
    """
    fx_series = fx_series.reindex(local_series.index).ffill()
    return local_series * fx_series

def convert_value_to_usd(value: float, fx_series: pd.Series, date: str | pd.Timestamp) -> float:
    """
    Convert a single aggregated value into USD using FX on a specific date.
    """
    date = pd.to_datetime(date)
    fx_value = fx_series.loc[:date].iloc[-1]  # last available FX up to that date
    return value * fx_value

def convert_series_cross(local_series: pd.Series, fx_from: pd.Series, fx_to: pd.Series) -> pd.Series:
    """
    Convert using a cross rate (fx_from / fx_to).
    For example: convert EUR PnL to USD using EURUSD.
    """
    cross = (fx_from / fx_to).reindex(local_series.index).ffill()
    return local_series * cross