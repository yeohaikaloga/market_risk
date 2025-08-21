import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
import pandas as pd


def time_to_expiry_in_years(current_date, expiry_date):
    """Calculate time to expiry in years using Actual/365 day count convention."""
    delta = (expiry_date - current_date).days
    return max(delta, 0) / 365.0


def bsm_price(
    S, K, T, r, sigma, option_type="call"
):
    """
    Black-Scholes-Merton option price.

    Parameters:
    - S: underlying price (float or np.array)
    - K: strike price
    - T: time to expiry in years
    - r: risk-free interest rate (annual)
    - sigma: volatility (annual std dev)
    - option_type: 'call' or 'put'

    Returns:
    - Option price (float or np.array)
    """

    if T <= 0 or sigma <= 0:
        # Option expired or zero vol, value is intrinsic value
        if option_type == "call":
            return max(S - K, 0)
        elif option_type == "put":
            return max(K - S, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def implied_volatility_from_price(
    market_price, S, K, T, r, option_type="call", sigma_bounds=(1e-6, 5), tol=1e-6, maxiter=100
):
    """
    Calculate implied volatility given a market price using Brent's method.

    Parameters:
    - market_price: observed option price
    - S: underlying price
    - K: strike price
    - T: time to expiry in years
    - r: risk-free rate
    - option_type: 'call' or 'put'
    - sigma_bounds: tuple of (min_vol, max_vol) for search
    - tol: tolerance for root-finder
    - maxiter: max iterations for root-finder

    Returns:
    - Implied volatility (float)
    """

    def objective_function(sigma):
        price = bsm_price(S, K, T, r, sigma, option_type)
        return price - market_price

    try:
        implied_vol = brentq(
            objective_function, sigma_bounds[0], sigma_bounds[1], xtol=tol, maxiter=maxiter
        )
    except ValueError:
        # Brent's method failed to converge, possibly no root in bounds
        implied_vol = np.nan

    return implied_vol


def vectorized_implied_volatility(
    prices_df,  # pd.DataFrame with option prices (index=date, columns=contracts)
    underlying_prices_df,  # pd.DataFrame with underlying prices (index=date, columns=contracts)
    strikes_dict,  # dict {contract: strike_price}
    expiries_dict,  # dict {contract: expiry_date (datetime)}
    r=0.0,  # risk-free rate
    option_types_dict=None,  # dict {contract: 'call' or 'put'}
):
    """
    Calculate implied volatilities for a dataframe of option prices.

    Returns a DataFrame with same shape as prices_df with IV values.

    Notes:
    - Assumes prices_df and underlying_prices_df have same indices.
    - Use contract metadata (strike, expiry, option_type) for each column.
    """

    iv_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)

    for contract in prices_df.columns:
        K = strikes_dict.get(contract)
        expiry_date = expiries_dict.get(contract)
        option_type = option_types_dict.get(contract, "call") if option_types_dict else "call"

        if K is None or expiry_date is None:
            # Missing metadata, skip contract
            continue

        for current_date, price in prices_df[contract].iteritems():
            if pd.isna(price):
                iv_df.at[current_date, contract] = np.nan
                continue

            S = underlying_prices_df.at[current_date, contract] if contract in underlying_prices_df.columns else np.nan
            if pd.isna(S):
                iv_df.at[current_date, contract] = np.nan
                continue

            T = time_to_expiry_in_years(current_date, expiry_date)
            if T <= 0:
                iv_df.at[current_date, contract] = np.nan
                continue

            iv = implied_volatility_from_price(price, S, K, T, r, option_type)
            iv_df.at[current_date, contract] = iv

    return iv_df