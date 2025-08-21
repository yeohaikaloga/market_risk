import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.stats import norm


def black_scholes(S, K, T, r, q, sigma, option_type):
    F = S * np.exp((r - q) * T)  # Adjusted forward price with cost of carry
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price


def implied_volatility(option_price, S, K, T, r, q, option_type, tol=1e-10, max_iter=100):
    def black_scholes_iv(sigma):
        return black_scholes(S, K, T, r, q, sigma, option_type) - option_price

    # Use Newton-Raphson method to find implied volatility
    sigma_guess = 0.2  # Initial guess for volatility
    for _ in range(max_iter):
        option_price_guess = black_scholes_iv(sigma_guess)
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma_guess ** 2) * T) / (sigma_guess * np.sqrt(T))) * np.sqrt(
            T)
        sigma_guess -= (option_price_guess - option_price) / vega
        print('sigma: ', round(sigma_guess, 7), ', option_price_guess: ', round(option_price_guess, 7))
        if abs(option_price_guess - option_price) < tol:
            break
    print('sigma: ', round(sigma_guess, 7))
    return sigma_guess


def black_76(S, K, r, q, T, sigma, option_type):
    """
    S: Current spot price of the underlying
    K: Strike price
    r: Risk-free rate
    T: Time to maturity
    sigma: Volatility of the underlying asset
    option_type: Type of the option. Can be either 'call' or 'put'
    q: cost of carry
    """
    F = S * np.exp((r - q) * T)  # Adjusted forward price with cost of carry
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = np.exp(-r * T) * (F * si.norm.cdf(d1, 0.0, 1.0) - K * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put':
        option_price = np.exp(-r * T) * (K * si.norm.cdf(-d2, 0.0, 1.0) - F * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return option_price


def calculate_greeks(F, K, r, q, T, sigma):
    """
    Calculate the Greeks: Delta, Gamma, Vega, Theta
    """
    d1 = (np.log(F / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # delta = np.exp(-r * T) * si.norm.cdf(d1, 0.0, 1.0)
    # gamma = np.exp(-r * T) * si.norm.pdf(d1, 0.0, 1.0) / (F * sigma * np.sqrt(T))
    # vega = F * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    # theta = -F * si.norm.pdf(d1, 0.0, 1.0) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

    delta = np.exp(-r * T) * si.norm.cdf(d1, 0.0, 1.0)
    gamma = np.exp(-r * T) * si.norm.pdf(d1, 0.0, 1.0) / (F * sigma * np.sqrt(T))
    vega = F * np.exp(-q * T) * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    theta = -F * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) * sigma / (2 * np.sqrt(T)) \
            - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) \
            + q * F * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)

    return delta, gamma, vega, theta


def generate_pnl_and_greeks_vectors(F, K, r, q, T, sigma, price_move, option_type):
    """
    price_move: array of price moves of the underlying
    """
    underlying_price_vectors = []
    black_76_opt_price_vectors = []
    pnl_vectors = []
    greeks_vectors = []
    for move in price_move:
        underlying_price_moved = F * (1 + move)
        underlying_price_vectors.append(underlying_price_moved)
        F_moved = F * (1 + move)
        black_76_opt_price_moved = black_76(F_moved, K, r, q, T, sigma, option_type)
        black_76_opt_price_vectors.append(black_76_opt_price_moved)
        pnl = black_76_opt_price_moved - black_76(F, K, r, q, T, sigma, option_type)
        pnl_vectors.append(pnl)
        delta, gamma, vega, theta = calculate_greeks(F_moved, K, r, q, T, sigma)
        greeks_vectors.append((delta, gamma, vega, theta))
    return pnl_vectors, greeks_vectors, underlying_price_vectors, black_76_opt_price_vectors


def generate_sensitivity_report(product_code, price_move, active_lots, S0, K, r, T, q, settlement_price, option_type):
    observed_option_price = settlement_price / 2
    sigma = implied_volatility(observed_option_price, S0, K, T, r, q, option_type)  # Volatility of the underlying asset

    if product_code == 'CT':
        delta_MT_multiplier = 22.68
    elif product_code == 'SB':
        delta_MT_multiplier = 50.8
    else:
        delta_MT_multiplier = 1

    pnl_vectors, greeks_vectors, underlying_price_vectors, black_76_opt_price_vectors = generate_pnl_and_greeks_vectors(
        S0, K, r, q, T, sigma, price_move, option_type)

    active_lots_vectors = [active_lots for greek in greeks_vectors]
    delta_vectors = [greek[0] * active_lots for greek in greeks_vectors]
    delta_in_MT_vectors = [greek[0] * active_lots * delta_MT_multiplier for greek in greeks_vectors]
    gamma_vectors = [greek[1] * active_lots for greek in greeks_vectors]
    vega_vectors = [greek[2] * active_lots / 100 for greek in greeks_vectors]
    theta_vectors = [greek[3] * active_lots / 365 for greek in greeks_vectors]  # Convert from /yr to /day

    sensi_rep = pd.DataFrame(
        [price_move * 100, underlying_price_vectors, black_76_opt_price_vectors, active_lots_vectors, delta_vectors,
         delta_in_MT_vectors, gamma_vectors, vega_vectors, theta_vectors, pnl_vectors]).transpose()
    sensi_rep.columns = ['%_Price', 'Fut Price (USc)', 'B76 Opt Price (USc)', 'Active Lots', 'Delta', 'Delta (MT)',
                         'Gamma', 'Vega (USc/%)', 'Theta (USc/day)', 'PnL (USc)']

    print("Black-76 0-move " f"{option_type}", "option: " f"{black_76(S0, K, r, q, T, sigma, option_type):.10f}\n")
    print("sigma :", sigma)
    return sensi_rep.round(4)


def days_between(d1, d2):
    return np.busday_count(d1.date(), d2.date())
