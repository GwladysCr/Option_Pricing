import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def price(S, K, T, r, sigma, option_type = "call", q = 0.0):
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if sigma <= 0:
        forward = S * np.exp((r - q) * T)
        intrinsic = max(forward - K, 0.0) if option_type == "call" else max(K - forward, 0.0)
        return np.exp(-r * T) * intrinsic

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def vega(S, K, T, r, sigma, q = 0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(market_price, S, K, T, r, option_type="call", q=0.0):
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    l = S * np.exp(-q * T) - K * np.exp(-r * T) if option_type == "call" else K * np.exp(-r * T) - S * np.exp(-q * T)
    lower = max(0.0, l)
    upper = S * np.exp(-q * T) if option_type == "call" else K * np.exp(-r * T)

    if market_price < lower - 1e-8 or market_price > upper + 1e-8:
        return np.nan

    f = lambda sigma: price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(f, 1e-6, 5.0, maxiter=200)
    except ValueError:
        return np.nan
