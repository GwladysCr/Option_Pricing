import numpy as np
from scipy.stats import norm


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Closed-form Black-Scholes-Merton Greeks"""
    if T <= 0 or sigma <= 0:
        return {"Delta": np.nan, "Gamma": np.nan, "Theta": np.nan, "Vega": np.nan, "Rho": np.nan}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # per 1 vol point (0.01)

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}


def fd_greeks(price_fn, S, K, T, r, sigma, option_type="call", q=0.0, hS=None, hSig=1e-3, hT=1e-3):
    """Central finite-difference Delta/Gamma/Vega and forward-difference Theta"""
    hS = hS if hS is not None else max(S * 1e-3, 1e-2)

    base = price_fn(S, K, T, r, sigma, option_type, q)
    up = price_fn(S + hS, K, T, r, sigma, option_type, q)
    down = price_fn(S - hS, K, T, r, sigma, option_type, q)
    delta = (up - down) / (2 * hS)
    gamma = (up - 2 * base + down) / (hS ** 2)

    vol_up = price_fn(S, K, T, r, sigma + hSig, option_type, q)
    vol_down = price_fn(S, K, T, r, max(sigma - hSig, 1e-6), option_type, q)
    vega = (vol_up - vol_down) / (2 * hSig) / 100

    T_down = max(T - hT, 1e-6)
    theta = (price_fn(S, K, T_down, r, sigma, option_type, q) - base) / hT / 365

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "price": base}
