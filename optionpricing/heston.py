import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

from .black_scholes import price as bs_price, vega as bs_vega


# ---------------------------------------------------------------------------
# Heston stochastic volatility model with Fourier-integral representation
#
#   dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
#   dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t,   corr(dW1, dW2) = rho
# ---------------------------------------------------------------------------

PARAMS = ("kappa", "theta", "sigma_v", "rho", "v0")


def _char_func(u, T, r, q, kappa, theta, sigma_v, rho, v0):
    x = 1j * u
    d = np.sqrt((rho * sigma_v * x - kappa) ** 2 - sigma_v ** 2 * (-x - u ** 2))
    g = (kappa - rho * sigma_v * x - d) / (kappa - rho * sigma_v * x + d)

    C = ((r - q) * x * T + (kappa * theta / sigma_v ** 2) * ((kappa - rho * sigma_v * x - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
    D = ((kappa - rho * sigma_v * x - d) / sigma_v ** 2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))))

    return np.exp(C + D * v0)


def _call_price_scalar(S, K, T, r, q, kappa, theta, sigma_v, rho, v0):
    if T <= 0:
        return max(0.0, S - K)

    def integrand(u):
        cf = _char_func(u - 0.5j, T, r, q, kappa, theta, sigma_v, rho, v0)
        return (np.exp(1j * u * np.log(S / K)) * cf / (u ** 2 + 0.25)).real

    integral, _ = quad(integrand, 0, 250, limit=250)
    call = S * np.exp(-q * T) - (np.exp(-r * T) * np.sqrt(S * K) / np.pi) * integral
    return max(0.0, call)


_GL_NODES_CACHE = {}


def _gauss_legendre_nodes(n=128, u_max=400.0):
    key = (n, u_max)
    if key not in _GL_NODES_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        u = 0.5 * u_max * (x + 1)
        wu = 0.5 * u_max * w
        _GL_NODES_CACHE[key] = (u, wu)
    return _GL_NODES_CACHE[key]


def _call_price_vector(S, strikes, T, r, q, kappa, theta, sigma_v, rho, v0, n_nodes=128, u_max=400.0):
    strikes = np.asarray(strikes, dtype=float)
    if T <= 0:
        return np.maximum(S - strikes, 0.0)

    u, w = _gauss_legendre_nodes(n_nodes, u_max)
    cf = _char_func(u - 0.5j, T, r, q, kappa, theta, sigma_v, rho, v0)

    log_SK = np.log(S / strikes)
    phase = np.exp(1j * np.outer(u, log_SK))
    integrand = (phase * cf[:, None] / (u[:, None] ** 2 + 0.25)).real

    integral = w @ integrand
    calls = S * np.exp(-q * T) - (np.exp(-r * T) * np.sqrt(S * strikes) / np.pi) * integral
    return np.maximum(calls, 0.0)


def price_vector(S, strikes, T, r, q, option_type, params):
    missing = [p for p in PARAMS if p not in params]
    if missing:
        raise ValueError(f"heston.price_vector: missing params {missing}")

    strikes = np.asarray(strikes, dtype=float)
    calls = _call_price_vector(S, strikes, T, r, q, params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"])
    if option_type == "call":
        return calls
    return calls - S * np.exp(-q * T) + strikes * np.exp(-r * T)


def price(S, K, T, r, q, option_type, params):
    missing = [p for p in PARAMS if p not in params]
    if missing:
        raise ValueError(f"heston.price: missing params {missing}")

    call = _call_price_scalar(S, K, T, r, q, params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"])
    if option_type == "call":
        return call
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)  # put-call parity


def fd_greeks(params, S, K, T, r, q, option_type="call", hS=None, hVol=1e-2, hT=1e-3):
    hS = hS if hS is not None else max(S * 1e-3, 1e-2)

    base = price(S, K, T, r, q, option_type, params)
    up = price(S + hS, K, T, r, q, option_type, params)
    down = price(S - hS, K, T, r, q, option_type, params)
    delta = (up - down) / (2 * hS)
    gamma = (up - 2 * base + down) / (hS ** 2)

    vol0 = np.sqrt(params["v0"])
    v_up = (vol0 + hVol) ** 2
    v_down = max(vol0 - hVol, 1e-6) ** 2
    p_up = price(S, K, T, r, q, option_type, {**params, "v0": v_up})
    p_down = price(S, K, T, r, q, option_type, {**params, "v0": v_down})
    vega = (p_up - p_down) / 2  

    T_down = max(T - hT, 1e-6)
    theta = (price(S, K, T_down, r, q, option_type, params) - base) / hT / 365

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "price": base}


def calibrate(strikes, market_ivs, S, r, q, T, option_type="call", initial_guess=None, bounds=None):
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)

    mask = np.isfinite(strikes) & np.isfinite(market_ivs) & (market_ivs > 0)
    strikes, market_ivs = strikes[mask], market_ivs[mask]

    if len(strikes) < 5:
        raise ValueError("At least 5 valid option IV observations are required")

    market_prices = np.array([bs_price(S, K, T, r, iv, option_type, q) for K, iv in zip(strikes, market_ivs)])
    vegas = np.array([bs_vega(S, K, T, r, iv, q) for K, iv in zip(strikes, market_ivs)])
    vegas = np.maximum(vegas, 1e-4)

    if initial_guess is None:
        v0_guess = float(np.median(market_ivs)) ** 2
        initial_guess = [1.5, v0_guess, 0.5, -0.5, v0_guess]  # kappa, theta, sigma_v, rho, v0
    if bounds is None:
        bounds = [(1e-2, 10.0), (1e-4, 2.0), (1e-2, 5.0), (-0.99, 0.99), (1e-4, 2.0)]

    def objective(x):
        kappa, theta, sigma_v, rho, v0 = x
        feller_penalty = 5.0 if 2 * kappa * theta <= sigma_v ** 2 else 0.0

        calls = _call_price_vector(S, strikes, T, r, q, kappa, theta, sigma_v, rho, v0)
        if option_type == "call":
            model_prices = calls
        else:
            model_prices = calls - S * np.exp(-q * T) + strikes * np.exp(-r * T)

        if np.any(~np.isfinite(model_prices)):
            return 1e6
        return float(np.sum(((model_prices - market_prices) / vegas) ** 2)) + feller_penalty

    result = minimize(objective, x0=initial_guess, bounds=bounds, method="L-BFGS-B", options={"maxiter": 150})

    if not result.success and result.fun >= 1e6:
        raise RuntimeError(f"Heston calibration failed: {result.message}")

    kappa, theta, sigma_v, rho, v0 = result.x
    feller_violated = bool(2 * kappa * theta <= sigma_v ** 2)
    fit_err = result.fun - (5.0 if feller_violated else 0.0)

    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_v": float(sigma_v),
        "rho": float(rho),
        "v0": float(v0),
        "feller_violated": feller_violated,
        "fit_err": float(fit_err),
    }
