import numpy as np
from scipy.optimize import minimize
from .black_scholes import price as bs_price


def implied_volatility(K, F, T, alpha, beta, rho, nu):
    """Hagan SABR lognormal implied-volatility approximation"""
    eps = 1e-10
    if min(F, K, T, alpha, nu) <= 0:
        return np.nan

    log_fk = np.log(F / K)
    FK = (F * K) ** ((1.0 - beta) / 2.0)
    correction = (((1 - beta) ** 2 / 24) * alpha**2 / FK**2 + (rho * beta * nu * alpha) / (4 * FK) + ((2 - 3 * rho**2) / 24) * nu**2)

    if abs(log_fk) < eps:
        return (alpha / F ** (1 - beta) * (1 + correction * T))

    z = (nu / alpha) * FK * log_fk
    sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
    x_z = np.log((1 + rho) / (sqrt_term - z + rho))
    numerator = alpha * (1 + correction * T)
    denominator = FK * (1 + ((1 - beta) ** 2 / 24) * log_fk**2 + ((1 - beta) ** 4 / 1920) * log_fk**4)

    return numerator / denominator * z / x_z


def calibrate(strikes, market_ivs, F, T, beta=0.5):
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)

    mask = np.isfinite(strikes) & np.isfinite(market_ivs) & (market_ivs > 0)
    strikes, market_ivs = strikes[mask], market_ivs[mask]

    if len(strikes) < 5:
        raise ValueError("At least 5 valid option IV observations are required")

    atm_idx = np.argmin(np.abs(strikes - F))
    alpha0 = market_ivs[atm_idx] * F ** (1 - beta)

    def objective(x):
        alpha, rho, nu = x
        model = np.array([implied_volatility(K, F, T, alpha, beta, rho, nu) for K in strikes])

        if np.any(~np.isfinite(model)):
            return 1e6
        return float(np.mean((model - market_ivs) ** 2))

    result = minimize(
        objective,
        x0=[alpha0, 0.0, 0.5],
        bounds=[(1e-5, 15.0), (-0.999, 0.999), (1e-5, 15.0)],
        method="L-BFGS-B",
        options={"maxiter": 500}
    )

    if not result.success:
        raise RuntimeError(f"SABR calibration failed: {result.message}")

    alpha, rho, nu = result.x
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "nu": float(nu),
        "iv_rmse": float(np.sqrt(result.fun))
    }


def price(S, K, T, r, q, option_type, params):
    F = S * np.exp((r - q) * T)
    iv = implied_volatility(K, F, T, params["alpha"], params["beta"], params["rho"], params["nu"])
    return bs_price(S, K, T, r, iv, option_type, q)
