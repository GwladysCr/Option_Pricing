import numpy as np


def price(S, K, T, r, sigma, option_type = "call", q = 0.0, simulations = 100_000, seed = 42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(simulations)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    payoff = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
    discount = np.exp(-r * T)
    pv_payoff = discount * payoff

    return {
        "price": float(np.mean(pv_payoff)),
        "std_error": float(np.std(pv_payoff, ddof=1) / np.sqrt(simulations)),
        "simulations": simulations
    }
