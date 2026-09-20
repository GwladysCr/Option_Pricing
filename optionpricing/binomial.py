import numpy as np


def price(S, K, T, r, sigma, option_type = "call", q = 0.0, steps = 200, american=False):
    """Binomial tree for European and American vanilla options"""

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not 0.0 < p < 1.0:
        raise ValueError("Risk-neutral probability outside [0,1]")

    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d ** j

    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    discount = np.exp(-r * dt)

    for _ in range(steps):
        V = discount * (p * V[:-1] + (1.0 - p) * V[1:])

        if american:
            n = len(V) - 1
            j = np.arange(n + 1)
            S_t = S * u ** (n - j) * d ** j
            exercise = (np.maximum(S_t - K, 0.0) if option_type == "call" else np.maximum(K - S_t, 0.0))
            V = np.maximum(V, exercise)

    return float(V[0])
