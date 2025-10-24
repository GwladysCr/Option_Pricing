import numpy as np

# Binomial Tree algorithm
def binomial_tree(S, K, T, r, sigma, N, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    #At maturity
    asset_prices = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    option_values = np.maximum(0, asset_prices - K) if option_type == 'call' else np.maximum(0, K - asset_prices)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            asset_price = S * (u ** j) * (d ** (i - j))
            hold = np.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
            exercise = max(0, asset_price - K) if option_type == 'call' else max(0, K - asset_price)
            option_values[j] = max(hold, exercise)

    return option_values[0]

# Greeks estimation
def american_option_greeks(S, K, T, r, sigma, N, option_type, h=1e-2):
    price = binomial_tree(S, K, T, r, sigma, N, option_type)
    delta = (binomial_tree(S + h, K, T, r, sigma, N, option_type) -
             binomial_tree(S - h, K, T, r, sigma, N, option_type)) / (2 * h)
    gamma = (binomial_tree(S + h, K, T, r, sigma, N, option_type) -
             2 * price +
             binomial_tree(S - h, K, T, r, sigma, N, option_type)) / (h ** 2)
    theta = (binomial_tree(S, K, T - h, r, sigma, N, option_type) -
             price) / h
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta}

# Parameters
S = 100         # Spot price
K = 100         # Strike price
T = 1           # Time to maturity
r = 0.05        # Risk-free rate
sigma = 0.2     # Volatility
N = 100         # Steps in binomial tree
option_type = 'call'

# Price and Greeks
price = binomial_tree(S, K, T, r, sigma, N, option_type)
greeks = american_option_greeks(S, K, T, r, sigma, N, option_type)

# Print results
print(f"American {option_type} option price: {price} \n")
for greek, value in greeks.items():
    print(f"{greek}: {value}")