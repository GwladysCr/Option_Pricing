import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# Black-Scholes formula 
def black_scholes(S, K, T, r, sigma, type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1) 
    
    return price

# Greeks calculation
def greeks_bs(price, K, T, r, sigma, N, option_type, h=1e-2):
    delta = (black_scholes(S + h, K, T, r, sigma, N, option_type) -
             black_scholes(S - h, K, T, r, sigma, N, option_type)) / (2 * h)
    gamma = (black_scholes(S + h, K, T, r, sigma, N, option_type) -
             2 * price +
             black_scholes(S - h, K, T, r, sigma, N, option_type)) / (h ** 2)
    theta = (black_scholes(S, K, T - h, r, sigma, N, option_type) -
             price) / h
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta}

# Characteristic function for Heston model
def heston_characteristic_function(phi, S, T, r, kappa, theta, sigma, rho, v0, u):
    x = np.log(S)
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

    C = (r * phi * 1j * T) + (a / sigma**2) * ((b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = ((b - rho * sigma * phi * 1j + d) / sigma**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + 1j * phi * x)

# Heston model pricing via Fourier integration
def heston_price(S, K, T, r, kappa, theta, sigma, rho, v0, option_type):
    def integrand(phi, u):
        cf = heston_characteristic_function(phi, S, T, r, kappa, theta, sigma, rho, v0, u)
        return (np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)).real

    # Integrate for P1 and P2
    integral1 = quad(lambda phi: integrand(phi, 0.5), 0, 100, limit=200)[0]
    integral2 = quad(lambda phi: integrand(phi, -0.5), 0, 100, limit=200)[0]
    P1 = 0.5 + (1 / np.pi) * integral1
    P2 = 0.5 + (1 / np.pi) * integral2
    if option_type == 'call':
        price = S * P1 - K * np.exp(-r * T) * P2
    elif option_type == 'put':
        price = K * np.exp(-r * T) * (1 - P2) - S * (1 - P1)
    return price

# Greeks calculation
def greeks_he(price, K, T, r, sigma, N, option_type, h=1e-2):
    delta = (heston_price(S + h, K, T, r, sigma, N, option_type) -
             heston_price(S - h, K, T, r, sigma, N, option_type)) / (2 * h)
    gamma = (heston_price(S + h, K, T, r, sigma, N, option_type) -
             2 * price +
             heston_price(S - h, K, T, r, sigma, N, option_type)) / (h ** 2)
    theta = (heston_price(S, K, T - h, r, sigma, N, option_type) -
             price) / h
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta}

# Monte carlo method
def monte_carlo_price(S, K, T, r, sigma, option_type, simulations=100000):
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

# Greeks
def greeks_mc(S, K, T, r, sigma, option_type, h=1e-2):
    price = monte_carlo_price(S, K, T, r, sigma, option_type)
    delta = (monte_carlo_price(S + h, K, T, r, sigma, option_type) -
             monte_carlo_price(S - h, K, T, r, sigma, option_type)) / (2 * h)
    gamma = (monte_carlo_price(S + h, K, T, r, sigma, option_type) -
             2 * price + monte_carlo_price(S - h, K, T, r, sigma, option_type)) / (h ** 2)
    theta = (monte_carlo_price(S, K, T - h, r, sigma, option_type) - price) / h
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta}


#___________________MAIN___________________
# Parameters
S = 100
T = 1
r = 0.05
sigma = 0.2
strike_prices = np.linspace(80, 120, 20)

# Heston model parameters
kappa = 2.0
theta = 0.04
v0 = sigma**2
rho = -0.7

# Calculate prices
bs_prices = [black_scholes(S, K, T, r, sigma, 'call') for K in strike_prices]
heston_prices = [heston_price(S, K, T, r, kappa, theta, sigma, rho, v0, 'call') for K in strike_prices]
mc_prices = [monte_carlo_price(S, K, T, r, sigma, 'call') for K in strike_prices]


# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(strike_prices, bs_prices, label='Black-Scholes', marker='o')
plt.plot(strike_prices, mc_prices, label='Monte-Carlo', marker='v')
plt.plot(strike_prices, heston_prices, label='Heston Model', marker='x')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title('European Call Option Pricing: Black-Scholes vs Heston Model vs Monte Carlo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Comparison for european option.png')