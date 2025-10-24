import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
import yfinance as yf
from scipy.optimize import brentq, minimize


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


def implied_vol(price, S, K, T, r, option_type):
    def f(vol):   
        return black_scholes(S, K, T, r, vol, option_type) - price 
    low_price = black_scholes(S, K, T, r, 1e-6, option_type)
    high_price = black_scholes(S, K, T, r, 5, option_type)
    if not (low_price <= price <= high_price):
        return np.nan
    return brentq(f, 1e-6, 5)


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

def calibrate_heston(strikes, market_prices, S, T, r):
    def objective(params):
        kappa, theta, sigma, rho, v0 = params
        model_prices = [heston_price(S, K, T, r, kappa, theta, sigma, rho, v0, 'call') for K in strikes]
        return np.sum(((np.array(model_prices) - np.array(market_prices)) / np.array(market_prices))**2)
    initial_guess = [2.0, 0.04, 0.2, -0.7, 0.04]
    bounds = [(0.01, 10), (0.001, 1), (0.01, 1), (-0.99, 0.99), (0.001, 1)]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

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
def monte_carlo_price(S, K, T, r, sigma, option_type, simulations=300000):
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


# SABR IV
def sabr_vol(K, F, T, alpha, beta, rho, volvol):
    if K == F:
        return alpha / (F**(1 - beta)) * (1 + ((1 - beta)**2 * alpha**2 / (24 * F**(2 - 2*beta)) + rho * beta * volvol * alpha / (4 * F**(1 - beta)) + volvol**2 * (2 - 3 * rho**2) / 24) * T)
    z = volvol / alpha * (F * K)**((1 - beta)/2) * np.log(F / K)
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    denom = (F * K)**((1 - beta)/2) * (1 + ((1 - beta)**2 / 24) * (np.log(F / K))**2 + ((1 - beta)**4 / 1920) * (np.log(F / K))**4)
    return (alpha / denom) * (z / x_z) * (1 + ((1 - beta)**2 * alpha**2 / (24 * (F * K)**(1 - beta)) + rho * beta * volvol * alpha / (4 * (F * K)**((1 - beta)/2)) + volvol**2 * (2 - 3 * rho**2) / 24) * T)

# SABR Calibration
def calibrate_sabr(strikes, market_iv, F, T):
    def objective(params):
        alpha, rho, volvol = params
        beta = 0.5
        model_iv = [sabr_vol(K, F, T, alpha, beta, rho, volvol) for K in strikes]
        return np.sum((np.array(model_iv) - np.array(market_iv))**2)
    initial_guess = [0.2, -0.3, 0.3]
    bounds = [(0.01, 2), (-0.99, 0.99), (0.01, 2)]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

# Local Volatility
def local_volatility(strikes, iv, T):
    from numpy import gradient
    K = np.array(strikes)
    sigma = np.array(iv)
    d_sigma = gradient(sigma, K)
    d2_sigma = gradient(d_sigma, K)
    lv = sigma**2 + 2 * K * sigma * d_sigma + K**2 * (d_sigma**2 + sigma * d2_sigma)
    return np.sqrt(np.maximum(lv, 0))


#___________________MAIN___________________

# Fetch option data
ticker = 'AAPL'
option_type = 'call'

stock = yf.Ticker(ticker)
expiration = stock.options[0]
options_chain = stock.option_chain(expiration)
dir = options_chain.calls if option_type == 'call' else options_chain.puts

# Market data
S = stock.history(period = '1d')['Close'].iloc[-1]
r = 0.05
T = 30/365 # 30 days to expiry

valid = (dir['impliedVolatility'] > 0) & (dir['volume'] > 0)
market_prices = dir['lastPrice'][valid].values
strikes = dir['strike'][valid].values

mask = (strikes > 0.8 * S) & (strikes < 1.2 * S)
strikes = strikes[mask]
market_prices = market_prices[mask]

# Historical volatility
hist = stock.history(period='6mo')['Close']
log_returns = np.log(hist / hist.shift(1)).dropna()
volat = log_returns.std() * np.sqrt(252)

# Compute market IV
market_iv = [implied_vol(p, S, K, T, r, option_type) for K, p in zip(strikes, market_prices)]


# Heston Calibration
heston_params = calibrate_heston(strikes, market_prices, S, T, r)
heston_iv = [implied_vol(heston_price(S, K, T, r, *heston_params, option_type), S, K, T, r, option_type) for K in strikes]

# SABR Calibration
alpha, rho, volvol = calibrate_sabr(strikes, market_prices, S, T)
sabr_iv = [sabr_vol(K, S, T, alpha, r, rho, volvol) for K in strikes]

# Local Volatility
local_vol = local_volatility(strikes, market_prices, T)

# Calculate prices
bs_prices = [black_scholes(S, K, T, r, volat, option_type) for K in strikes]
heston_prices = [heston_price(S, K, T, r, *heston_params, option_type) for K in strikes]
mc_prices = [monte_carlo_price(S, K, T, r, volat, option_type) for K in strikes]


# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(strikes, bs_prices, label='Black-Scholes', marker='o')
plt.plot(strikes, mc_prices, label='Monte-Carlo', marker='v')
plt.plot(strikes, market_prices, label='Market', marker='.')
plt.plot(strikes, heston_prices, label='Heston Model', marker='x')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title('European Call Option Pricing: Black-Scholes vs Heston Model vs Monte Carlo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Comparison for european option.png')

# Plot IV smooth smile
plt.figure(figsize=(10, 6))
plt.plot(strikes, market_iv, 'o-', label='Market IV')
plt.plot(strikes, heston_iv[:len(strikes)], 'x-', label='Heston IV')
plt.plot(strikes, sabr_iv, 'v-', label='SABR IV')
plt.plot(strikes, local_vol, 's-', label='Local Volatility')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Improved Volatility Smile')
plt.legend()
plt.grid(True)
plt.ylim(0, 0.6)
plt.tight_layout()
plt.savefig('IV smile models comparison.png')
