#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 17:15:31 2025

@author: gwladyscrenn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
import yfinance as yf
from datetime import datetime



class OptionPricingTool:
    
    def __init__(self, ticker, option_type='call', strike=None, expiration_index=2):

        self.ticker = ticker.upper()
        self.option_type = option_type.lower()
        self.strike = strike
        self.expiration_index = expiration_index
        
        # Fetch market data
        self._fetch_market_data()
        self._calculate_historical_volatility()
        self._get_dividend_yield()
     
    # Fetch stock and market data from Yahoo Finance
    def _fetch_market_data(self):
        self.stock = yf.Ticker(self.ticker)
            
        # Current stock price
        hist = self.stock.history(period='5d')
        self.S = hist['Close'].iloc[-1]
            
        # Option
        expirations = self.stock.options
        self.expiration = expirations[min(self.expiration_index, len(expirations)-1)]
        
        options_chain = self.stock.option_chain(self.expiration)
        self.options_df = options_chain.calls if self.option_type == 'call' else options_chain.puts
            
        exp_date = datetime.strptime(self.expiration, '%Y-%m-%d')
        self.T = (exp_date - datetime.now()).days / 365.0

            
        # Strike price = ATM if not specified
        if self.strike is None:
            self.strike = self.options_df.iloc[(self.options_df['strike'] - self.S).abs().argsort()[:1]]['strike'].values[0]
            
        # Market Price
        option_row = self.options_df[self.options_df['strike'] == self.strike]
            
        self.market_price = option_row['lastPrice'].values[0]
        self.market_iv = option_row['impliedVolatility'].values[0]
            
        # Risk-free rate (ois?)
        self._get_risk_free_rate()
            
        print(f"  {self.ticker} {self.option_type.upper()} option")
        print(f"  Stock Price: ${self.S}")
        print(f"  Strike: ${self.strike}")
        print(f"  Expiration: {self.expiration} ({self.T*365:.2} days)")
        print(f"  Market Price: ${self.market_price}")
        print(f"  Market IV: {self.market_iv:.2}\n")
        print(f"  Risk-free Rate: {self.r:.2}\n")


    def _get_risk_free_rate(self):
        if self.T <= 0.25:  # <= 3M
            ticker = "^IRX"  
        elif self.T <= 0.5:  # <= 6M
            ticker = "^FVX"
        elif self.T <= 2:    # <= 2Y
            ticker = "^TNX"  
        else:
            ticker = "^TYX"
            
        treasury = yf.Ticker(ticker)
        treasury_data = treasury.history(period='5d')
            
        if not treasury_data.empty:
            self.r = treasury_data['Close'].iloc[-1] / 100
        else:
            self.r = 0.05

    
    def _calculate_historical_volatility(self):
        try:
            hist = self.stock.history(period='6mo')['Close']
            log_returns = np.log(hist / hist.shift(1)).dropna()
            self.hist_vol = log_returns.std() * np.sqrt(252)
            print(f"  Historical Volatility (6mo): {self.hist_vol:.2}\n")
        except:
            self.hist_vol = 0.20
    
    def _get_dividend_yield(self):
        info = self.stock.info
        if 'dividendYield' in info and info['dividendYield']:
            raw_q = info['dividendYield']
        elif 'trailingAnnualDividendYield' in info and info['trailingAnnualDividendYield']:
            raw_q = info['trailingAnnualDividendYield']
        else:
            raw_q = 0.0
        self.q = raw_q / 100
        print(f"  Dividend Yield: {self.q:.4f}\n")
            

    # Black-Scholes model
    def black_scholes(S, K, T, r, sigma, option_type, q):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
        
        return price
    
    def calculate_bs_greeks(self):
        S, K, T, r, sigma = self.S, self.strike, self.T, self.r, self.hist_vol
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if self.option_type == 'call':
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if self.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    
    

    # Monte-Carlo method
    def monte_carlo_price(self, simulations=100000):
        np.random.seed(42)
        Z = np.random.standard_normal(simulations)
        ST = self.S * np.exp((self.r - 0.5 * self.hist_vol**2) * self.T + self.hist_vol * np.sqrt(self.T) * Z)
        
        if self.option_type == 'call':
            payoff = np.maximum(ST - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - ST, 0)
        
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        std_error = np.exp(-self.r * self.T) * np.std(payoff) / np.sqrt(simulations)
        
        return price, std_error
    


    
    # SABR Model
    
    def sabr_implied_volatility(K, F, T, alpha, beta, rho, nu):
        eps = 1e-7
        logFK = np.log(F / K)
        FK = (F * K) ** ((1 - beta) / 2)
        
        if abs(logFK) > eps:
            z = (nu / alpha) * FK * logFK
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            num = alpha * (1 + ((1 - beta)**2 / 24 * alpha**2 / FK**2 + rho * beta * nu * alpha / (4 * FK) + (2 - 3 * rho**2) / 24 * nu**2) * T)
            denom = FK * (1 + (1 - beta)**2 / 24 * logFK**2 + (1 - beta)**4 / 1920 * logFK**4)
            return num / denom * z / x_z
        else:
            # ATM 
            return alpha / (F ** (1 - beta)) * (1 + ((1 - beta)**2 / 24 * alpha**2 / (F ** (2 - 2 * beta)) + rho * beta * nu * alpha / (4 * F ** (1 - beta)) + (2 - 3 * rho**2) / 24 * nu**2) * T)
    
    
    def sabr_price(self, K, alpha, beta, rho, nu):
        
        F = self.S * np.exp((self.r - self.q) * self.T)
        sabr_iv = OptionPricingTool.sabr_implied_volatility(K, F, self.T, alpha, beta, rho, nu)
        
        # Price using Black-Scholes with SABR IV
        return OptionPricingTool.black_scholes(self.S, K, self.T, self.r, sabr_iv, self.option_type, self.q)
    
    def calibrate_sabr(self):

        valid = (self.options_df['impliedVolatility'] > 0) & (self.options_df['volume'] > 0)
        options = self.options_df[valid]       
        mask = (options['strike'] > 0.85 * self.S) & (options['strike'] < 1.15 * self.S)
        options = options[mask]
        
        if len(options) < 5:
            # Default params
            atm_iv = self.market_iv if self.market_iv > 0 else 0.2
            return [atm_iv, 0.5, 0.0, 0.4]
        
        strikes = options['strike'].values
        market_ivs = options['impliedVolatility'].values
        
        F = self.S * np.exp((self.r - self.q) * self.T)
        
        def objective(params):
            alpha, rho, nu = params
            beta = 0.5
            errors = []
            for K, market_iv in zip(strikes, market_ivs):
                sabr_iv = OptionPricingTool.sabr_implied_volatility(K, F, self.T, alpha, beta, rho, nu)
                if not np.isnan(sabr_iv) and market_iv > 0:
                    errors.append((sabr_iv - market_iv)**2)
            return np.sum(errors) if len(errors)>=3 else 1e10
        
        # Initial guess
        atm_iv = market_ivs[len(market_ivs)//2] if len(market_ivs) > 0 else 0.2
        initial_alpha = atm_iv * (F ** 0.5)
        initial_guess = [initial_alpha, 0.0, 0.4]
        bounds = [(0.001, 1.0), (-0.95, 0.95), (0.01, 1.5)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': 100})
        # Validate calibration
        if result.fun < 0.1:  # Check if calibration was successful
            alpha, rho, nu = result.x
            beta = 0.5
            return [alpha, beta, rho, nu]
        else:
            print("No convergence")
            return [atm_iv * (F ** 0.5), 0.5, 0.0, 0.4]

       


    # Binomial model 
    def binomial_tree_price(self, steps=100):
        dt = self.T / steps
        u = np.exp(self.hist_vol * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        ST = np.array([self.S * u**(steps - i) * d**i for i in range(steps + 1)])
        
        if self.option_type == 'call':
            V = np.maximum(ST - self.strike, 0)
        else:
            V = np.maximum(self.strike - ST, 0)
        
        for j in range(steps - 1, -1, -1):
            V = np.exp(-self.r * dt) * (p * V[:-1] + (1 - p) * V[1:])
        
        return V[0]
    
    
    def implied_volatility(market_price, S, K, T, r, option_type, q, max_iter=100):
        if market_price <= 0 or T <= 0:
            return np.nan
        
        sigma = 0.2
        
        for i in range(max_iter):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                
            # Calculate price and vega
            bs_price = OptionPricingTool.black_scholes(S, K, T, r, sigma, option_type, q)
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
                
            if vega < 1e-10:
                return np.nan
                
            # Newton-Raphson update
            diff = bs_price - market_price
            if abs(diff) < 1e-6:
                return sigma
                
            sigma = sigma - diff / vega
                
            if sigma <= 0.001:
                sigma = 0.001
            if sigma > 5:
                sigma = 5
        
        return sigma




    
    def run_analysis(self):
        
        results = {}
        
        # BLACK-SCHOLES
        # Using historical volatility
        bs_price_hist = OptionPricingTool.black_scholes(self.S, self.strike, self.T, self.r, self.hist_vol, self.option_type, self.q)
        # Recalculate IV from market price to verify
        calculated_iv = OptionPricingTool.implied_volatility(self.market_price, self.S, self.strike, self.T, self.r, self.option_type, self.q)
        
        bs_greeks = self.calculate_bs_greeks()
        results['BS (Historical Vol)'] = {'price': bs_price_hist, 'greeks': bs_greeks}
        print(f"Price (Historical Vol): ${bs_price_hist}")
        print(f"  Market reported IV:   {self.market_iv:.4f} ({self.market_iv*100:.2f}%)")
        print(f"  Calculated IV:        {calculated_iv:.4f} ({calculated_iv*100:.2f}%)")
        print(f"  Difference:           {abs(self.market_iv - calculated_iv):.4f}")
        print(f"Greeks (Historical Vol):")
        for greek, value in bs_greeks.items():
            print(f"  {greek}: {value:}")
        
        # Monte Carlo
        print("\n2. MONTE CARLO SIMULATION")
        mc_price, mc_error = self.monte_carlo_price()
        results['Monte Carlo'] = {'price': mc_price, 'std_error': mc_error}
        print(f"Price: ${mc_price} ± ${mc_error}")
        
        # Binomial Tree
        print("\n3. BINOMIAL TREE MODEL")
        bt_price = self.binomial_tree_price()
        results['Binomial Tree'] = {'price': bt_price}
        print(f"Price: ${bt_price}")
        
        # 4. SABR Model
        print("\n4. SABR MODEL")
        sabr_params = self.calibrate_sabr()
        sabr_price = self.sabr_price(self.strike, *sabr_params)
        results['SABR'] = {
            'price': sabr_price,
            'params': {
                'alpha': sabr_params[0],
                'beta': sabr_params[1],
                'rho': sabr_params[2],
                'nu': sabr_params[3]
            }
        }
        print(f"Price: ${sabr_price}")
        print(f"Parameters: α={sabr_params[0]:.2f}, β={sabr_params[1]:.2f}, "
              f"ρ={sabr_params[2]:.2f}, ν={sabr_params[3]:.2f}")
        
        # Market Price
        print("\n5. MARKET PRICE")
        print(f"Price: ${self.market_price}")
        print(f"Implied Volatility: {self.market_iv}")
        
        # Summary
        print("PRICING COMPARISON")
        print(f"{'Model':<20} {'Price':<12} {'Diff from Market':<20}")
        print("-" * 70)
        
        for model, data in results.items():
            diff = ((data['price'] - self.market_price) / self.market_price) * 100
            print(f"{model:<20} ${data['price']:<11.4f} {diff:+.2f}%")
        
        print(f"{'Market':<20} ${self.market_price:<11.4f} {'--':<20}")
        
        self.results = results
        return results
    
    def plot_results(self):
        # Price comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Model prices vs Market
        models = ['BS_HV', 'MC', 'BT', 'SABR', 'Market']
        prices = [
            self.results['BS (Historical Vol)']['price'],
            self.results['Monte Carlo']['price'],
            self.results['Binomial Tree']['price'],
            self.results['SABR']['price'],
            self.market_price
        ]
        colors = ['blue', 'green', 'orange', 'red', 'black']
        
        ax1.bar(models, prices, color=colors, alpha=0.7)
        ax1.axhline(y=self.market_price, color='black', linestyle='--', label='Market Price', linewidth=2)
        ax1.set_ylabel('Option Price ($)')
        ax1.set_title(f'{self.ticker} {self.option_type.upper()} Option Price Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Greeks
        greeks_data = self.results['BS (Historical Vol)']['greeks']
        greeks_names = list(greeks_data.keys())
        greeks_values = list(greeks_data.values())        
        ax2.barh(greeks_names, greeks_values, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Value')
        ax2.set_title('Black-Scholes Greeks')
        ax2.grid(True, alpha=0.3, axis='x')      
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_option_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Volatility smile analysis
        self.plot_volatility_smile()
        self.plot_errors()
    
    def plot_volatility_smile(self):
        valid = (self.options_df['impliedVolatility'] > 0) & (self.options_df['volume'] > 0)
        options = self.options_df[valid]    
        mask = (options['strike'] > 0.7 * self.S) & (options['strike'] < 1.3 * self.S)
        options = options[mask].sort_values('strike')
        
        fig, ax = plt.subplots(figsize=(10, 6))     
        strikes = options['strike'].values
        market_iv = options['impliedVolatility'].values      
        ax.plot(strikes, market_iv, 'o-', linewidth=2, markersize=8, label='Market IV', color='blue')
        ax.axvline(x=self.S, color='red', linestyle='--', label=f'Spot Price (${self.S:.2f})', alpha=0.7)
        ax.axvline(x=self.strike, color='green', linestyle='--', label=f'Selected Strike (${self.strike:.2f})', alpha=0.7)
        ax.set_xlabel('Strike Price ($)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'{self.ticker} {self.option_type.upper()} Options - Volatility Smile\n Expiration: {self.expiration}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_volatility_smile.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Error comparison
    def plot_errors(self):
       errors = []
       error_labels = []
       for model in ['BS (Historical Vol)', 'Monte Carlo', 'Binomial Tree', 'SABR']:
           error = abs(self.results[model]['price'] - self.market_price)
           errors.append(error)
           error_labels.append(model.replace(' ', '\n'))
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.bar(error_labels, errors, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
       plt.tight_layout()
       plt.savefig(f'{self.ticker} {self.option_type} - Error analysis.png')

