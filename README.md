# Option Pricing Toolkit

Python project for pricing equity options and comparing market, Black-Scholes, Monte Carlo, binomial, SABR, and Heston approaches on realistic option chains

## Overview

This toolkit is designed to:

- download an option chain from Yahoo Finance,
- compute historical volatility and implied volatilities,
- price options under several models,
- compare model prices to market quotes,
- calibrate SABR and Heston parameters,
- generate Greeks and volatility-surface diagnostics,
- visualize model performance and calibration quality.

## Included models

- Black-Scholes with historical volatility
- Monte Carlo pricing
- Binomial European pricing
- SABR calibration and pricing
- Heston calibration and pricing
- Greeks for Black-Scholes and Heston-based setups

## Package structure

- `main.py` — entry point; loads a live option chain and runs the main pricing workflow
- `optionpricing/data.py` — market snapshot loading and historical volatility utilities
- `optionpricing/experiment.py` — single-expiration pricing and Greeks experiments
- `optionpricing/evaluation.py` — comparison and error summaries against market prices
- `optionpricing/surface.py` — multi-expiration SABR/Heston calibration and surface assembly
- `optionpricing/plotting.py` — charts for price comparisons, smiles, errors, Greeks, and IV surfaces
- `optionpricing/black_scholes.py` — Black-Scholes pricing and implied vol helpers
- `optionpricing/binomial.py` — binomial option pricing
- `optionpricing/monte_carlo.py` — Monte Carlo pricer
- `optionpricing/sabr.py` — SABR calibration and pricing
- `optionpricing/heston.py` — Heston model pricing and calibration
- `optionpricing/greeks.py` — Greeks utilities
- `tests/test_offline_smoke.py` — offline smoke test using a synthetic option chain

## Requirements

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python main.py
```

By default, `main.py` uses Yahoo Finance data and evaluates the symbol `^SPX` with call options. It performs two major workflows:

1. Single-expiration snapshot benchmark
   - loads one option chain,
   - computes historical volatility,
   - prices each strike under the supported models,
   - compares model values to market prices,
   - prints summary metrics,
   - plots model price comparisons and Greeks.

2. Multi-expiration surface build
   - calibrates SABR and Heston at several maturities,
   - compares calibrated IV surfaces,
   - plots term-structure diagnostics and ATM volatility curves.

## Offline validation

```bash
python tests/test_offline_smoke.py
```


