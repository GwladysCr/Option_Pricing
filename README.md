# Option Pricing Toolkit

## What's new in this pass

- **`optionpricing/heston.py`** — Heston stochastic-volatility model, priced via the Fourier-integral formula. Calibration is vega-weighted and uses a fixed Gauss-Legendre quadrature (128 nodes, cached) instead of adaptive `quad` inside the optimizer loop — reduce time for calibration

- **`optionpricing/greeks.py`** — closed-form Black-Scholes Greeks, plus generic finite-difference Greeks for Heston

- **`optionpricing/experiment.py`** 

- **`optionpricing/data.py`** — 
- **`optionpricing/surface.py`** - calibrates SABR and Heston independently at each of several expirations and builds comparable IV-surface grids (market vs SABR vs Heston)

- **`optionpricing/plotting.py`** — price-vs-strike comparison, volatility smile (market + SABR fit + Heston fit), error summary (RMSE/MAE bars), error-by-strike, Greeks (Delta/Gamma/Theta/Vega BS vs Heston), 3D IV surfaces, and parameter/ATM term structures

- **`tests/test_offline_smoke.py`** — an end-to-end test against a synthetic option chain (built from a known Heston process)

## Running it

```bash
pip install -r requirements.txt
python main.py
```

`main.py` runs two things for AAPL calls:

1. **Multi-strike test** — one expiration, every liquid strike, all six models (Black-Scholes, Monte Carlo, Binomial European/American, SABR, Heston), plus error metrics and a Greeks comparison.
2. **Surface test** — several expirations, SABR and Heston only, producing 3D IV surfaces and term-structure plots.
