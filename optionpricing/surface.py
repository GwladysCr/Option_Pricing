from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from . import data
from .black_scholes import implied_volatility
from . import sabr
from . import heston


@dataclass
class SurfaceResult:
    ticker: str
    option_type: str
    moneyness_grid: np.ndarray      
    maturities: np.ndarray          
    market_iv_surface: np.ndarray   
    sabr_iv_surface: np.ndarray    
    heston_iv_surface: np.ndarray   
    params_df: pd.DataFrame        

def run_surface(ticker, option_type="call", max_expirations=12, min_days=5, calib_band=(0.85, 1.15), moneyness_grid=None):
    """Calibrate SABR and Heston independently at each of several expirations for `ticker`, thenevaluate all three onto common moneyness grid"""
    ticker = ticker.upper()
    obj = yf.Ticker(ticker)

    expirations = data.list_expirations(obj, min_days=min_days, max_expirations=max_expirations)
    if not expirations:
        raise ValueError(f"No expirations at least {min_days} days out for {ticker}")

    moneyness_grid = moneyness_grid if moneyness_grid is not None else np.linspace(0.85, 1.15, 13)

    rows = []
    maturities = []
    market_iv_rows, sabr_iv_rows, heston_iv_rows = [], [], []

    for exp in expirations:
        try:
            snapshot, _, T = data.load_current_snapshot(ticker, option_type=option_type, obj=obj, expiration=exp)
        except ValueError as exc:
            print(f"  [{exp}] skipped: {exc}")
            continue

        S, r, q = snapshot.spot, snapshot.rate, snapshot.dividend_yield
        df = snapshot.data

        lo, hi = calib_band
        band = df[(df["strike"] > lo * S) & (df["strike"] < hi * S)].copy()
        band["iv"] = [implied_volatility(mid, S, K, T, r, option_type, q) for K, mid in zip(band["strike"], band["mid"])]
        band = band[band["iv"].notna() & np.isfinite(band["iv"])]

        if len(band) < 5:
            print(f"  [{exp}] skipped: only {len(band)} invertible strikes")
            continue

        strikes = band["strike"].to_numpy()
        ivs = band["iv"].to_numpy()
        F = S * np.exp((r - q) * T)

        try:
            sabr_params = sabr.calibrate(strikes, ivs, S, T, beta=0.5)
        except Exception as exc:
            print(f"  [{exp}] SABR calibration failed: {exc}")
            continue

        try:
            heston_params = heston.calibrate(strikes, ivs, S, r, q, T, option_type=option_type)
        except Exception as exc:
            print(f"  [{exp}] Heston calibration failed: {exc}")
            continue

        grid_strikes = moneyness_grid * S
        sabr_iv_grid = np.array([sabr.implied_volatility(K, F, T, sabr_params["alpha"], sabr_params["beta"], sabr_params["rho"], sabr_params["nu"]) for K in grid_strikes])
        heston_prices_grid = heston.price_vector(S, grid_strikes, T, r, q, option_type, heston_params)
        heston_iv_grid = np.array([implied_volatility(p, S, K, T, r, option_type, q) for p, K in zip(heston_prices_grid, grid_strikes)])
        
        order = np.argsort(strikes)
        market_iv_grid = np.interp(grid_strikes, strikes[order], ivs[order], left=np.nan, right=np.nan)

        atm_market_iv = float(np.interp(S, strikes[order], ivs[order]))

        maturities.append(T)
        market_iv_rows.append(market_iv_grid)
        sabr_iv_rows.append(sabr_iv_grid)
        heston_iv_rows.append(heston_iv_grid)

        rows.append({
            "expiration": exp, "T": T, "r": r, "q": q, "n_strikes": len(band),
            "sabr_alpha": sabr_params["alpha"], "sabr_beta": sabr_params["beta"],
            "sabr_rho": sabr_params["rho"], "sabr_nu": sabr_params["nu"],
            "sabr_iv_rmse": sabr_params["iv_rmse"],
            "heston_kappa": heston_params["kappa"], "heston_theta": heston_params["theta"],
            "heston_sigma_v": heston_params["sigma_v"], "heston_rho": heston_params["rho"],
            "heston_v0": heston_params["v0"], "heston_fit_err": heston_params["fit_err"],
            "heston_feller_violated": heston_params["feller_violated"],
            "atm_market_iv": atm_market_iv,
        })
        print(f"  [{exp}] T={T:.3f}  n={len(band)}  "
              f"SABR RMSE={sabr_params['iv_rmse']:.4f}  Heston fit_err={heston_params['fit_err']:.2e}")

    if not rows:
        raise ValueError("No expiration could be calibrated -- check ticker/liquidity")

    return SurfaceResult(
        ticker=ticker,
        option_type=option_type,
        moneyness_grid=moneyness_grid,
        maturities=np.array(maturities),
        market_iv_surface=np.array(market_iv_rows),
        sabr_iv_surface=np.array(sabr_iv_rows),
        heston_iv_surface=np.array(heston_iv_rows),
        params_df=pd.DataFrame(rows),
    )
