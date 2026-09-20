import time
import numpy as np
import pandas as pd

from .black_scholes import price as bs_price, implied_volatility
from .monte_carlo import price as mc_price
from .binomial import price as binomial_price
from .sabr import calibrate as calibrate_sabr, price as sabr_price
from .heston import calibrate as calibrate_heston, price as heston_price
from . import greeks as greeks_mod


def run_single_snapshot(df, S, T, r, q, option_type="call", hist_vol=None, mc_simulations=100_000, binomial_steps=200):
    rows = []
    liquid = df.copy()

    if hist_vol is None:
        raise ValueError("hist_vol is required for historical-volatility models")

    iv_df = liquid.copy()
    iv_df["market_iv_calc"] = [implied_volatility(mid, S, K, T, r, option_type, q)for K, mid in zip(iv_df["strike"], iv_df["mid"])]
    valid = iv_df["market_iv_calc"].notna() & np.isfinite(iv_df["market_iv_calc"])

    F = S * np.exp((r - q) * T)
    sabr_params = None
    heston_params = None

    if valid.sum() >= 5:
        strikes_for_calib = iv_df.loc[valid, "strike"].to_numpy()
        ivs_for_calib = iv_df.loc[valid, "market_iv_calc"].to_numpy()

        try:
            sabr_params = calibrate_sabr(strikes_for_calib, ivs_for_calib, F, T)
        except Exception as exc:
            print(f"  [warn] SABR calibration failed: {exc}")

        try:
            heston_params = calibrate_heston(strikes_for_calib, ivs_for_calib, S, r, q, T, option_type=option_type)
        except Exception as exc:
            print(f"  [warn] Heston calibration failed: {exc}")

    for _, row in liquid.iterrows():
        K, market = float(row["strike"]), float(row["mid"])
        experiments = []

        start = time.perf_counter()
        experiments.append(("BS_HistoricalVol", bs_price(S, K, T, r, hist_vol, option_type, q), time.perf_counter() - start))

        start = time.perf_counter()
        mc = mc_price(S, K, T, r, hist_vol, option_type, q, mc_simulations)
        experiments.append(("MonteCarlo", mc["price"], time.perf_counter() - start))

        start = time.perf_counter()
        experiments.append(("Binomial_European", binomial_price(S, K, T, r, hist_vol, option_type, q, binomial_steps, american=False), time.perf_counter() - start))

        if sabr_params is not None:
            start = time.perf_counter()
            experiments.append(("SABR", sabr_price(S, K, T, r, q, option_type, sabr_params), time.perf_counter() - start))

        if heston_params is not None:
            start = time.perf_counter()
            experiments.append(("Heston", heston_price(S, K, T, r, q, option_type, heston_params), time.perf_counter() - start))

        for model, model_price, runtime in experiments:
            rows.append({
                "strike": K,
                "spot": S,
                "T": T,
                "log_moneyness": np.log(K / (S * np.exp((r - q) * T))),
                "market_price": market,
                "market_iv": row.get("impliedVolatility", np.nan),
                "model": model,
                "model_price": model_price,
                "runtime_seconds": runtime,
            })
    calibration_params = {"sabr": sabr_params, "heston": heston_params}
    return pd.DataFrame(rows), calibration_params


def run_greeks_table(df, S, T, r, q, option_type="call", hist_vol=None, heston_params=None, binomial_steps=150):
    if hist_vol is None:
        raise ValueError("hist_vol is required")

    rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])

        bs_g = greeks_mod.bs_greeks(S, K, T, r, hist_vol, option_type, q)

        record = {
            "strike": K,
            "BS_Delta": bs_g["Delta"], "BS_Gamma": bs_g["Gamma"],
            "BS_Theta": bs_g["Theta"], "BS_Vega": bs_g["Vega"], "BS_Rho": bs_g["Rho"],
        }

        if heston_params is not None:
            from . import heston as heston_mod
            h_g = heston_mod.fd_greeks(heston_params, S, K, T, r, q, option_type)
            record.update({
                "Heston_Delta": h_g["Delta"], "Heston_Gamma": h_g["Gamma"],
                "Heston_Theta": h_g["Theta"], "Heston_Vega": h_g["Vega"],
            })

        rows.append(record)

    return pd.DataFrame(rows)
