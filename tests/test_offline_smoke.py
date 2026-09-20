from datetime import datetime, timedelta
import sys
import os
import types
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from optionpricing import heston as heston_mod

np.random.seed(0)

S0 = 150.0
TRUE_HESTON = dict(kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.6, v0=0.045)
R_TRUE, Q_TRUE = 0.045, 0.005


def make_expirations(n=6):
    today = datetime.now()
    days_out = [10, 24, 45, 80, 140, 230, 320][:n]
    return [(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in days_out]


def make_chain_df(T, option_type="call"):
    strikes = np.arange(0.7, 1.31, 0.025) * S0
    params = TRUE_HESTON
    prices = heston_mod.price_vector(S0, strikes, T, R_TRUE, Q_TRUE, "call", params)
    if option_type == "put":
        prices = prices - S0 * np.exp(-Q_TRUE * T) + strikes * np.exp(-R_TRUE * T)

    spread = np.maximum(prices * 0.03, 0.01)
    bid = np.maximum(prices - spread / 2, 0.0)
    ask = prices + spread / 2

    from optionpricing.black_scholes import implied_volatility
    ivs = np.array([implied_volatility((b + a) / 2, S0, k, T, R_TRUE, option_type, Q_TRUE)
                     for b, a, k in zip(bid, ask, strikes)])
    ivs = np.nan_to_num(ivs, nan=0.3)

    volume = np.random.randint(10, 500, size=len(strikes))
    return pd.DataFrame({"strike": strikes, "bid": bid, "ask": ask,
                          "impliedVolatility": ivs, "volume": volume})


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self._expirations = make_expirations()

    def history(self, period="5d", start=None, end=None):
        if self.symbol.startswith("^"):
            return pd.DataFrame({"Close": [4.5] * 5})
        if period == "1y" or start is not None:
            n = 300
            rets = np.random.normal(0, 0.015, n)
            prices = S0 * np.exp(np.cumsum(rets))
            return pd.DataFrame({"Close": prices})
        return pd.DataFrame({"Close": [S0] * 5})

    @property
    def options(self):
        return self._expirations

    def option_chain(self, expiration):
        T = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days / 365.0
        calls_df = make_chain_df(T, "call")
        puts_df = make_chain_df(T, "put")
        return types.SimpleNamespace(calls=calls_df, puts=puts_df)

    @property
    def info(self):
        return {"dividendYield": Q_TRUE * 100}


def run_tests():
    with patch("yfinance.Ticker", FakeTicker):
        print("### Testing data.py ###")
        import yfinance as yf
        from optionpricing import data
        obj = yf.Ticker("TEST")
        snapshot, strike, T = data.load_current_snapshot("TEST", expiration_index=1, option_type="call", obj=obj)
        hist_vol = data.historical_volatility(obj)
        print(f"spot={snapshot.spot:.2f} rate={snapshot.rate:.4f} q={snapshot.dividend_yield:.4f} "
              f"T={T:.3f} hist_vol={hist_vol:.4f} n_rows={len(snapshot.data)}")
        assert not np.isnan(snapshot.rate), "rate should no longer be NaN"
        assert snapshot.dividend_yield > 0, "dividend yield should be picked up from info"
        print("OK: data.py\n")

        print("### Testing experiment.py (run_single_snapshot + run_greeks_table) ###")
        from optionpricing import experiment, evaluation
        results_df, calib = experiment.run_single_snapshot(
            snapshot.data, snapshot.spot, T, snapshot.rate, snapshot.dividend_yield,
            option_type="call", hist_vol=hist_vol, mc_simulations=20_000)
        print(results_df["model"].value_counts())
        assert calib["sabr"] is not None and calib["heston"] is not None
        results_df = evaluation.add_error_columns(results_df)
        summary = evaluation.summarize(results_df)
        print(summary.round(4).to_string(index=False))

        greeks_df = experiment.run_greeks_table(
            snapshot.data, snapshot.spot, T, snapshot.rate, snapshot.dividend_yield,
            option_type="call", hist_vol=hist_vol, heston_params=calib["heston"])
        print(greeks_df.round(4).head().to_string(index=False))
        print("OK: experiment.py\n")

        print("### Testing plotting.py ###")
        from optionpricing import plotting
        plotting.plot_price_comparison(results_df, S=snapshot.spot)
        plotting.plot_volatility_smile(results_df, calib, snapshot.spot, snapshot.rate,
                                        snapshot.dividend_yield, T, option_type="call")
        plotting.plot_error_summary(summary)
        plotting.plot_error_by_strike(results_df)
        plotting.plot_greeks(greeks_df, S=snapshot.spot)
        print("OK: plotting.py (multi-strike plots)\n")

        print("### Testing surface.py ###")
        from optionpricing import surface
        surface_result = surface.run_surface("TEST", option_type="call", max_expirations=4)
        print(surface_result.params_df.round(4).to_string(index=False))
        plotting.plot_iv_surfaces(surface_result)
        plotting.plot_param_term_structure(surface_result)
        plotting.plot_atm_term_structure(surface_result)
        print("OK: surface.py + surface plots\n")

    print("ALL MOCK E2E TESTS PASSED")


if __name__ == "__main__":
    run_tests()
