#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Run from one level above the `optionpricing` package, e.g.:
    python main.py
"""
import yfinance as yf

from optionpricing import data, experiment, evaluation, surface, plotting

TICKER = "^SPX"
OPTION_TYPE = "call"
EXPIRATION_INDEX = 1

if __name__ == "__main__":

    """1 - One expiration, every liquid strike"""
    print("=" * 70)
    print(f"1. MULTI-STRIKE TEST: {TICKER} {OPTION_TYPE}s")
    print("=" * 70)

    obj = yf.Ticker(TICKER)
    snapshot, _, T = data.load_current_snapshot(TICKER, expiration_index=EXPIRATION_INDEX, option_type=OPTION_TYPE, obj=obj)
    hist_vol = data.historical_volatility(obj)

    print(f"{TICKER}: spot=${snapshot.spot:.2f}  expiration={snapshot.expiration.date()} "
          f"(T={T:.3f}y)  r={snapshot.rate:.4f}  q={snapshot.dividend_yield:.4f}  "
          f"hist_vol={hist_vol:.4f}  n_strikes={len(snapshot.data)}")

    results_df, calibration_params = experiment.run_single_snapshot(snapshot.data, snapshot.spot, T, snapshot.rate, snapshot.dividend_yield, option_type=OPTION_TYPE, hist_vol=hist_vol)

    results_df = evaluation.add_error_columns(results_df)
    summary_df = evaluation.summarize(results_df)
    print("\nPer-model error summary:")
    print(summary_df.round(4).to_string(index=False))

    greeks_df = experiment.run_greeks_table(snapshot.data, snapshot.spot, T, snapshot.rate, snapshot.dividend_yield, option_type=OPTION_TYPE, hist_vol=hist_vol, heston_params=calibration_params["heston"])

    plotting.plot_price_comparison(results_df, S=snapshot.spot)
    plotting.plot_volatility_smile(results_df, calibration_params, snapshot.spot, snapshot.rate, snapshot.dividend_yield, T, option_type=OPTION_TYPE)
    plotting.plot_error_summary(summary_df)
    plotting.plot_error_by_strike(results_df)
    plotting.plot_greeks(greeks_df, S=snapshot.spot)

    """2 - Several expirations, every strike (SABR & Heston)"""
    print("\n" + "=" * 70)
    print(f"2. SURFACE TEST: {TICKER} {OPTION_TYPE}s")
    print("=" * 70)

    surface_result = surface.run_surface(TICKER, option_type=OPTION_TYPE, max_expirations=6)
    print("\nPer-expiration calibration summary:")
    print(surface_result.params_df.round(4).to_string(index=False))

    plotting.plot_iv_surfaces(surface_result)
    plotting.plot_param_term_structure(surface_result)
    plotting.plot_atm_term_structure(surface_result)
