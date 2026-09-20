from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class OptionSnapshot:
    ticker: str
    valuation_date: pd.Timestamp
    expiration: pd.Timestamp
    option_type: str
    spot: float
    rate: float
    dividend_yield: float
    data: pd.DataFrame


def historical_volatility(ticker_obj, end_date=None, lookback_days=126):
    kwargs = {"period": "1y"}
    if end_date is not None:
        end = pd.Timestamp(end_date)
        start = end - pd.Timedelta(days=lookback_days * 2)
        kwargs = {"start": start.strftime("%Y-%m-%d"),
                  "end": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}

    hist = ticker_obj.history(**kwargs)["Close"].dropna()
    log_returns = np.log(hist / hist.shift(1)).dropna()
    if len(log_returns) < 30:
        raise ValueError("Not enough observations for volatility estimation")
    return float(log_returns.std(ddof=1) * np.sqrt(252))


def clean_option_chain(df, spot):
    out = df.copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    out["relative_spread"] = out["spread"] / out["mid"].replace(0, np.nan)

    out = out[
        (out["strike"] > 0)
        & (out["bid"] > 0)
        & (out["ask"] >= out["bid"])
        & np.isfinite(out["mid"])
        & (out["relative_spread"] <= 0.25)
        & (out["volume"].fillna(0) > 0)
    ].copy()

    out["log_moneyness"] = np.log(out["strike"] / spot)
    return out.sort_values("strike").reset_index(drop=True)


def get_risk_free_rate(T):
    if T <= 0.25:
        proxy_ticker = "^IRX"
    elif T <= 0.5:
        proxy_ticker = "^FVX"
    elif T <= 2:
        proxy_ticker = "^TNX"
    else:
        proxy_ticker = "^TYX"

    data = yf.Ticker(proxy_ticker).history(period="5d")["Close"].dropna()
    if data.empty:
        return 0.05 
    return float(data.iloc[-1]) / 100


def get_dividend_yield(ticker_obj):
    info = ticker_obj.info
    raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
    return float(raw) / 100


def list_expirations(ticker_obj, min_days=5, max_expirations=12):
    expirations = list(ticker_obj.options)
    selected = []
    now = pd.Timestamp.now(tz=None)
    for exp in expirations:
        days = (pd.Timestamp(exp) - now).days
        if days >= min_days:
            selected.append(exp)
        if len(selected) >= max_expirations:
            break
    return selected


def load_current_snapshot(ticker, expiration_index=5, option_type="call", strike=None,obj=None, expiration=None):
    obj = obj if obj is not None else yf.Ticker(ticker.upper())
    hist = obj.history(period="5d")["Close"].dropna()
    if hist.empty:
        raise ValueError("Could not retrieve spot price")

    spot = float(hist.iloc[-1])

    if expiration is None:
        expirations = list(obj.options)
        if not expirations:
            raise ValueError("No option expirations available")
        expiration = expirations[min(expiration_index, len(expirations) - 1)]

    chain = obj.option_chain(expiration)
    df = chain.calls if option_type == "call" else chain.puts
    df = clean_option_chain(df, spot)

    if df.empty:
        raise ValueError("No liquid options remain after filtering")

    exp = pd.Timestamp(expiration)
    now = pd.Timestamp.now(tz=None)
    T = max((exp - now).total_seconds() / (365.0 * 24 * 3600), 1e-8)

    rate = get_risk_free_rate(T)
    q = get_dividend_yield(obj)

    if strike is not None:
        idx = np.argmin(np.abs(df["strike"].to_numpy() - strike))
    else:
        idx = np.argmin(np.abs(df["strike"].to_numpy() - spot))

    selected = df.iloc[idx]
    return OptionSnapshot(
        ticker = ticker.upper(),
        valuation_date = pd.Timestamp.now().normalize(),
        expiration = exp,
        option_type = option_type,
        spot = spot,
        rate = rate,
        dividend_yield = q,
        data = df,
    ), float(selected["strike"]), T
