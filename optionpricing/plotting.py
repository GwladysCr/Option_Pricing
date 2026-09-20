import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed to register 3d projection)

from . import sabr as sabr_mod
from . import heston as heston_mod
from .black_scholes import implied_volatility

MODEL_COLORS = {
    "BS_HistoricalVol": "blue",
    "MonteCarlo": "green",
    "Binomial_European": "orange",
    "SABR": "red",
    "Heston": "purple",
}


def plot_price_comparison(results_df, S=None):
    """Model price vs strike, overlaid with market prices, for one expiration"""
    fig, ax = plt.subplots(figsize=(12, 6))

    market = results_df.drop_duplicates("strike")[["strike", "market_price"]].sort_values("strike")
    ax.scatter(market["strike"], market["market_price"], color="black", s=20, zorder=5, label="Market")

    for model, group in results_df.groupby("model"):
        group = group.sort_values("strike")
        ax.plot(group["strike"], group["model_price"], "-", color=MODEL_COLORS.get(model), label=model, alpha = 0.55, linewidth=2)

    if S is not None:
        ax.axvline(S, color="gray", linestyle=":", label=f"Spot (${S:.2f})")

    ax.set_xlabel("Strike Price ($)")
    ax.set_ylabel("Option Price ($)")
    ax.set_title("Model Prices Across Strike Chain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_volatility_smile(results_df, calibration_params, S, r, q, T, option_type="call"):
    """Market IV vs strike: the calibrated SABR/Heston smile fits"""
    market = results_df.drop_duplicates("strike")[["strike", "market_iv"]].dropna().sort_values("strike")
    strikes = market["strike"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(strikes, market["market_iv"], "o-", color="blue", label="Market IV")

    sabr_params = calibration_params.get("sabr")
    if sabr_params is not None:
        F = S * np.exp((r - q) * T)
        sabr_ivs = [sabr_mod.implied_volatility(K, F, T, sabr_params["alpha"], sabr_params["beta"], sabr_params["rho"], sabr_params["nu"]) for K in strikes]
        ax.plot(strikes, sabr_ivs, "--", color="red", label="SABR fit")

    heston_params = calibration_params.get("heston")
    if heston_params is not None:
        heston_prices = heston_mod.price_vector(S, strikes, T, r, q, option_type, heston_params)
        heston_ivs = [implied_volatility(p, S, K, T, r, option_type, q) for p, K in zip(heston_prices, strikes)]
        ax.plot(strikes, heston_ivs, "--", color="blue", label="Heston fit")

    ax.axvline(S, color="gray", linestyle=":", label=f"Spot (${S:.2f})")
    ax.set_xlabel("Strike Price ($)")
    ax.set_ylabel("Implied Volatility")
    ax.set_title("Volatility Smile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_summary(summary_df):
    """Bar chart of RMSE and MAE per model"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["RMSE"], width, label="RMSE", color=[MODEL_COLORS.get(m, "gray") for m in summary_df["model"]])
    ax.bar(x + width / 2, summary_df["MAE"], width, label="MAE", color=[MODEL_COLORS.get(m, "gray") for m in summary_df["model"]], alpha = 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=20)
    ax.set_ylabel("Error vs Market Price ($)")
    ax.set_title("Pricing Error Summary")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_error_by_strike(results_df):
    """Model price - market price, per strike"""
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, group in results_df.groupby("model"):
        group = group.sort_values("strike")
        ax.plot(group["strike"], group["model_price"] - group["market_price"], label=model, color=MODEL_COLORS.get(model))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Strike Price ($)")
    ax.set_ylabel("Model Price - Market Price ($)")
    ax.set_title("Pricing Error by Strike")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_greeks(greeks_df, S=None):
    """Delta/Gamma/Theta/Vega vs strike, Black-Scholes vs Heston (if available)"""
    panels = [("Delta", "BS_Delta", "Heston_Delta"),
              ("Gamma", "BS_Gamma", "Heston_Gamma"),
              ("Theta", "BS_Theta", "Heston_Theta"),
              ("Vega", "BS_Vega", "Heston_Vega")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (label, bs_col, he_col) in zip(axes.flat, panels):
        ax.plot(greeks_df["strike"], greeks_df[bs_col], label="Black-Scholes", color="blue")
        if he_col in greeks_df.columns:
            ax.plot(greeks_df["strike"], greeks_df[he_col], label="Heston", color="purple", linestyle=":")
        if S is not None:
            ax.axvline(S, color="gray", linestyle=":")
        ax.set_xlabel("Strike Price ($)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Greeks Across Strikes")
    plt.tight_layout()
    plt.show()


def plot_iv_surfaces(surface_result):
    """3D IV surfaces: market, SABR-fitted, Heston-fitted"""
    sr = surface_result
    X, Y = np.meshgrid(sr.moneyness_grid, sr.maturities)

    fig = plt.figure(figsize=(19, 6))
    specs = [(sr.market_iv_surface, f"{sr.ticker} Market IV Surface", "viridis"),
             (sr.sabr_iv_surface, "SABR-Fitted Surface", "plasma"),
             (sr.heston_iv_surface, "Heston-Fitted Surface", "inferno")]

    for i, (Z, title, cmap) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="k")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Maturity (yrs)")
        ax.set_zlabel("Implied Vol")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_param_term_structure(surface_result):
    """Calibrated SABR and Heston parameters as a function of maturity"""
    df = surface_result.params_df
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    sabr_specs = [("sabr_alpha", "SABR alpha"), ("sabr_rho", "SABR rho"), ("sabr_nu", "SABR nu"), ("sabr_iv_rmse", "SABR IV RMSE")]
    heston_specs = [("heston_kappa", "Heston kappa"), ("heston_theta", "Heston theta"), ("heston_sigma_v", "Heston sigma_v"), ("heston_rho", "Heston rho")]

    for ax, (col, title) in zip(axes[0], sabr_specs):
        ax.plot(df["T"], df[col], "o-", color="red")
        ax.set_xlabel("Maturity (yrs)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for ax, (col, title) in zip(axes[1], heston_specs):
        ax.plot(df["T"], df[col], "o-", color="blue")
        ax.set_xlabel("Maturity (yrs)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{surface_result.ticker} - Calibrated Parameter Term Structure")
    plt.tight_layout()
    plt.show()


def plot_atm_term_structure(surface_result):
    """ATM implied vol vs maturity: market vs SABR fit vs Heston fit"""
    sr = surface_result
    df = sr.params_df
    atm_idx = int(np.argmin(np.abs(sr.moneyness_grid - 1.0)))
    sabr_atm = sr.sabr_iv_surface[:, atm_idx]
    heston_atm = sr.heston_iv_surface[:, atm_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["T"], df["atm_market_iv"], "o-", color="black", linewidth=2, label="Market ATM IV")
    ax.plot(df["T"], sabr_atm, "s--", color="red", label="SABR ATM IV")
    ax.plot(df["T"], heston_atm, "^--", color="blue", label="Heston ATM IV")
    ax.set_xlabel("Maturity (yrs)")
    ax.set_ylabel("ATM Implied Volatility")
    ax.set_title(f"{sr.ticker} - ATM Volatility Term Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
