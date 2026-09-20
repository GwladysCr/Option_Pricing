import numpy as np
import pandas as pd


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    denom = np.where(np.abs(actual) > 1e-12, np.abs(actual), np.nan)

    return {
        "MAE": float(np.nanmean(np.abs(err))),
        "RMSE": float(np.sqrt(np.nanmean(err**2))),
        "Mean_Error": float(np.nanmean(err)),
        "Mean_Relative_Error": float(np.nanmean(np.abs(err) / denom))
    }


def summarize(results):
    rows = []
    for model, group in results.groupby("model"):
        m = metrics(group["market_price"], group["model_price"])
        rows.append({"model": model, **m, "n": len(group)})
    return pd.DataFrame(rows).sort_values("RMSE")


def add_error_columns(df):
    out = df.copy()
    out["error"] = out["model_price"] - out["market_price"]
    out["abs_error"] = out["error"].abs()
    out["relative_error"] = out["abs_error"] / out["market_price"].abs()
    return out
