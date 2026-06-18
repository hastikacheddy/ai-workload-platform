"""
Out-of-sample evaluation of the forecaster on the committed demand aggregates.
Trains on the first 80% (chronological), forecasts the held-out tail, reports
MAE / MAPE / 99% VaR coverage, and saves a forecast-vs-actual chart.

    python scripts/evaluate.py
"""
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from src.common.config import load_config, get_hparams                # noqa: E402
from src.forecasting.forecaster import build_training_frame           # noqa: E402
from src.inference.risk import compute_garch_sigma, monte_carlo_var   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(granularity, hparams):
    name = "daily" if granularity == "D" else "hourly"
    df = pd.read_csv(os.path.join(ROOT, "data", f"{name}_demand.csv"), parse_dates=["TimePeriod"])
    X, y = build_training_frame(df, granularity)
    split = int(len(X) * 0.8)
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1, **hparams)
    model.fit(X.iloc[:split], y.iloc[:split])

    idx = X.index[split:]
    actual = y.iloc[split:].to_numpy()
    pred = np.clip(model.predict(X.iloc[split:]), 0, None)
    resid = actual - pred
    sigma = compute_garch_sigma(resid, fallback=float(np.std(resid)))
    upper = np.array([monte_carlo_var(p, sigma, 0.99) for p in pred])

    mae = float(np.mean(np.abs(resid)))
    mape = float(np.mean(np.abs(resid) / np.clip(actual, 1, None)) * 100)
    coverage = float((actual <= upper).mean() * 100)
    return dict(name=name, idx=idx, actual=actual, pred=pred, upper=upper,
                mae=mae, mape=mape, coverage=coverage, n=len(actual))


def main():
    cfg = load_config()
    hp = get_hparams(cfg, "forecast")
    daily = evaluate("D", hp)
    hourly = evaluate("H", hp)

    for r in (daily, hourly):
        print(f"{r['name']:>6}: MAE={r['mae']:.0f}  MAPE={r['mape']:.1f}%  "
              f"99%-coverage={r['coverage']:.1f}%  (n={r['n']})")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    ax1.plot(daily["idx"], daily["actual"], label="actual", color="#1f77b4")
    ax1.plot(daily["idx"], daily["pred"], "--", label="forecast", color="#2ca02c")
    ax1.fill_between(daily["idx"], daily["pred"], daily["upper"], color="#ff7f0e", alpha=0.15,
                     label="99% VaR band")
    ax1.set_title(f"Daily trip demand — forecast vs actual  "
                  f"(MAE {daily['mae']:.0f}, MAPE {daily['mape']:.1f}%, "
                  f"99% coverage {daily['coverage']:.0f}%)")
    ax1.legend(loc="upper left", fontsize=8)

    h = {k: (v[-168:] if k in ("idx", "actual", "pred", "upper") else v) for k, v in hourly.items()}
    ax2.plot(h["idx"], h["actual"], label="actual", color="#1f77b4")
    ax2.plot(h["idx"], h["pred"], "--", label="forecast", color="#2ca02c")
    ax2.set_title(f"Hourly trip demand — last 7 days  "
                  f"(MAE {hourly['mae']:.0f}, MAPE {hourly['mape']:.1f}%, "
                  f"99% coverage {hourly['coverage']:.0f}%)")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "forecast_vs_actual.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
