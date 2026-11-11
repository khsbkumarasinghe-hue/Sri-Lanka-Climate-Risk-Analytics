"""Direct multi-horizon (1–7 day) gradient-boosting forecaster, one model per variable and horizon.
Hold-out: the last 365 days. Final models are refit on all data. Uncertainty = hold-out residual 10/90th percentiles."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR

VARS = ["temp_mean", "temp_min", "temp_max", "rainfall_mm", "humidity_mean", "solar_radiation_mean"]
H = 7


def features(d):
    X = pd.DataFrame(index=d.index)
    for v in VARS:
        for l in (0, 1, 2, 3, 7):
            X[f"{v}_l{l}"] = d[v].shift(l)
        X[f"{v}_m7"] = d[v].rolling(7).mean()
        X[f"{v}_m30"] = d[v].rolling(30).mean()
    X["api"], X["dry_spell"] = d.api, d.dry_spell
    X["doy_sin"] = np.sin(2 * np.pi * d.doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * d.doy / 365.25)
    return X


def _reg():
    return HGR(max_iter=120, learning_rate=0.06, max_depth=4, random_state=0)


def run_forecast(d):
    """d: one district's daily frame sorted by date. Returns (forecast_df, metrics_df)."""
    d = d.sort_values("date").reset_index(drop=True)
    X, dates = features(d), d.date
    cut = dates.max() - pd.Timedelta(days=365)
    last = X.iloc[[-1]]
    rows = {h: {"date": dates.max() + pd.Timedelta(days=h)} for h in range(1, H + 1)}
    met = []
    for v in VARS:
        log = v == "rainfall_mm"
        clim = d[dates <= cut].groupby("month")[v].mean()
        for h in range(1, H + 1):
            y_raw = d[v].shift(-h)
            y = np.log1p(y_raw) if log else y_raw
            ok = X.notna().all(axis=1) & y.notna()
            tr, te = ok & (dates + pd.Timedelta(days=h) <= cut), ok & (dates > cut)
            inv = (lambda a: np.expm1(np.clip(a, 0, None))) if log else (lambda a: a)
            p = inv(_reg().fit(X[tr], y[tr]).predict(X[te]))
            act = y_raw[te].values
            res = act - p
            base = pd.Series((dates[te] + pd.Timedelta(days=h)).dt.month.values).map(clim).values
            met.append(dict(variable=v, horizon=h, mae_model=np.abs(res).mean(), mae_climatology=np.abs(act - base).mean()))
            pred = inv(_reg().fit(X[ok], y[ok]).predict(last))[0]
            q10, q90 = np.percentile(res, [10, 90])
            rows[h][v], rows[h][v + "_lo"], rows[h][v + "_hi"] = pred, pred + q10, pred + q90
    for h in range(1, H + 1):  # probability of a rainy day (>= 1 mm)
        yc = (d.rainfall_mm.shift(-h) >= 1).astype(int)
        ok = X.notna().all(axis=1) & d.rainfall_mm.shift(-h).notna()
        rows[h]["rain_prob"] = HGC(max_iter=100, learning_rate=0.06, max_depth=3, random_state=0).fit(X[ok], yc[ok]).predict_proba(last)[0, 1]
    f = pd.DataFrame(rows.values())
    f["rainfall_mm_lo"] = f.rainfall_mm_lo.clip(lower=0)
    f["humidity_mean"] = f.humidity_mean.clip(upper=100)
    for s in ("", "_lo", "_hi"):  # keep min <= mean <= max
        f["temp_min" + s] = np.minimum(f["temp_min" + s], f["temp_mean" + s])
        f["temp_max" + s] = np.maximum(f["temp_max" + s], f["temp_mean" + s])
    return f, pd.DataFrame(met)
