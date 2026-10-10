"""Compare the hourly SHAPE of measured NYC DOT speeds with the parametric congestion model.

Shape only (Pearson/Spearman on 24-point profiles); absolute speeds are not compared.
Data source: NYC DOT Traffic Speeds (NYC OpenData, dataset i4gi-tjb9), via the SODA API
or a local CSV export (--csv) with columns data_as_of, speed, borough.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

API = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"
MIN_HOURS = 22          # a day needs >= this many of 24 hourly means to be used


def fetch(start, end, borough):
    import requests
    params = {
        "$select": "date_trunc_ymd(data_as_of) as d, date_extract_hh(data_as_of) as h, "
                   "avg(speed) as v, count(*) as n",
        "$where": f"data_as_of >= '{start}T00:00:00' AND data_as_of < '{end}T00:00:00' "
                  f"AND borough = '{borough}'",
        "$group": "d, h", "$limit": 50000,
    }
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise SystemExit("API returned no rows; check dates and the borough column name.")
    df["d"] = pd.to_datetime(df["d"])
    df["h"] = df["h"].astype(int)
    df["v"] = df["v"].astype(float)
    df["n"] = df["n"].astype(int)
    return df[["d", "h", "v", "n"]]


def fetch_csv(path, start, end, borough, chunksize=1_000_000):
    """Aggregate a (possibly huge) local CSV to per-day, per-hour mean speed."""
    parts = []
    for ch in pd.read_csv(path, usecols=["data_as_of", "speed", "borough"], chunksize=chunksize):
        ch = ch[ch["borough"] == borough].copy()
        ch["t"] = pd.to_datetime(ch["data_as_of"], errors="coerce")
        ch["speed"] = pd.to_numeric(ch["speed"], errors="coerce")
        ch = ch.dropna(subset=["t", "speed"])
        ch = ch[(ch["t"] >= pd.Timestamp(start)) & (ch["t"] < pd.Timestamp(end))]
        if ch.empty:
            continue
        ch["d"] = ch["t"].dt.normalize()
        ch["h"] = ch["t"].dt.hour
        parts.append(ch.groupby(["d", "h"])["speed"].agg(["sum", "count"]))
    if not parts:
        raise SystemExit("CSV contains no rows for this borough/date range.")
    g = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    return pd.DataFrame({"d": g["d"], "h": g["h"].astype(int),
                         "v": g["sum"] / g["count"], "n": g["count"].astype(int)})


def daily_matrix(df, min_hours=MIN_HOURS):
    """(day x 24) matrix of hourly mean speeds: weekdays only, near-complete days only."""
    df = df[df["d"].dt.dayofweek < 5]
    piv = df.pivot_table(index="d", columns="h", values="v").reindex(columns=range(24))
    return piv.dropna(thresh=min_hours)


def _profile(days):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # an hour missing on every day
        m = np.nanmean(days, axis=0)
    return pd.Series(m).interpolate(limit_direction="both").to_numpy()


def compare_shape(days, model, n_boot=2000, seed=0):
    """Pearson/Spearman between the mean measured 24-h profile and the model profile,
    with a bootstrap-over-days 95% CI for Pearson."""
    days = np.asarray(days, float)
    model = np.asarray(model, float)
    if model.shape != (24,) or not np.isfinite(model).all():
        raise ValueError("model profile must be 24 finite values")
    if days.ndim != 2 or days.shape[1] != 24 or days.shape[0] < 5:
        raise ValueError("need a (n_days>=5, 24) matrix of hourly speeds")
    if np.ptp(model) < 1e-12:
        raise ValueError("model profile is constant; shape correlation is undefined")
    prof = _profile(days)
    if np.ptp(prof) < 1e-12:
        raise ValueError("measured profile is constant; shape correlation is undefined")
    r, p = pearsonr(prof, model)
    rho, _ = spearmanr(prof, model)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bp = _profile(days[rng.integers(0, len(days), len(days))])
        if np.ptp(bp) > 1e-12:
            boots.append(pearsonr(bp, model)[0])
    return {"n_weekdays": int(len(days)), "pearson": float(r), "p": float(p),
            "spearman": float(rho),
            "pearson_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "measured_profile": prof.tolist(), "model_profile": model.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--borough", default="Manhattan")
    ap.add_argument("--csv", default=None, help="local CSV (data_as_of,speed,borough); skips the API")
    ap.add_argument("--model-json", required=True,
                    help="JSON list of 24 model values: mean speed per hour (NOT cost)")
    ap.add_argument("--out", default="results/congestion_shape_validation.json")
    a = ap.parse_args()
    model = json.load(open(a.model_json))
    df = fetch_csv(a.csv, a.start, a.end, a.borough) if a.csv else fetch(a.start, a.end, a.borough)
    piv = daily_matrix(df)
    if len(piv) < 5:
        raise SystemExit(f"only {len(piv)} usable weekdays; widen the date range")
    out = compare_shape(piv.to_numpy(), model)
    out.update(city=a.borough, start=a.start, end=a.end, source="csv" if a.csv else "soda-api",
               scope="shape only; 1 of 12 cities; sensors mostly arterials/highways")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n_weekdays", "pearson", "pearson_ci95", "spearman")}, indent=2))


if __name__ == "__main__":
    main()
