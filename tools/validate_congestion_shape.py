"""Compare the hourly SHAPE of NYC DOT measured speeds with the parametric model.
Shape only (Pearson/Spearman on 24-point profiles); absolute values are not compared."""
import argparse, json
import numpy as np, pandas as pd, requests
from scipy.stats import pearsonr, spearmanr

API = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"

def fetch(start, end, borough):
    params = {
        "$select": "date_trunc_ymd(data_as_of) as d, date_extract_hh(data_as_of) as h, avg(speed) as v, count(*) as n",
        "$where": f"data_as_of >= '{start}T00:00:00' AND data_as_of < '{end}T00:00:00' AND borough = '{borough}'",
        "$group": "d, h", "$limit": 50000,
    }
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise SystemExit("API returned no rows; check dates and the borough column name.")
    df["d"] = pd.to_datetime(df["d"]); df["h"] = df["h"].astype(int)
    df["v"] = df["v"].astype(float); df["n"] = df["n"].astype(int)
    return df

def fetch_csv(path, start, end, borough):
    parts = []
    for ch in pd.read_csv(path, usecols=["data_as_of", "speed", "borough"], chunksize=1_000_000):
        ch = ch[ch["borough"] == borough]
        t = pd.to_datetime(ch["data_as_of"], errors="coerce")
        ch = ch[(t >= start) & (t < end)].assign(t=t)
        ch["d"] = ch["t"].dt.normalize(); ch["h"] = ch["t"].dt.hour
        parts.append(ch.groupby(["d", "h"])["speed"].agg(["sum", "count"]))
    g = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    return pd.DataFrame({"d": g["d"], "h": g["h"], "v": g["sum"] / g["count"], "n": g["count"]})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--borough", default="Manhattan")
    ap.add_argument("--csv", default=None, help="local CSV with data_as_of,speed,borough (skips the API)")
    ap.add_argument("--model-json", required=True,
                    help="JSON list of 24 model values: mean speed per hour (NOT cost)")
    ap.add_argument("--out", default="results/congestion_shape_validation.json")
    a = ap.parse_args()
    model = np.asarray(json.load(open(a.model_json)), float)
    assert model.shape == (24,), "model profile must have 24 values"
    df = fetch_csv(a.csv, a.start, a.end, a.borough) if a.csv else fetch(a.start, a.end, a.borough)
    df = df[df["d"].dt.dayofweek < 5]
    piv = df.pivot_table(index="d", columns="h", values="v").reindex(columns=range(24))
    piv = piv.dropna(thresh=22)
    if len(piv) < 5:
        raise SystemExit(f"only {len(piv)} usable weekdays; widen the date range")
    prof = piv.mean(axis=0).interpolate(limit_direction="both").to_numpy()
    r, p = pearsonr(prof, model); rho, _ = spearmanr(prof, model)
    rng = np.random.default_rng(0); days = piv.to_numpy(); boots = []
    for _ in range(2000):
        s = days[rng.integers(0, len(days), len(days))]
        bp = pd.DataFrame(s).mean(axis=0).interpolate(limit_direction="both").to_numpy()
        boots.append(pearsonr(bp, model)[0])
    out = dict(city=a.borough, n_weekdays=int(len(piv)), pearson=float(r), p=float(p),
               spearman=float(rho),
               pearson_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               measured_profile=prof.tolist(), model_profile=model.tolist(),
               scope="shape only; 1 of 12 cities; sensors mostly arterials/highways")
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n_weekdays", "pearson", "pearson_ci95", "spearman")}, indent=2))

if __name__ == "__main__":
    main()
