#!/usr/bin/env python3
"""Let tools/validate_congestion_shape.py read a small pre-aggregated CSV (columns d,h,v,n) downloaded in the browser."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "tools", "validate_congestion_shape.py")
s = open(p, encoding="utf-8").read()
if "pre-aggregated" in s:
    print("already patched"); raise SystemExit
anchor = '    """Aggregate a (possibly huge) local CSV to per-day, per-hour mean speed."""\n    parts = []\n'
assert anchor in s, "anchor not found"
new = ('    """Aggregate a (possibly huge) local CSV to per-day, per-hour mean speed. Also accepts a pre-aggregated CSV with columns d,h,v,n."""\n'
       '    hdr = [c.strip().lower() for c in pd.read_csv(path, nrows=0).columns]\n'
       '    if {"d", "h", "v", "n"} <= set(hdr):   # pre-aggregated\n'
       '        df = pd.read_csv(path)\n'
       '        df.columns = [c.strip().lower() for c in df.columns]\n'
       '        df["d"] = pd.to_datetime(df["d"])\n'
       '        df["h"] = df["h"].astype(int)\n'
       '        df["v"] = df["v"].astype(float)\n'
       '        df["n"] = df["n"].astype(int)\n'
       '        return df[["d", "h", "v", "n"]]\n'
       '    parts = []\n')
open(p, "w", encoding="utf-8").write(s.replace(anchor, new, 1))
print("patched")
