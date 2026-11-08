# -*- coding: utf-8 -*-
"""Shared helpers for the analyses added in README sections 5l-5r (no optional dependencies)."""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_spanner_benchmark as core  # noqa: E402


def md_table(df: pd.DataFrame, nd: int = 2) -> str:
    """GitHub markdown table without the optional `tabulate` package."""
    def cell(x):
        if isinstance(x, float):
            return "" if x != x else f"{x:.{nd}f}"
        return str(x)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def purge_synthetic_cache() -> int:
    n = 0
    for m in glob.glob(os.path.join(ROOT, "results", "graph_cache", "*.meta.json")):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        if d.get("source") == "synthetic":
            os.remove(m)
            pkl = m.replace(".meta.json", ".pkl")
            if os.path.exists(pkl):
                os.remove(pkl)
            n += 1
    return n


def load_city(spec: Dict, grid_n: int = 15, synthetic: bool = False) -> Optional["core.nx.DiGraph"]:
    """City network from the cache or OpenStreetMap.  Returns None (and removes the synthetic fallback that the
    core loader would otherwise cache under the real-city key) when real data cannot be obtained and
    `synthetic` was not requested, so a real-data result file never contains synthetic rows."""
    G = core.build_city_network(spec, grid_n=grid_n, force_synthetic=synthetic)
    if not synthetic and not G.graph.get("used_real_osm"):
        purge_synthetic_cache()
        print(f"[skip] {spec['name']}: real OpenStreetMap data unavailable")
        return None
    return G
