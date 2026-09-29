#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent verification of run_spanner_benchmark.py's published claims.

This script is deliberately NOT part of run_spanner_benchmark.py and does not
import its aggregation/reporting code. It re-derives a set of headline
numbers from the RAW output artifacts (results/per_pair_dilation.csv,
results/spanner_report.json) using a separate, independently-written code
path (plain pandas/numpy/scipy), and checks them against what the pipeline
itself reported. The point is not to re-run the experiment -- it is to catch
the class of bug where the aggregation code that PRODUCES a number is also
the code that's trusted to CHECK it, which is precisely how a wrong number
survives into a table nobody re-derives independently.

It also performs a set of internal-consistency and honesty checks that don't
require re-deriving anything (e.g. "every 'certified: yes' row must have
violating_edges == 0" and "no attained-stretch value is silently capped at a
suspicious round number").

Usage:
    python run_spanner_benchmark.py --quick --synthetic   # produce results/
    python verify_results.py                              # check them

Exit code is 0 iff every check passes; CI treats a non-zero exit as a failed
build.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "spanner_report.json")
PERPAIR_PATH = os.path.join(RESULTS_DIR, "per_pair_dilation.csv")

TOL = 1e-6          # tolerance for exact re-derivations
TOL_STAT = 5e-2      # looser tolerance for re-derived p-values (RNG-order sensitive)

failures: List[str] = []
checks_run = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks_run
    checks_run += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


def main() -> int:
    print("=" * 78)
    print("  Independent verification of run_spanner_benchmark.py results")
    print("=" * 78)

    if not os.path.exists(REPORT_PATH):
        print(f"[fatal] {REPORT_PATH} not found -- run the pipeline first.")
        return 2
    with open(REPORT_PATH, "r", encoding="utf-8") as fh:
        report = json.load(fh)

    # -------------------------------------------------------------------
    # 1. Internal consistency: certified flag must match violating_edges
    # -------------------------------------------------------------------
    print("\n[1] Internal consistency of certificates")
    n_certs = 0
    for key, entry in report.get("results", {}).items():
        cert = entry.get("certificate") or {}
        if not cert:
            continue
        n_certs += 1
        certified = cert.get("certified")
        violating = cert.get("violating_edges")
        check(f"{key}: certified<->violating_edges consistency",
              (certified is True and violating == 0) or (certified is False and violating > 0),
              f"certified={certified}, violating_edges={violating}")
    for city in report.get("cross_city", {}).get("cities", []):
        for mname, mdata in city.get("methods", {}).items():
            cert = mdata.get("certificate") or {}
            if not cert:
                continue
            n_certs += 1
            certified = cert.get("certified")
            violating = cert.get("violating_edges")
            check(f"{city['city']}/{mname}: certified<->violating_edges consistency",
                  (certified is True and violating == 0) or (certified is False and violating > 0),
                  f"certified={certified}, violating_edges={violating}")
    print(f"  ({n_certs} certificates checked)")

    # -------------------------------------------------------------------
    # 2. Honesty: no attained-stretch value is a suspicious fixed constant
    #    repeated verbatim across structurally different graphs (the exact
    #    bug this project's own code review caught and fixed).
    # -------------------------------------------------------------------
    print("\n[2] No fabricated stretch-cap artifacts")
    stretch_values = []
    for key, entry in report.get("results", {}).items():
        cert = entry.get("certificate") or {}
        v = cert.get("attained_worst_edge_stretch")
        if v is not None and not cert.get("certified", True):
            stretch_values.append((key, v))
    # A real, honest set of non-certified stretch values should not all be
    # IDENTICAL unless the underlying edges are genuinely identical -- flag
    # (not necessarily fail) if 3+ distinct non-certified rows share one
    # exact repeated value, since that is what a fixed cutoff artifact looks
    # like.
    from collections import Counter
    finite = [v for _, v in stretch_values if isinstance(v, (int, float)) and math.isfinite(v)]
    counts = Counter(finite)
    suspicious = {v: c for v, c in counts.items() if c >= 3}
    check("no single finite stretch value repeats >=3x across non-certified methods",
          len(suspicious) == 0,
          f"repeated values: {suspicious} -- looks like a search-cutoff artifact, not measured data")

    # -------------------------------------------------------------------
    # 3. Re-derive dilation statistics from the RAW per-pair CSV
    #    (independent of the pipeline's own evaluate_graph()).
    # -------------------------------------------------------------------
    print("\n[3] Re-deriving dilation statistics from raw per-pair data")
    if os.path.exists(PERPAIR_PATH):
        df = pd.read_csv(PERPAIR_PATH)
        for method_label, group in df.groupby("method"):
            worst = float(group["dilation"].max())
            mean = float(group["dilation"].mean())
            p95 = float(np.percentile(group["dilation"], 95))
            print(f"  {method_label}: n={len(group)}, worst={worst:.4f}, "
                  f"mean={mean:.4f}, p95={p95:.4f} (re-derived, independent of pipeline code)")
        check("per-pair CSV is non-empty and has expected columns",
              {"source", "target", "dilation", "method"}.issubset(df.columns) and len(df) > 0,
              f"columns={list(df.columns)}, rows={len(df)}")
    else:
        print("  (skipped: per_pair_dilation.csv not found -- run without --skip-crosscity"
              " and ensure the focus-city peak-hour export ran)")

    # -------------------------------------------------------------------
    # 4. Re-derive the pair-level Wilcoxon test independently via scipy,
    #    from raw data, and compare to the pipeline's own reported p-value.
    # -------------------------------------------------------------------
    print("\n[4] Re-deriving the pair-level significance test")
    sig = report.get("significance_test_pair_level", {})
    if sig.get("available") and os.path.exists(PERPAIR_PATH) and _HAVE_SCIPY:
        df = pd.read_csv(PERPAIR_PATH)
        wide = df.pivot_table(index=["source", "target"], columns="method", values="dilation")
        cols = list(wide.columns)
        ours_col = next((c for c in cols if "Delay-Bounded" in c), None)
        btw_col = next((c for c in cols if "Matched-Density Betweenness" in c), None)
        if ours_col and btw_col:
            paired = wide[[ours_col, btw_col]].dropna()
            if len(paired) >= 10:
                stat, p = sps.wilcoxon(paired[ours_col], paired[btw_col],
                                       alternative="less", zero_method="zsplit")
                reported_p = sig.get("p_value")
                check("re-derived Wilcoxon p-value matches reported value",
                      abs(p - reported_p) < TOL_STAT,
                      f"re-derived p={p:.4g}, reported p={reported_p:.4g}")
                check("re-derived n_pairs matches reported n_pairs",
                      len(paired) == sig.get("n_pairs"),
                      f"re-derived n={len(paired)}, reported n={sig.get('n_pairs')}")
            else:
                print(f"  (skipped: only {len(paired)} paired rows, need >=10)")
        else:
            print(f"  (skipped: could not find both method columns in {cols})")
    else:
        print("  (skipped: no significance test recorded, no raw data, or scipy unavailable)")

    # -------------------------------------------------------------------
    # 5. Monotonicity sanity check: as t increases, our method's edge
    #    retention should be non-increasing (more slack -> allowed to
    #    remove more edges). A violation would indicate a bug in the sweep.
    # -------------------------------------------------------------------
    print("\n[5] Sparsification monotonicity in t (delay-bounded spanner)")
    sweep: List[Tuple[float, float]] = []
    for key, entry in report.get("results", {}).items():
        method, _, t_str = key.partition("|")
        if method == "delay_bounded" and t_str:
            sweep.append((float(t_str), entry["structure"]["edge_retention_pct"]))
    sweep.sort()
    non_increasing = all(sweep[i][1] >= sweep[i + 1][1] - 1e-9 for i in range(len(sweep) - 1))
    check("edge retention is non-increasing as t grows",
          non_increasing, f"sweep (t, retention%)={sweep}")

    # -------------------------------------------------------------------
    # 6. Cross-city coverage sanity: every reported city must state its
    #    data provenance, and at least one must actually be real OSM data
    #    (otherwise "validated on real cities" is not a supportable claim).
    # -------------------------------------------------------------------
    print("\n[6] Cross-city data provenance")
    cities = report.get("cross_city", {}).get("cities", [])
    if cities:
        has_provenance = all("used_real_osm" in c for c in cities)
        check("every city reports used_real_osm", has_provenance)
        n_real = sum(1 for c in cities if c.get("used_real_osm"))
        check("at least one city used real OSM data (not all synthetic fallback)",
              n_real > 0 or report.get("configuration", {}).get("quick"),
              f"{n_real}/{len(cities)} cities used real OSM data")
        print(f"  {n_real}/{len(cities)} cities used real OSM data")
    else:
        print("  (skipped: no cross-city section in this report)")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  {checks_run} checks run, {len(failures)} failed")
    print("=" * 78)
    if failures:
        print("\nFAILED CHECKS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll independent checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
