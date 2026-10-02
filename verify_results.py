#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent verification of run_spanner_benchmark.py's published claims.

This script is deliberately NOT part of run_spanner_benchmark.py and does not
import its aggregation/reporting code. It re-derives a set of headline
numbers from the RAW output artifacts (per_pair_dilation.csv,
spanner_report.json) using a separate, independently-written code path
(plain pandas/numpy/scipy), and checks them against what the pipeline itself
reported. The point is not to re-run the experiment -- it is to catch the
class of bug where the aggregation code that PRODUCES a number is also the
code trusted to CHECK it, which is precisely how a wrong number survives
into a table nobody re-derives independently.

It also performs internal-consistency and honesty checks that don't require
re-deriving anything (e.g. "every 'certified: yes' row must have
violating_edges == 0", "no attained-stretch value is silently capped at a
suspicious round number", "an SCC filter can only remove nodes, never add
them", "a Benjamini-Hochberg-adjusted p-value is never smaller than its own
raw p-value").

Usage:
    python run_spanner_benchmark.py --quick --synthetic   # produce results/
    python verify_results.py                              # check them
    python verify_results.py --results-dir /some/other/dir

Exit code is 0 iff every check passes; CI treats a non-zero exit as a failed
build. See tests/test_verify_results.py for proof that this script actually
catches injected errors rather than rubber-stamping whatever it is given.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT, "results")

TOL = 1e-6           # tolerance for exact re-derivations
TOL_STAT = 5e-2       # looser tolerance for re-derived p-values (RNG-order sensitive)


@dataclass
class CheckLog:
    """Local (not module-global) mutable state for one verification run, so
    run_checks() can be called repeatedly in-process (e.g. from unit tests)
    without one call's results leaking into the next."""
    checks_run: int = 0
    failures: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks_run += 1
        status = "PASS" if ok else "FAIL"
        self.say(f"  [{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(f"{name}: {detail}")


def run_checks(report_path: str, perpair_path: str) -> Tuple[int, CheckLog]:
    """Run every independent check against the artifacts at the given paths.

    Returns (exit_code, log). exit_code is 0 iff every check passed, 2 if
    the report file itself could not be found/parsed. This function has no
    side effects besides reading the two input files, so it is safe to call
    repeatedly in a single process (see tests/test_verify_results.py).
    """
    log = CheckLog()
    log.say("=" * 78)
    log.say("  Independent verification of run_spanner_benchmark.py results")
    log.say("=" * 78)

    if not os.path.exists(report_path):
        log.say(f"[fatal] {report_path} not found -- run the pipeline first.")
        return 2, log
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except json.JSONDecodeError as exc:
        log.say(f"[fatal] {report_path} is not valid JSON: {exc}")
        return 2, log

    # -------------------------------------------------------------------
    # 1. Internal consistency: certified flag must match violating_edges
    # -------------------------------------------------------------------
    log.say("\n[1] Internal consistency of certificates")
    n_certs = 0
    for key, entry in report.get("results", {}).items():
        cert = entry.get("certificate") or {}
        if not cert:
            continue
        n_certs += 1
        certified = cert.get("certified")
        violating = cert.get("violating_edges")
        log.check(f"{key}: certified<->violating_edges consistency",
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
            log.check(f"{city['city']}/{mname}: certified<->violating_edges consistency",
                     (certified is True and violating == 0) or (certified is False and violating > 0),
                     f"certified={certified}, violating_edges={violating}")
    log.say(f"  ({n_certs} certificates checked)")

    # -------------------------------------------------------------------
    # 2. Honesty: no attained-stretch value is a suspicious fixed constant
    #    repeated verbatim across structurally different graphs (the exact
    #    bug this project's own code review caught and fixed).
    # -------------------------------------------------------------------
    log.say("\n[2] No fabricated stretch-cap artifacts")
    stretch_values = []
    for key, entry in report.get("results", {}).items():
        cert = entry.get("certificate") or {}
        v = cert.get("attained_worst_edge_stretch")
        if v is not None and not cert.get("certified", True):
            stretch_values.append((key, v))
    finite = [v for _, v in stretch_values if isinstance(v, (int, float)) and math.isfinite(v)]
    counts = Counter(finite)
    suspicious = {v: c for v, c in counts.items() if c >= 3}
    log.check("no single finite stretch value repeats >=3x across non-certified methods",
             len(suspicious) == 0,
             f"repeated values: {suspicious} -- looks like a search-cutoff artifact, not measured data")

    # -------------------------------------------------------------------
    # 3. Re-derive dilation statistics from the RAW per-pair CSV
    #    (independent of the pipeline's own evaluate_graph()).
    # -------------------------------------------------------------------
    log.say("\n[3] Re-deriving dilation statistics from raw per-pair data")
    if os.path.exists(perpair_path):
        df = pd.read_csv(perpair_path)
        for method_label, group in df.groupby("method"):
            worst = float(group["dilation"].max())
            mean = float(group["dilation"].mean())
            p95 = float(np.percentile(group["dilation"], 95))
            log.say(f"  {method_label}: n={len(group)}, worst={worst:.4f}, "
                   f"mean={mean:.4f}, p95={p95:.4f} (re-derived, independent of pipeline code)")
        log.check("per-pair CSV is non-empty and has expected columns",
                 {"source", "target", "dilation", "method"}.issubset(df.columns) and len(df) > 0,
                 f"columns={list(df.columns)}, rows={len(df)}")
    else:
        log.say("  (skipped: per_pair_dilation.csv not found -- run without --skip-crosscity"
               " and ensure the focus-city peak-hour export ran)")

    # -------------------------------------------------------------------
    # 4. Re-derive the pair-level Wilcoxon test independently via scipy,
    #    from raw data, and compare to the pipeline's own reported p-value.
    # -------------------------------------------------------------------
    log.say("\n[4] Re-deriving the pair-level significance test")
    sig = report.get("significance_test_pair_level", {})
    if sig.get("available") and os.path.exists(perpair_path) and _HAVE_SCIPY:
        df = pd.read_csv(perpair_path)
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
                log.check("re-derived Wilcoxon p-value matches reported value",
                         abs(p - reported_p) < TOL_STAT,
                         f"re-derived p={p:.4g}, reported p={reported_p:.4g}")
                log.check("re-derived n_pairs matches reported n_pairs",
                         len(paired) == sig.get("n_pairs"),
                         f"re-derived n={len(paired)}, reported n={sig.get('n_pairs')}")
            else:
                log.say(f"  (skipped: only {len(paired)} paired rows, need >=10)")
        else:
            log.say(f"  (skipped: could not find both method columns in {cols})")
    else:
        log.say("  (skipped: no significance test recorded, no raw data, or scipy unavailable)")

    # -------------------------------------------------------------------
    # 5. Monotonicity sanity check: as t increases, our method's edge
    #    retention should be non-increasing. A violation indicates a bug
    #    in the sweep.
    # -------------------------------------------------------------------
    log.say("\n[5] Sparsification monotonicity in t (delay-bounded spanner)")
    sweep: List[Tuple[float, float]] = []
    for key, entry in report.get("results", {}).items():
        method, _, t_str = key.partition("|")
        if method == "delay_bounded" and t_str:
            sweep.append((float(t_str), entry["structure"]["edge_retention_pct"]))
    sweep.sort()
    non_increasing = all(sweep[i][1] >= sweep[i + 1][1] - 1e-9 for i in range(len(sweep) - 1))
    log.check("edge retention is non-increasing as t grows",
             non_increasing, f"sweep (t, retention%)={sweep}")

    # -------------------------------------------------------------------
    # 6. Cross-city coverage sanity.
    # -------------------------------------------------------------------
    log.say("\n[6] Cross-city data provenance")
    cities = report.get("cross_city", {}).get("cities", [])
    if cities:
        has_provenance = all("used_real_osm" in c for c in cities)
        log.check("every city reports used_real_osm", has_provenance)
        n_real = sum(1 for c in cities if c.get("used_real_osm"))
        log.check("at least one city used real OSM data (not all synthetic fallback)",
                 n_real > 0 or report.get("configuration", {}).get("quick"),
                 f"{n_real}/{len(cities)} cities used real OSM data")
        log.say(f"  {n_real}/{len(cities)} cities used real OSM data")
    else:
        log.say("  (skipped: no cross-city section in this report)")

    # -------------------------------------------------------------------
    # 7. Connectivity-funnel sanity: after-counts can never exceed before-
    #    counts, for the focus city and every cross-city entry that reports one.
    # -------------------------------------------------------------------
    log.say("\n[7] Connectivity-funnel sanity (raw fetch -> largest SCC)")
    funnel = report.get("network", {}).get("connectivity_funnel")
    if funnel:
        log.check("focus city: nodes_after <= nodes_before",
                 funnel["nodes_after"] <= funnel["nodes_before"], str(funnel))
        log.check("focus city: edges_after <= edges_before",
                 funnel["edges_after"] <= funnel["edges_before"], str(funnel))
    else:
        log.say("  (skipped: no connectivity_funnel recorded for the focus city)")
    n_city_funnels = 0
    for city in report.get("cross_city", {}).get("cities", []):
        cf = city.get("connectivity_funnel")
        if not cf:
            continue
        n_city_funnels += 1
        log.check(f"{city['city']}: nodes_after <= nodes_before",
                 cf["nodes_after"] <= cf["nodes_before"], str(cf))
        log.check(f"{city['city']}: edges_after <= edges_before",
                 cf["edges_after"] <= cf["edges_before"], str(cf))
    log.say(f"  ({n_city_funnels} city funnels checked)")

    # -------------------------------------------------------------------
    # 8. FDR correction sanity: a Benjamini-Hochberg-adjusted p-value must
    #    never be smaller than its own raw p-value.
    # -------------------------------------------------------------------
    log.say("\n[8] FDR correction sanity (cross-city morphology regression)")
    reg = report.get("cross_city", {}).get("morphology_regression", {})
    if reg.get("available"):
        n_fdr = 0
        for key, entry in reg.items():
            if not isinstance(entry, dict) or "p" not in entry or "p_fdr_bh" not in entry:
                continue
            n_fdr += 1
            log.check(f"{key}: FDR-adjusted p >= raw p",
                     entry["p_fdr_bh"] >= entry["p"] - TOL,
                     f"raw p={entry['p']:.4f}, adjusted p={entry['p_fdr_bh']:.4f}")
        log.say(f"  ({n_fdr} corrected tests checked)")
    else:
        log.say("  (skipped: no morphology regression available in this report)")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    log.say("\n" + "=" * 78)
    log.say(f"  {log.checks_run} checks run, {len(log.failures)} failed")
    log.say("=" * 78)
    if log.failures:
        log.say("\nFAILED CHECKS:")
        for f in log.failures:
            log.say(f"  - {f}")
        return 1, log
    log.say("\nAll independent checks passed.")
    return 0, log


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="directory containing spanner_report.json and "
                         "per_pair_dilation.csv (default: ./results next to this script)")
    ap.add_argument("--report", default=None, help="override path to spanner_report.json")
    ap.add_argument("--perpair", default=None, help="override path to per_pair_dilation.csv")
    args = ap.parse_args(argv)

    report_path = args.report or os.path.join(args.results_dir, "spanner_report.json")
    perpair_path = args.perpair or os.path.join(args.results_dir, "per_pair_dilation.csv")

    exit_code, log = run_checks(report_path, perpair_path)
    print("\n".join(log.lines))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
