#!/usr/bin/env python3
"""Apply the README corrections found in the hostile-referee audit (docs/REFEREE_AUDIT.md). Idempotent."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "README.md")
s = open(p, encoding="utf-8").read()
applied = []

def sub(old, new, tag):
    global s
    if old in s:
        s = s.replace(old, new, 1); applied.append(tag)
    elif new in s:
        applied.append(tag + " (already)")
    else:
        print(f"[warn] anchor not found for: {tag}", file=sys.stderr)

sub("validated end-to-end on real metropolitan road networks pulled live from OpenStreetMap",
    "validated end-to-end on real OpenStreetMap city cores (about 1.5 km radius) pulled live from OpenStreetMap",
    "headline scope")

sub("the delay-bounded construction keeps equal or fewer edges (about 0.82x at t = 3 on random digraphs)",
    "the delay-bounded construction keeps fewer edges on average (about 0.82x at t = 3 on random digraphs) but is not provably never larger: in 6 of 236 random instances (t <= 1.5) it kept 1-2 more edges",
    "5b union claim")

sub("congestion model is parametric, not fitted to telemetry",
    "congestion model is parametric; its hourly shape agrees with measured freeway telemetry (METR-LA r = 0.86, PEMS-BAY r = 0.94, section 5b) but is not validated on urban street speeds",
    "research-program cell")

old = "- **Congestion model vs. measured traffic: not validated.**"
if old in s and "METR-LA r = 0.86" not in s.split(old)[1][:900]:
    i = s.index(old)
    j = s.index("\n", i)
    s = s[:i] + ("- **Congestion model vs. measured traffic: partially validated.** The hourly shape of the parametric model correlates with measured weekday-median speeds on two US freeway sensor networks (METR-LA r = 0.86, PEMS-BAY r = 0.94; section 5b), and the model overstates the peak/off-peak slowdown (1.50x vs. 1.34-1.38x measured). It has not been compared with measured speeds on urban street networks of any of the twelve cities; the NYC DOT comparison (`tools/validate_congestion_shape.py`) remains to be run.") + s[j:]
    applied.append("validation bullet")

sub("Numbers are **not** hand-copied into this README.",
    "Numbers in this section are **not** hand-copied into this README. (Sections 5b-5d quote values from `results/benchmarks/*.csv`; `tests/test_offgrid_and_readme.py` checks section 5d against its CSV.)",
    "section 4 note")

lim = "- Snapshot (time-slice) model:"
if lim in s and "grid hours only" not in s:
    s = s.replace(lim, "- The certificate (Theorem 1) holds at the 24 grid hours only. With the piecewise-linear interpolation used by `td_cost`, off-grid times and time-dependent traversal can exceed `t` slightly (observed: at most about 0.5% above `t` in random-instance tests, `benchmarks/offgrid_guarantee.py`). Theorem 3 gives a provable continuous-time bound `t` by certifying on a finer grid at `t / rho^2`, at the price of more retained edges.\n- Evaluation scale: city cores of about 1.5 km radius; 7 sources x 7 targets x 3 repeats of OD pairs per city by default (see `docs/REFEREE_AUDIT.md` for larger-scale commands).\n" + lim, 1)
    applied.append("limitations")

if "## Data and attribution" not in s:
    block = '''## Data and attribution

Street-network data: (c) OpenStreetMap contributors, available under the Open Database Licence (ODbL, https://www.openstreetmap.org/copyright). Derived graphs cached in `results/graph_cache/` are derivative databases under the same licence. Traffic speeds: METR-LA and PEMS-BAY as distributed by Li et al. (ICLR 2018; https://zenodo.org/records/5724362), sensor graphs from `liyaguang/DCRNN`. The graph snapshots behind the reported numbers should be archived with a DOI (e.g. Zenodo) at submission time.

---

'''
    k = "## 10. Citation"
    if k in s:
        s = s.replace(k, block + k, 1); applied.append("attribution")

open(p, "w", encoding="utf-8").write(s)
print("applied:", ", ".join(applied))
