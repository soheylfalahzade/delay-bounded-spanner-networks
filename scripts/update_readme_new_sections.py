#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insert / refresh README sections 5l (region faults) and 5m (forced edges by hour) from the raw result files.

    python scripts/update_readme_new_sections.py            # write
    python scripts/update_readme_new_sections.py --check    # exit 1 if README differs from the files

Sections are generated only from results/benchmarks/region_fault.csv and forced_by_hour.csv; every sentence
that states a number is computed from those files.  A section whose CSV does not exist is not written, so the
README never describes a result that has not been produced.
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEN = os.path.join(ROOT, "results", "benchmarks")
README = os.path.join(ROOT, "README.md")
ANCHOR = "---\n\n## 5e. Figures"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def region_fault_section(df: pd.DataFrame) -> str:
    assert (df.ft_fails_train_regions_pct == 0).all(), "fault-tolerant construction failed on its own family"
    assert (df.fault_tolerant_retention_pct + 1e-9 >= df.fault_forced_lb_pct).all()
    assert (df.fault_forced_lb_pct + 1e-9 >= df.nominal_forced_lb_pct).all()
    assert (df.fault_tolerant_retention_pct + 1e-9 >= df.nominal_retention_pct).all()
    rows = [[r.city, f"{r.edges}", f"{r.t:g}", f"{r.nominal_retention_pct:.1f}", f"{r.nominal_forced_lb_pct:.1f}",
             f"{r.fault_forced_lb_pct:.1f}", f"{r.fault_tolerant_retention_pct:.1f}",
             f"{r.nominal_fails_train_regions_pct:.0f}", f"{r.nominal_mean_violating_edge_pct:.2f}",
             f"{r.ft_fails_heldout_regions_pct:.0f}", f"{r.nominal_fails_heldout_regions_pct:.0f}"]
            for r in df.itertuples()]
    table = md(["City", "Edges", "t", "Nominal retention (%)", "Nominal forced bound (%)",
                "Forced bound over the family (%)", "Fault-tolerant retention (%)",
                "Regions where the nominal spanner fails (%)", "Violating edges of G-F, nominal (%, mean)",
                "Held-out regions where the fault-tolerant spanner fails (%)",
                "Held-out regions where the nominal spanner fails (%)"], rows)
    k = df.regions_train.iloc[0]
    h = df.regions_heldout.iloc[0]
    text = f"""### 5l. Region faults (the definition of Abam et al. applied to road networks)

Abam, de Berg, Farshi and Gudmundsson define region-fault tolerance by comparing the spanner and the reference graph *after the same fault*: for a convex region F, the spanner minus F must be a t-spanner of the reference graph minus F [1]. Here a fault is a disc: every intersection inside it and every street segment touching it is removed from both G and H, and H-F must satisfy the delay bound against G-F at every hour. Theorem 1 applies unchanged to the pair (G-F, H-F), so the check is exact (`benchmarks/region_fault.py`).

Three quantities are measured on a random family of {k} disc faults (radii 5%, 10% and 20% of the network radius, centred on random intersections):

1. how often the spanner built for the intact network fails the bound after a fault;
2. a lower bound for any edge-subset spanner that is fault tolerant for the family: an edge with detour ratio above t in some G-F is forced in every such spanner, and nominally forced edges stay forced because G-F is a subgraph of G;
3. a fault-tolerant greedy construction that drops an edge only if a detour within budget exists in H-F for every fault of the family that leaves the edge intact and every hour, certified for the family by Theorem 1 and checked exactly.

<!-- BEGIN:region_fault -->
{table}

Fault-tolerant retention exceeds the nominal retention by {(df.fault_tolerant_retention_pct - df.nominal_retention_pct).min():.2f} to {(df.fault_tolerant_retention_pct - df.nominal_retention_pct).max():.2f} points. The spanner built for the intact network fails the bound in {df.nominal_fails_train_regions_pct.min():.0f}% to {df.nominal_fails_train_regions_pct.max():.0f}% of the family's regions. On {h} held-out regions drawn with a different seed, the fault-tolerant spanner fails in {df.ft_fails_heldout_regions_pct.min():.0f}% to {df.ft_fails_heldout_regions_pct.max():.0f}% of regions, against {df.nominal_fails_heldout_regions_pct.min():.0f}% to {df.nominal_fails_heldout_regions_pct.max():.0f}% for the nominal spanner.
<!-- END:region_fault -->

Scope: tolerance holds for the sampled family only; regions outside it are covered by the held-out column, not by the certificate. Faults are discs, a street segment touches a disc if its stored polyline does, and the family is a finite random sample. The traffic model is the parametric one used elsewhere in this repository. The forced-bound over the family is a lower bound for that family only and is not claimed to be tight.

"""
    return text


def by_hour_section(df: pd.DataFrame) -> str:
    g = df.groupby("t").mean(numeric_only=True)
    rows = [[f"{t:g}", f"{r.forced_any_pct:.1f}", f"{r.forced_all_hours_pct:.1f}", f"{r.forced_some_hours_pct:.1f}",
             f"{r.forced_free_flow_pct:.1f}", f"{r.time_only_forced_pct:.1f}", f"{r.static_only_forced_pct:.1f}",
             f"{r.peak_not_night_pct:.1f}"] for t, r in g.iterrows()]
    table = md(["t", "forced at some hour (%)", "forced at every hour (%)", "forced at some but not all hours (%)",
                "forced with free-flow weights (%)", "forced only because of time variation (%)",
                "forced at free flow but at no hour (%)", "forced at a peak hour and at no night hour (%)"], rows)
    n = df.city.nunique()
    tb = df.groupby("city").true_bridge_pct.first()
    t2 = g.loc[2.0] if 2.0 in g.index else g.iloc[0]
    t2v = 2.0 if 2.0 in g.index else g.index[0]
    return f"""### 5m. Forced edges by hour

An edge is forced (Theorem 2) if its detour ratio exceeds t at *some* hour. `benchmarks/forced_by_hour.py` splits that set by hour and compares it with the forced set that a static analysis would obtain from free-flow weights, to show how much of the bound depends on time variation. Mean over {n} networks:

<!-- BEGIN:by_hour -->
{table}

At t = {t2v:g}, {t2.forced_any_pct:.1f}% of the edges are forced at some hour, {t2.forced_all_hours_pct:.1f}% at every hour, and {t2.time_only_forced_pct:.1f}% are forced only because of the hourly profile (not forced with free-flow weights). The share of true bridges (arcs with no alternative directed path at all) is {tb.min():.1f}% to {tb.max():.1f}% across the networks. Per-city values: `results/benchmarks/forced_by_hour.csv`; share forced at each hour: `forced_by_hour_profile.csv`.
<!-- END:by_hour -->

The congestion profiles of the street networks are parametric (section 8), so the time-dependent part of this decomposition describes the model, not measured street speeds.

"""


def render(text: str) -> str:
    out = text
    for name, csv, fn in (("region_fault", "region_fault.csv", region_fault_section),
                          ("by_hour", "forced_by_hour.csv", by_hour_section)):
        path = os.path.join(BEN, csv)
        if not os.path.exists(path):
            continue
        sec = fn(pd.read_csv(path))
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        if begin in out:
            # replace only the generated block, keeping the hand-written text around it
            new_block = sec[sec.index(begin): sec.index(end) + len(end)]
            out = out[: out.index(begin)] + new_block + out[out.index(end) + len(end):]
        else:
            assert ANCHOR in out, "README anchor not found"
            out = out.replace(ANCHOR, sec + ANCHOR, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = open(README, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    new = render(text)
    if a.check:
        if new != text:
            print("README sections 5l/5m differ from the result files")
            sys.exit(1)
        print("README sections 5l/5m match the result files (or their CSVs do not exist)")
        return
    open(README, "w", encoding="utf-8", newline="").write(new)
    print("README updated")


if __name__ == "__main__":
    main()
