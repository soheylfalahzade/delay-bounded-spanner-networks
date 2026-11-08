#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insert / refresh README sections 5l-5r (region faults, forced edges by hour, minimality, exact optimum on road windows,
emergency response, more cities, metropolitan scale) from the raw result files.

    python scripts/update_readme_new_sections.py            # write
    python scripts/update_readme_new_sections.py --check    # exit 1 if README differs from the files

Sections are generated only from results/benchmarks/region_fault.csv and forced_by_hour.csv; every sentence
that states a number is computed from those files.  A section whose CSV does not exist is not written, so the
README never describes a result that has not been produced.
"""
import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEN = os.path.join(ROOT, "results", "benchmarks")
README = os.path.join(ROOT, "README.md")
ANCHOR = "---\n\n## 5e. Figures"


def real_only(df, name):
    if "real_osm" in df.columns:
        assert df.real_osm.astype(bool).all(), f"{name}: synthetic rows must not enter the README"
    return df


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


def minimality_section(df: pd.DataFrame) -> str:
    real_only(df, "minimality")
    rows = [[r.city, f"{r.edges}", f"{r.t:g}", f"{r.greedy_retention_pct:.1f}", f"{r.removable_edges}",
             f"{r.removable_pct_of_spanner:.2f}", f"{r.pruned_retention_pct:.1f}"] for r in df.itertuples()]
    table = md(["City", "Edges", "t", "Greedy retention (%)", "Removable spanner edges", "Removable (% of spanner)",
                "Retention after pruning (%)"], rows)
    n_min = int(df.greedy_is_minimal.astype(bool).sum())
    return f"""### 5n. Is the greedy spanner edge-minimal?

For one static weighting the greedy spanner contains no proper t-spanner of itself [15]. With hourly weights the construction orders edges by their maximum travel time, so an edge kept early can become redundant once later edges provide a bypass in the hour that matters. `benchmarks/minimality.py` tests this: after the construction, edges of H are examined in reverse order and deleted whenever H minus the edge still satisfies the exact Theorem-1 certificate for every edge of G at all 24 hours. Every pruned spanner is re-certified exactly. (With a constant profile no edge is removable; this is checked in `tests/test_minimality.py`.)

<!-- BEGIN:minimality -->
{table}

The construction is edge-minimal in {n_min} of {len(df)} (city, t) cells. Where it is not, {df[df.removable_edges > 0].removable_pct_of_spanner.min():.2f}% to {df.removable_pct_of_spanner.max():.2f}% of the spanner's edges can be removed by the pruning pass (largest over cells), without losing the certificate.
<!-- END:minimality -->

"""


def road_exact_section(df: pd.DataFrame) -> str:
    real_only(df, "road_exact")
    skipped = int(df.skipped.astype(bool).sum())
    d = df[~df.skipped.astype(bool)].copy()
    d["ex"] = d.greedy - d.optimum
    d["exp"] = d.greedy_pruned - d.optimum
    assert (d.forced <= d.optimum).all() and (d.optimum <= d.greedy_pruned).all() and (d.greedy_pruned <= d.greedy).all()
    rows = []
    for t, g in d.groupby("t"):
        rows.append([f"{t:g}", f"{len(g)}", f"{g.edges.mean():.1f}", f"{g.free.mean():.1f}",
                     f"{100 * (g.ex == 0).mean():.0f}", f"{100 * (g.exp == 0).mean():.0f}",
                     f"{g.ex.mean():.2f}", f"{int(g.ex.max())}"])
    table = md(["t", "Windows solved exactly", "Edges per window (mean)", "Non-forced edges per window (mean)",
                "Greedy optimal (%)", "Greedy + pruning optimal (%)", "Mean greedy excess (edges)",
                "Largest greedy excess (edges)"], rows)
    return f"""### 5o. Exact optimum on windows of real road networks

Section 5j compares the greedy construction with the exact optimum on small random digraphs. `benchmarks/road_exact.py` repeats the comparison on windows cut from the real city cores: the subgraph induced by the first k intersections of a breadth-first search from a random intersection, restricted to its largest strongly connected component. The edge-wise certificate is necessary and sufficient for the all-pairs, all-hours bound, and every forced edge is in every spanner, so the optimum is the number of forced edges plus the size of the smallest set of non-forced edges completing the certificate; subsets are enumerated by increasing size. Windows with too many non-forced edges for the enumeration are skipped ({skipped} here), never approximated.

<!-- BEGIN:road_exact -->
{table}

Over all solved windows the greedy construction is optimal in {100 * (d.ex == 0).mean():.0f}% of the cases and exceeds the optimum by at most {int(d.ex.max())} edge(s); after the pruning pass of section 5n it is optimal in {100 * (d.exp == 0).mean():.0f}%.
<!-- END:road_exact -->

Scope: windows are small induced subgraphs of the city cores, so their boundary is artificial, and the windows solved exactly are biased towards those with few non-forced edges. This is a statement about small road-like instances, not about a whole network.

"""


def emergency_section(df: pd.DataFrame) -> str:
    real_only(df, "emergency_response")
    intact = df[df.scenario == "intact"]
    assert (intact.over_t_pct == 0).all(), "the certificate is violated on the intact network: bug"
    rows = [[r.city, r.scenario, r.spanner, f"{r.unreachable_pct:.2f}", f"{r.mean_ratio:.3f}", f"{r.p95_ratio:.3f}",
             f"{r.max_ratio:.3f}", f"{r.over_t_pct:.2f}"] for r in df.itertuples()]
    table = md(["City", "Scenario", "Spanner", "Targets unreachable (%)", "Mean ratio", "95th percentile ratio",
                "Maximum ratio", "Ratio above t (%)"], rows)
    t = df.t.iloc[0]
    hr = int(df.hour.iloc[0])
    return f"""### 5p. Emergency response times

Response bases are placed by farthest-point sampling (they stand for response bases, not for real hospitals or fire stations). For each base and each target the travel time on the spanner is compared with the travel time on the full network at {hr}:00, ratio = d_H / d_G, at t = {t:g}. After a disc fault the comparison is between H-F and G-F (bases and targets outside the fault). Two spanners are compared: the one built for the intact network and the fault-tolerant one of section 5l, built for a training family of faults. `benchmarks/emergency_response.py`.

<!-- BEGIN:emergency -->
{table}

On the intact network the ratio never exceeds t (the certificate). Under faults the intact-network spanner leaves {df[(df.scenario == 'held-out faults') & (df.spanner == 'intact-network spanner')].unreachable_pct.mean():.2f}% of the targets unreachable on average over held-out faults, the fault-tolerant spanner {df[(df.scenario == 'held-out faults') & (df.spanner == 'fault-tolerant spanner')].unreachable_pct.mean():.2f}%; the ratio exceeds t for {df[(df.scenario == 'held-out faults') & (df.spanner == 'intact-network spanner')].over_t_pct.mean():.2f}% and {df[(df.scenario == 'held-out faults') & (df.spanner == 'fault-tolerant spanner')].over_t_pct.mean():.2f}% of the (base, target) pairs respectively (means over cities).
<!-- END:emergency -->

Scope: one hour of the day, the parametric traffic model, disc faults, five bases per city.

"""


def more_cities_section(df: pd.DataFrame) -> str:
    real_only(df, "more_cities")
    cor_path = os.path.join(BEN, "more_cities_correlations.csv")
    cor = pd.read_csv(cor_path) if os.path.exists(cor_path) else pd.DataFrame()
    rows = []
    for t, g in df.groupby("t"):
        rows.append([f"{t:g}", f"{g.city.nunique()}", f"{g.forced_lower_bound_pct.mean():.1f}",
                     f"{g.retention_pct.mean():.1f}", f"{100 * g.capture.mean():.0f}% ({100 * g.capture.min():.0f}-{100 * g.capture.max():.0f}%)"])
    table = md(["t", "Cities", "Forced bound (%, mean)", "Greedy retention (%, mean)",
                "Share of slack removed: mean (range)"], rows)
    n = df.city.nunique()
    sig = cor[cor["significant_BH_0.05"]] if len(cor) else cor
    if len(cor):
        sig_rows = [[f"{r.t:g}", r.target, r.descriptor, f"{int(r.n)}", f"{r.spearman_rho:.2f}", f"{r.p_BH:.3f}"]
                    for r in sig.itertuples()]
        sigtext = (md(["t", "Target", "Descriptor", "n", "Spearman rho", "BH-adjusted p"], sig_rows)
                   if len(sig) else "No descriptor is significant after Benjamini-Hochberg adjustment.")
        ctext = f"\n\nSpearman correlations with the morphology descriptors, {len(cor)} tests pooled; significant after adjustment:\n\n{sigtext}"
    else:
        ctext = ""
    return f"""### 5q. More cities

The twelve registry cities give a low-powered morphology test (section 5d). `benchmarks/more_cities.py` adds further real OpenStreetMap city cores (about 1.5 km radius; a city whose download fails is skipped, never replaced by a synthetic network) and pools them with the registry cities: {n} cities in total ({int((df.drop_duplicates('city').source == 'added').sum())} added).

<!-- BEGIN:more_cities -->
{table}{ctext}
<!-- END:more_cities -->

`bridge_pct` is, by construction, a lower bound on retention (an arc with no detour within ten times its travel time is forced for every t up to ten), so its correlation with retention is structural; `true_bridge_pct`, `mean_degree` and the other descriptors are the informative ones. Correlations are descriptive: the descriptors are correlated with each other, city cores are not independent draws from a population of cities, and the added cities are a convenience sample.

"""


def metro_section(df: pd.DataFrame) -> str:
    real_only(df, "metro")
    assert df.delay_bounded_certified.astype(bool).all(), "an uncertified spanner at metropolitan scale: bug"
    rows = [[r.city, f"{int(r.dist_m)}", f"{int(r.nodes)}", f"{int(r.edges)}", f"{r.t:g}",
             f"{r.delay_bounded_retention_pct:.1f}", f"{r.forced_lower_bound_pct:.1f}"] for r in df.itertuples()]
    table = md(["City", "Radius (m)", "Nodes", "Edges", "t", "Delay-bounded kept (%)", "Forced lower bound (%)"], rows)
    return f"""### 5r. Metropolitan scale

Real OpenStreetMap networks at radii above 3.5 km (`python benchmarks/large_scale_run.py --dist ...`), every spanner certified.

<!-- BEGIN:metro -->
{table}

The largest network has {int(df.edges.max())} edges.
<!-- END:metro -->

"""


def metro_frame():
    frames = []
    for f in sorted(glob.glob(os.path.join(BEN, "large_scale_*m.csv"))):
        b = os.path.basename(f)
        if b == "large_scale_3500m.csv" or "synthetic" in b:
            continue
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True) if frames else None


def replace_or_insert(text: str, sec: str) -> str:
    """Replace the whole section that starts with the same heading (intro text included, so that numbers
    quoted in the introduction cannot go stale), or insert it before the anchor."""
    heading = sec.split("\n", 1)[0]
    start = text.find(heading)
    if start < 0:
        assert ANCHOR in text, "README anchor not found"
        return text.replace(ANCHOR, sec + ANCHOR, 1)
    ends = [i for i in (text.find("\n### ", start + len(heading)) + 1, text.find(ANCHOR, start)) if i > 0]
    end = min(ends)
    return text[:start] + sec + text[end:]


def render(text: str) -> str:
    out = text
    jobs = [("region_fault", "region_fault.csv", region_fault_section), ("by_hour", "forced_by_hour.csv", by_hour_section),
            ("minimality", "minimality.csv", minimality_section), ("road_exact", "road_exact_raw.csv", road_exact_section),
            ("emergency", "emergency_response.csv", emergency_section), ("more_cities", "more_cities.csv", more_cities_section),
            ("metro", None, metro_section)]
    for name, csv, fn in jobs:
        if name == "metro":
            frame = metro_frame()
            if frame is None:
                continue
        else:
            path = os.path.join(BEN, csv)
            if not os.path.exists(path):
                continue
            frame = pd.read_csv(path)
        sec = fn(frame)
        out = replace_or_insert(out, sec)
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
