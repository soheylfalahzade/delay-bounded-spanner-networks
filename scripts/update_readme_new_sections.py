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
    df = df.copy()
    if "profile" not in df.columns:
        df["profile"] = "hourly"
    rows = [[r.city, r.profile, f"{r.edges}", f"{r.t:g}", f"{r.greedy_retention_pct:.1f}", f"{r.removable_edges}",
             f"{r.removable_pct_of_spanner:.2f}", f"{r.pruned_retention_pct:.1f}"] for r in df.itertuples()]
    table = md(["City", "Profile", "Arcs", "t", "Greedy retention (%)", "Removable spanner arcs",
                "Removable (% of spanner)", "Retention after pruning (%)"], rows)
    n_min = int(df.greedy_is_minimal.astype(bool).sum())
    bits = []
    for prof, g in df.groupby("profile"):
        nz = g[g.removable_edges > 0]
        rng = f"{nz.removable_pct_of_spanner.min():.2f}% to {g.removable_pct_of_spanner.max():.2f}%" if len(nz) else "none"
        bits.append(f"with the {prof} profile the construction is edge-minimal in {int(g.greedy_is_minimal.astype(bool).sum())} of {len(g)} (city, t) cells and the removable share of the spanner is {rng}")
    return f"""### 5n. Is the greedy spanner edge-minimal?

For an undirected graph with one static weighting, the greedy spanner has no proper t-spanner of itself [15]. The argument does not carry over to directed graphs: the cycle closed by the last arc of a removed arc's bypass gives a path in the wrong direction. So non-minimality is possible on a directed network even with one static weighting, and with hourly weights the construction also orders arcs by their maximum travel time, so an arc kept early can become redundant once later arcs provide a bypass in the hour that matters. `benchmarks/minimality.py` tests this on every network with the hourly profile and with a static profile (free-flow travel time at every hour): after the construction, arcs of H are examined in reverse order and deleted whenever H minus the arc still satisfies the exact Theorem-1 certificate for every arc of G at all 24 hours. Every pruned spanner is re-certified exactly.

<!-- BEGIN:minimality -->
{table}

Over {len(df)} (city, profile, t) cells: {'; '.join(bits)}.
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



def directedness_section(df: pd.DataFrame) -> str:
    real_only(df, "directedness")
    cp = os.path.join(BEN, "directedness_correlations.csv")
    cor = pd.read_csv(cp) if os.path.exists(cp) else pd.DataFrame()
    rows = []
    for t, g in df.groupby("t"):
        rows.append([f"{t:g}", f"{g.city.nunique()}", f"{g.one_way_pct.mean():.1f}", f"{g.forced_directed.mean():.1f}",
                     f"{g.forced_two_way.mean():.1f}", f"{g.retention_directed.mean():.1f}",
                     f"{g.retention_two_way.mean():.1f}", f"{g.diff_retention.mean():+.2f}"])
    table = md(["t", "Cities", "One-way arcs (%)", "Forced, directed (%)", "Forced, two-way version (%)",
                "Retention, directed (%)", "Retention, two-way version (%)", "Retention difference (points)"], rows)
    if len(cor):
        ct = md(["t", "Target", "n", "Spearman rho (one-way share)", "BH-adjusted p"],
                [[f"{r.t:g}", r.target, f"{int(r.n)}", f"{r.spearman_rho:.2f}", f"{r.p_BH:.3f}"] for r in cor.itertuples()])
        ctext = "\n\nRelation to the share of one-way arcs across cities:\n\n" + ct
    else:
        ctext = ""
    return f"""### 5s. How much does direction matter?

Each network is compared with its two-way version, in which every arc without a reverse arc gets one with the same hourly travel times; arcs that already have a reverse are unchanged. The two versions are different graphs, so percentages are of each graph's own arcs. `benchmarks/directedness.py`.

<!-- BEGIN:directedness -->
{table}{ctext}
<!-- END:directedness -->

Scope: the two-way version is a counterfactual (the same congestion on the added arcs), not a real network; the cities are a convenience sample.

"""


def lightness_section(df: pd.DataFrame) -> str:
    real_only(df, "lightness")
    rows = []
    for t, g in df.groupby("t"):
        rows.append([f"{t:g}", f"{g.city.nunique()}", f"{g.kept_count_pct.mean():.1f}", f"{g.kept_length_pct.mean():.1f}",
                     f"{g.forced_length_pct.mean():.1f}", f"{g.lightness_graph.mean():.2f}", f"{g.lightness_spanner.mean():.2f}",
                     f"{g.lightness_lower_bound.mean():.2f}",
                     f"{100 * g.capture_length.mean():.0f}% ({100 * g.capture_length.min():.0f}-{100 * g.capture_length.max():.0f}%)"])
    table = md(["t", "Cities", "Arcs kept (%)", "Street length kept (%)", "Forced street length (%)",
                "Lightness of the network", "Lightness of the spanner", "Lightness lower bound",
                "Share of the removable length removed: mean (range)"], rows)
    return f"""### 5t. Total street length and lightness

Street length, not the number of arcs, is what a road authority maintains. On physical street segments (a two-way street counts once; a segment is kept if any of its arcs is kept) this section reports the share of the street length kept, the lightness (kept length divided by the length of a minimum spanning tree of the segments, the standard spanner measure), and a lower bound for the length of any edge-subset spanner: the length of the segments that carry a forced arc. `benchmarks/lightness.py`.

<!-- BEGIN:lightness -->
{table}
<!-- END:lightness -->

The lower bound is a bound on edge-subset spanners, as elsewhere; the minimum spanning tree is computed on the undirected segment graph, so lightness ignores one-way restrictions.

"""


def edge_order_section(df: pd.DataFrame) -> str:
    real_only(df, "edge_order")
    rows = []
    for t, g in df.groupby("t"):
        base = g[g["order"] == "temporal_max"].groupby("city").retention_pct.mean()
        for name, h in g.groupby("order"):
            per_city = h.groupby("city")[["retention_pct", "pruned_retention_pct"]].mean()
            diff = per_city.retention_pct - base.reindex(per_city.index)
            rows.append([f"{t:g}", name, f"{per_city.index.nunique()}", f"{per_city.retention_pct.mean():.2f}",
                         f"{diff.mean():+.2f}", f"{diff.min():+.2f} to {diff.max():+.2f}",
                         f"{per_city.pruned_retention_pct.mean():.2f}"])
    table = md(["t", "Order", "Cities", "Retention (%)", "Difference from temporal_max (points)",
                "Range over cities (points)", "Retention after pruning (%)"], rows)
    return f"""### 5u. Does the processing order matter?

The greedy construction depends on the order in which arcs are examined. Six orders are compared on every network, each result certified exactly and also pruned by the reverse-delete pass of section 5n: ascending maximum travel time (default), ascending mean, a betweenness-damped order, ascending free-flow time, random (several seeds, averaged) and, as a negative control, descending maximum. `benchmarks/edge_order.py`.

<!-- BEGIN:edge_order -->
{table}
<!-- END:edge_order -->

"""



def all_discs_section(df: pd.DataFrame) -> str:
    real_only(df, "all_discs")
    assert (df.verified_failing_scenarios == 0).all(), "a repaired spanner failed re-verification: bug"
    assert (df.family_forced_lb_pct <= df.repaired_retention_pct + 1e-9).all()
    df = df.copy()
    if "t_fault" not in df.columns:
        df["t_fault"] = df["t"]
    df = df.sort_values(["city", "t_fault", "radius_m"]).reset_index(drop=True)
    rows = [[r.city, f"{r.arcs}", f"{r.radius_m:g}", f"{r.t_fault:g}", f"{r.distinct_scenarios}", f"{r.nominal_fails_area_pct:.1f}",
             f"{int(r.worst_violating_arcs)}", f"{r.nominal_retention_pct:.1f}", f"{r.repaired_retention_pct:.1f}",
             f"{int(r.added_arcs)}", f"{r.nominal_forced_lb_pct:.1f}", f"{r.family_forced_lb_pct:.1f}"]
            for r in df.itertuples()]
    table = md(["City", "Arcs", "Disc radius (m)", "Stretch after a fault", "Distinct fault scenarios", "Centres where the nominal spanner fails (% of area)",
                "Worst scenario: violating arcs", "Nominal retention (%)", "Retention of the certified spanner (%)",
                "Arcs added", "Nominal forced bound (%)", "Forced bound over all discs (%)"], rows)
    t = df.t.iloc[0]
    strict = df[df.t_fault <= df.t + 1e-9]
    relaxed = df[df.t_fault > df.t + 1e-9]
    full = strict[strict.family_forced_lb_pct >= 99.995]
    extra = ""
    if len(relaxed):
        per = relaxed.groupby("t_fault").agg(lb=("family_forced_lb_pct", "mean"), ret=("repaired_retention_pct", "mean"))
        extra = (" With a larger stretch allowed after the fault, the mean forced bound and certified retention over the rows are "
                 + "; ".join(f"{k:g}: {v.lb:.1f}% and {v.ret:.1f}%" for k, v in per.iterrows()) + ".")
    df = strict if len(strict) else df
    cover = (f"In {len(full)} of {len(df)} rows with the strict definition (stretch after a fault equal to t) the arcs forced in "
             f"some G-F cover every arc")
    cover += ("; for those rows no edge-subset spanner that is tolerant to all discs of the radius can omit any arc, so the "
              "certified spanner is the whole network and the exact lower bound equals it." if len(full) else ".")
    sentence = (cover + f" The spanner built for the intact network fails the bound for {df.nominal_fails_area_pct.min():.1f}% to "
                f"{df.nominal_fails_area_pct.max():.1f}% of the centres (radii {df.radius_m.min():g} to {df.radius_m.max():g} m, "
                f"{df.city.nunique()} cities), and the worst scenario leaves between {int(df.worst_violating_arcs.min())} and "
                f"{int(df.worst_violating_arcs.max())} arcs without a detour." + extra)
    return f"""### 5v. All disc faults of one radius (exact)

Section 5l samples faults. Here the family is every disc of radius r with its centre anywhere. The set of street segments removed by a disc depends only on which offset regions N_r(e) = {{c : dist(c, e) <= r}} contain its centre, so the continuous family has finitely many distinct scenarios, namely the distinct arc sets of the cells of the arrangement of these regions (computed with `shapely`). Three exact results follow at t = {t:g} (`benchmarks/all_discs.py`): (1) the area of centres for which the spanner built for the intact network fails the delay bound in G-F, and the worst scenario; (2) a spanner obtained by adding every violating arc of every scenario, which is a t-spanner of G-F at all 24 hours for every disc of the radius (adding arcs never breaks a satisfied pair, so one pass suffices; the result is re-verified from scratch); (3) the union over all scenarios of the arcs forced in G-F, a lower bound for any edge-subset spanner that is fault tolerant for the family.

<!-- BEGIN:all_discs -->
{table}

{sentence}
<!-- END:all_discs -->

Scope: discs of one radius per row (other radii and other shapes are separate families); a segment is removed if its stored polyline touches the disc; the guarantee holds for all centres except a set of measure zero (discs tangent to a polyline) and up to the polygonal approximation of the offset regions (relative error of the radius below 1e-3); area percentages assume a uniformly distributed centre over the filled footprint of the offset regions; the traffic profile is the parametric model.

"""


def oneway_dose_section(df: pd.DataFrame) -> str:
    real_only(df, "oneway_dose")
    g = df.groupby(["t", "dose"]).mean(numeric_only=True).reset_index()
    rows = [[f"{r.t:g}", f"{100 * r.dose:.0f}%", f"{r.forced_original_pct:.1f}", f"{r.retention_original_pct:.1f}",
             f"{r.forced_all_pct:.1f}", f"{r.retention_all_pct:.1f}"] for r in g.itertuples()]
    table = md(["t", "One-way arcs made two-way", "Forced, original arcs (%)", "Retention, original arcs (%)",
                "Forced, all arcs (%)", "Retention, all arcs (%)"], rows)
    mono = []
    for t, d in df.groupby("t"):
        ok = tot = 0
        for _, c in d.groupby("city"):
            c = c.sort_values("dose")
            tot += 1
            ok += int((c.forced_original_pct.diff().dropna() <= 1e-9).all())
        mono.append(f"at t = {t:g} the forced share of the original arcs never increases with the dose in {ok} of {tot} cities")
    return f"""### 5w. Dose-response for one-way streets

One-way arcs are made two-way gradually, in a fixed random order that is nested across doses, and the forced share and the retention are recomputed at each dose, over the original arcs (fixed denominator) and over all arcs. `benchmarks/oneway_dose.py`. Mean over {df.city.nunique()} cities.

<!-- BEGIN:oneway_dose -->
{table}

{'; '.join(mono)}.
<!-- END:oneway_dose -->

Scope: a counterfactual (the reverse arc gets the same hourly travel times), one random order per city.

"""



def cut_structure_section(df: pd.DataFrame) -> str:
    real_only(df, "cut_structure")
    cols = ["bridge_pct", "vertex_cut_pct", "street_cut_pct", "robust_pct"]
    assert ((df[cols].sum(axis=1) - 100.0).abs() < 1e-6).all(), "categories must partition the arcs: bug"
    rows = [[r.city, f"{r.arcs}", f"{r.bridge_pct:.1f}", f"{r.vertex_cut_pct:.1f}", f"{r.street_cut_pct:.1f}",
             f"{r.robust_pct:.1f}", f"{r.in_some_single_failure_cut_pct:.1f}"] for r in df.itertuples()]
    table = md(["City", "Arcs", "Bridges (%)", "Intersection cut (%)", "Street cut (%)", "Robust (%)",
                "Lose every path after some single failure (%)"], rows)
    return f"""### 5x. Disconnection or stretch? Single failures

For an arc e = (u, v), is there one street or one intersection whose removal leaves no directed u-to-v path in G-e? Such an arc is in every fault-tolerant spanner for any allowed stretch. (The two arcs of a two-way street fail together.) `benchmarks/cut_structure.py`.

<!-- BEGIN:cut_structure -->
{table}

Across {len(df)} cities, {df.in_some_single_failure_cut_pct.min():.1f}% to {df.in_some_single_failure_cut_pct.max():.1f}% of the arcs lose every path after some single street or intersection failure; the other {df.robust_pct.min():.1f}% to {df.robust_pct.max():.1f}% keep a path after any single failure. For the robust arcs, being forced under a fault (section 5v) can only come from path length, not from disconnection.
<!-- END:cut_structure -->

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
            ("directedness", "directedness.csv", directedness_section), ("lightness", "lightness.csv", lightness_section),
            ("edge_order", "edge_order.csv", edge_order_section), ("all_discs", "all_discs.csv", all_discs_section), ("oneway_dose", "oneway_dose.csv", oneway_dose_section),
            ("cut_structure", "cut_structure.csv", cut_structure_section),
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
