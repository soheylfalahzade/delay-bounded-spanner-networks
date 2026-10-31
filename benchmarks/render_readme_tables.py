"""Render the result tables of README.md (sections 5b-5d) from the raw files in results/benchmarks/.

Blocks sit between  <!-- BEGIN:name -->  and  <!-- END:name -->  and are generated here:

    python benchmarks/render_readme_tables.py            # rewrite the blocks in README.md
    python benchmarks/render_readme_tables.py --check    # exit 1 if README.md differs from the data

    literature   results/benchmarks/{static_reduction,theorem1,general_graph,metrla,pemsbay}_*
    retention    results/benchmarks/retention_diagnostics.csv
    large_t      results/benchmarks/retention_large_t.csv, retention_morphology_correlations.csv

Statements such as "certified at every t" are asserted against the data before they are printed, so a
changed result stops the renderer instead of leaving a stale sentence in the README.
"""
import argparse
import json
import math
import os
import re
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEN = os.path.join(ROOT, "results", "benchmarks")
README = os.path.join(ROOT, "README.md")


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def js(name):
    with open(os.path.join(BEN, name), encoding="utf-8") as f:
        return json.load(f)


def sensor_row(ds, label, source):
    t = pd.read_csv(os.path.join(BEN, f"{ds}_table.csv"))
    cs = js(f"{ds}_summary.json")["congestion_shape"]
    g = lambda m, col: t[t.method == m].set_index("t")[col]  # noqa: E731
    db = g("DelayBounded", "certified")
    assert db.all(), f"{ds}: delay-bounded is not certified at every t"
    ts = sorted(t.t.unique())
    kept = lambda m, x: f"{g(m, 'edges_kept_pct')[x]:.1f}%"  # noqa: E731
    sf, mt = t[t.method == "StaticFreeFlow"], t[t.method == "MeanTemporal"]
    assert not sf.certified.any(), f"{ds}: static free-flow certified somewhere"
    viol = lambda d: f"{int(d.violating_edges.min())}-{int(d.violating_edges.max())}"  # noqa: E731
    if not mt.certified.any():
        text = (f"delay-bounded: certified at every t in {{{ts[0]},...,{ts[-1]}}}; static free-flow and mean-temporal: "
                f"not certified ({int(min(sf.violating_edges.min(), mt.violating_edges.min()))}-"
                f"{int(max(sf.violating_edges.max(), mt.violating_edges.max()))} violating edges).")
    else:
        ok = mt[mt.certified].t.tolist()
        assert ok == [2.0], f"{ds}: mean-temporal certified at {ok}, text assumes only t = 2.0"
        text = (f"delay-bounded: certified at every t; static free-flow: not certified at any t ({viol(sf)} violating edges); "
                f"mean-temporal: certified only at t = 2.0 ({g('MeanTemporal', 'edges_kept_pct')[2.0]:.1f}% edges vs. "
                f"{g('DelayBounded', 'edges_kept_pct')[2.0]:.1f}% for delay-bounded).")
    text += (f" Edges kept vs. union-of-snapshots: {kept('DelayBounded', ts[0])} vs. {kept('UnionOfSnapshots', ts[0])} (t = {ts[0]}), "
             f"{kept('DelayBounded', ts[-1])} vs. {kept('UnionOfSnapshots', ts[-1])} (t = {ts[-1]})")
    shape = (f"Pearson r = {cs['pearson_r_normalised_hourly_speed']:.2f}"
             f"{' on normalised hourly mean speed' if ds == 'metrla' else ''}; peak/off-peak slowdown "
             f"{cs['measured_peak_to_offpeak_slowdown']:.2f}x measured vs. {cs['model_peak_to_offpeak_slowdown']:.2f}x modelled")
    return [[label, "Li et al. (ICLR 2018)", text], [f"Congestion-model shape vs. {source}", "this work", shape]]


def table_literature():
    st = js("static_reduction_summary.json")
    assert not st["mismatches"] and st["stretch_within_t"], "static reduction no longer matches the reference"
    n = st["instances_checked_against_reference"]
    th = js("theorem1_summary.json")
    ev = pd.read_csv(os.path.join(BEN, "theorem1_edges_vs_union_baseline.csv"))
    assert th["union_all_certified_and_bruteforce_ok"] and th["ours_all_certified_and_bruteforce_ok"]
    assert (ev.ours_vs_union_edge_ratio <= 1.0).all(), "delay-bounded keeps more edges than the union baseline somewhere"
    gg = js("general_graph_summary.json")
    assert gg["baswana_sen_stretch_ok"] and gg["greedy_stretch_ok"] and gg["delay_bounded_equals_reference"]
    g = pd.read_csv(os.path.join(BEN, "general_graph_table.csv"))
    pct = lambda s: f"{100 * s.min():.0f}-{100 * s.max():.0f}%"  # noqa: E731
    greedy = g[g.method == "PathGreedy(ref)"].edge_frac
    bs = g[g.method.str.startswith("BaswanaSen")].edge_frac
    r3 = ev[ev.t == 3.0].ours_vs_union_edge_ratio.iloc[0]
    raw = pd.read_csv(os.path.join(BEN, "theorem1_raw.csv"))
    diff = raw.ours_edges - raw.union_edges
    more = diff[diff > 0]
    if more.empty:
        union_text = f"also certified; the delay-bounded construction keeps equal or fewer edges in all {len(raw)} instances (about {r3:.2f}x at t = 3 on random digraphs)"
    else:
        union_text = (f"also certified; the delay-bounded construction keeps fewer edges on average (about {r3:.2f}x at t = 3 on random digraphs) "
                      f"but is not provably never larger: in {len(more)} of {len(raw)} random instances (t <= {raw.loc[more.index, 't'].max():g}) "
                      f"it kept {int(more.min())}-{int(more.max())} more edges")
    rows = [
        ["Reduction to the classical path-greedy spanner (constant profile), edge-for-edge against an independent reference; "
         "Yao/Theta theoretical stretch bounds", "Farshi & Gudmundsson (ESA 2005; WEA 2007); Althofer et al. (1993)",
         f"{n}/{n} instances identical; {st['yao_theta_bound_violations']} bound violations"],
        ["General graphs (ER, BA, random geometric) vs. Baswana-Sen (2k-1)-spanners",
         "Baswana & Sen (2007); Angrick et al. (arXiv:2107.02018)",
         f"stretch <= 2k-1 for all; greedy keeps {pct(greedy)} of edges vs. {pct(bs)} for Baswana-Sen on these graphs "
         "(range of the per-setting means)"],
        ["Exhaustive all-pairs, all-hours check of Theorem 1 on random time-varying digraphs, with negative controls", "this work",
         f"{th['soundness_violations']} soundness violations on {th['instances']} instances"],
        ["Certified baseline: union of per-hour static greedy spanners", "this work", union_text],
    ]
    rows += sensor_row("metrla", "Measured speeds, METR-LA (207 sensors, Los Angeles)", "METR-LA")
    rows += sensor_row("pemsbay", "Measured speeds, PEMS-BAY (325 sensors, SF Bay Area)", "PEMS-BAY")
    return md(["Check", "Protocol source", "Result"], rows)


def table_retention():
    d = pd.read_csv(os.path.join(BEN, "retention_diagnostics.csv"))
    ts = [1.1, 1.5, 2.0, 3.0, 5.0]
    g = d.groupby("t").agg(ret=("retention_pct", "mean"), forced=("forced_lower_bound_pct", "mean"),
                           share=("share_of_retention_forced", "mean"))
    rows = [[t, f"{g.forced[t]:.1f}", f"{g.ret[t]:.1f}", f"{g.share[t]:.3f}"] for t in ts]
    return (f"Mean over the {d.city.nunique()} OSM city cores (about 1.5 km radius, degree-2 contracted):\n\n" +
            md(["t", "forced lower bound (%)", "delay-bounded greedy retention (%)", "share of retained edges that are forced"], rows) +
            f"\n\nSource: `results/benchmarks/retention_diagnostics.csv`.")


def table_large_t():
    L = pd.read_csv(os.path.join(BEN, "retention_large_t.csv"))
    C = pd.read_csv(os.path.join(BEN, "retention_morphology_correlations.csv"))
    n = L.city.nunique()
    g = L.groupby("t")[["forced_pct", "retention_pct", "gap_pct"]].mean()
    rows = [[t, f"{r.forced_pct:.1f}", f"{r.retention_pct:.1f}", f"{r.gap_pct:.1f}"] for t, r in g.iterrows()]
    piv = L.pivot(index="city", columns="t", values="retention_pct")
    lo3, lo10 = piv[[t for t in piv.columns if t <= 3]].min().min(), piv.min().min()
    first = {c: next((t for t in piv.columns if piv.loc[c, t] <= 90), None) for c in piv.index}
    hit = {c: t for c, t in first.items() if t is not None}
    tmin, tmax = min(hit, key=hit.get), max(hit, key=hit.get)
    flat = [c for c, t in first.items() if t is None]
    sig = C[C["significant_BH_0.05"]]
    br = C[(C.descriptor == "bridge_pct") & (C.target == "retention_pct") & (C.t >= 2)]
    assert set(sig.descriptor) == {"bridge_pct"}, "a geometric descriptor is now significant; the text below is wrong"
    tmax_t = g.index.max()
    bullets = [
        f"- Retention stays above {int(lo3)}% for every city up to t = 3, and above {int(lo10)}% up to t = {tmax_t:g}; retention falls to 90% or below "
        f"at t = {hit[tmin]:g} ({tmin}) to t = {hit[tmax]:g} ({tmax}); " +
        ", ".join(f"{c} stays above {int(piv.loc[c].min())}% through t = {tmax_t:g}" for c in flat) + ".",
        f"- Across the {n} cities, retention correlates strongly with the share of arcs that have no alternative directed path "
        f"(bridge share; Spearman rho = {br.spearman_rho.min():.2f}-{br.spearman_rho.max():.2f} for t >= 2, BH-adjusted p <= {math.ceil(br.p_BH.max() * 1000) / 1000:.3f}). "
        "This is largely structural: such arcs are forced for every t, so the share is itself a lower bound on retention.",
        f"- None of the geometric morphology descriptors (orientation entropy, circuity, mean degree, edge-length CV) is significant after "
        f"Benjamini-Hochberg correction over {len(C)} tests (n = {int(C.n.iloc[0])} cities); the study has low power and reports no morphology effect.",
        f"- The gap between greedy retention and the forced bound grows with t ({g.gap_pct[tmax_t]:.1f} points at t = {tmax_t:g}). "
        "The bound is not claimed to be tight, so no optimality claim is made for the construction.",
    ]
    return (md(["t", "forced lower bound (%)", "greedy retention (%)", "gap (points)"], rows) + "\n\n" + "\n".join(bullets))


RENDERERS = {"literature": table_literature, "retention": table_retention, "large_t": table_large_t}


def render(text):
    for name, fn in RENDERERS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)\n.*?(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            sys.exit(f"README.md has no marker block for '{name}'")
        text = pat.sub(lambda m: m.group(1) + "\n" + fn() + "\n" + m.group(2), text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    current = open(README, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in current else "\n"
    rendered = render(current.replace("\r\n", "\n")).replace("\n", nl)
    if args.check:
        if rendered != current:
            print("README.md tables differ from the raw data; run: python benchmarks/render_readme_tables.py")
            sys.exit(1)
        print("README.md tables match the raw data.")
        return
    open(README, "w", encoding="utf-8", newline="").write(rendered)
    print("README.md tables rendered from results/benchmarks/")


if __name__ == "__main__":
    main()
