#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional figures, drawn from the raw result files (same style as make_figures.py).

  python benchmarks/make_figures_extra.py

  fig0_overview            the problem on a small illustrative network (computed with the repository's own construction)
  fig7_scale_12cities      3.5 km networks of all twelve cities: retention vs the forced-edge bound
  fig8_optimality_gap      exact optimum vs greedy vs union baseline on small random instances
  fig9_offgrid_cost        cost of the continuous-time guarantee on METR-LA and PEMS-BAY
  fig10_severity           sensitivity of the results to the severity of the congestion model
  fig11_certification_matrix   which method keeps the bound on measured speeds
  fig12_edge_anatomy       how much alternative each street segment has
  fig13_scaling            construction time and probes per edge (four subnetworks of one city)
  fig14_scaling_large      scaling study over many sizes, synthetic and real
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_figures import C, _panel_label, _read, _save  # noqa: E402  (also applies the shared plot style)

NAVY, TEAL, ORANGE, PURPLE, RED, GREY = C["DelayBounded"], C["UnionOfSnapshots"], C["StaticFreeFlow"], C["MeanTemporal"], "#c0392b", "#9aa3ad"
CONGESTION = LinearSegmentedColormap.from_list("congestion", ["#2e9e5b", "#f1c40f", "#e67e22", "#c0392b"])


# --------------------------------------------------------------------------------------------------
# fig0: illustrative network
# --------------------------------------------------------------------------------------------------
def toy_network(seed=7, n=16):
    import networkx as nx
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2)) * [4.0, 2.6]
    tri = Delaunay(pts)
    und = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            u, v = sorted((int(s[a]), int(s[b])))
            if np.hypot(*(pts[u] - pts[v])) < 1.5:
                und.add((u, v))
    centre = np.array([2.0, 1.3])
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    hours = np.arange(24)
    peak = 0.55 * np.exp(-0.5 * ((hours - 17.0) / 2.2) ** 2) + 0.35 * np.exp(-0.5 * ((hours - 8.0) / 1.6) ** 2)
    for u, v in sorted(und):
        length = float(np.hypot(*(pts[u] - pts[v])))
        mid = (pts[u] + pts[v]) / 2
        load = np.exp(-np.hypot(*(mid - centre)) / 1.2) * (0.8 + 0.4 * rng.random())
        tt = [length * (1.0 + 3.2 * load * p) for p in peak]
        for a, b in ((u, v), (v, u)):
            G.add_edge(a, b, tt=tt, free_flow=length, length=length)
    return G, pts


def fig0(out, t=2.0):
    import run_spanner_benchmark as core
    import optimality_gap as og
    G, pts = toy_network()
    H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    assert core.temporal_stretch_certificate(G, H, t)["certified"], "the illustrative spanner must be certified"
    edges, cost = og.edge_arrays(G)
    n = G.number_of_nodes()
    forced = {edges[i] for i in og.forced_edges(n, edges, cost, t, og.all_pairs(n, edges, cost))}
    und = sorted({tuple(sorted(e)) for e in G.edges()})

    def kept(e):
        return H.has_edge(*e) or H.has_edge(*e[::-1])

    def is_forced(e):
        return e in forced or e[::-1] in forced

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.7))
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.01, right=0.99, wspace=0.02)
    for ax, hour, title in zip(axes[:2], (3, 17), ("03:00, free-flowing", "17:00, peak hour")):
        for u, v in und:
            ratio = G[u][v]["tt"][hour] / G[u][v]["free_flow"]
            ax.plot(*zip(pts[u], pts[v]), color=CONGESTION(min(1.0, (ratio - 1.0) / 2.4)), lw=2.8, solid_capstyle="round", zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], s=36, color="white", edgecolor="#2c3e50", linewidth=1.2, zorder=3)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    for u, v in und:
        e = (u, v)
        if not kept(e):
            axes[2].plot(*zip(pts[u], pts[v]), color=GREY, lw=1.3, ls=(0, (3, 3)), zorder=1)
        else:
            axes[2].plot(*zip(pts[u], pts[v]), color=NAVY if is_forced(e) else TEAL, lw=3.2, solid_capstyle="round", zorder=2)
    axes[2].scatter(pts[:, 0], pts[:, 1], s=36, color="white", edgecolor="#2c3e50", linewidth=1.2, zorder=3)
    nk, nt = sum(kept(e) for e in und), len(und)
    axes[2].set_title(f"Spanner H at t = {t:g}: {nk} of {nt} segments kept", fontsize=11.5, fontweight="bold", pad=8)
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(pts[:, 0].min() - 0.15, pts[:, 0].max() + 0.15)
        ax.set_ylim(pts[:, 1].min() - 0.15, pts[:, 1].max() + 0.15)
    cax = fig.add_axes([0.07, 0.115, 0.40, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CONGESTION, norm=plt.Normalize(1.0, 3.4)), cax=cax, orientation="horizontal")
    cb.set_label("travel time relative to free flow", fontsize=9)
    fig.legend(handles=[Line2D([0], [0], color=NAVY, lw=3.2, label="kept, forced (every spanner needs it)"),
                        Line2D([0], [0], color=TEAL, lw=3.2, label="kept, not forced"),
                        Line2D([0], [0], color=GREY, lw=1.3, ls=(0, (3, 3)), label="removed: a certified detour exists in every hour")],
               loc="lower center", bbox_to_anchor=(0.80, 0.0), fontsize=8.6, ncol=1, handlelength=2.2)
    fig.suptitle("Delay-bounded spanner: keep fewer streets, guarantee travel time in every hour of the day", fontsize=12.5, fontweight="bold", y=0.97)
    _save(fig, out, "fig0_overview")


# --------------------------------------------------------------------------------------------------
# fig7: twelve cities at 3.5 km
# --------------------------------------------------------------------------------------------------
def fig7(res, out):
    d = _read(res, "large_scale_3500m.csv")
    if d is None:
        return
    assert d.delay_bounded_certified.all(), "a spanner of the larger-radius run is not certified"
    d["gap"] = d.delay_bounded_retention_pct - d.forced_lower_bound_pct
    ts = sorted(d.t.unique())
    size = d.drop_duplicates("city").set_index("city")
    order = list(size.sort_values("edges").index)
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.1, 1.25], wspace=0.14)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (ax, t) in enumerate(zip(axes, ts)):
        s = d[d.t == t].set_index("city").loc[order]
        y = np.arange(len(order))
        ax.hlines(y, s.forced_lower_bound_pct, s.delay_bounded_retention_pct, color=NAVY, alpha=0.35, lw=3)
        ax.scatter(s.forced_lower_bound_pct, y, s=46, facecolor="white", edgecolor=NAVY, linewidth=1.6, zorder=3, label="forced-edge lower bound")
        ax.scatter(s.delay_bounded_retention_pct, y, s=46, color=NAVY, zorder=4, label="delay-bounded greedy (certified)")
        unc = s[~s.static_ff_certified]
        ax.scatter(unc.static_ff_retention_pct, [order.index(c) for c in unc.index], marker="^", s=34, color=ORANGE, zorder=2, label="static free-flow (not certified)")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  ({int(size.loc[c, 'edges']) / 1000:.0f}k edges)" for c in order] if i == 0 else [], fontsize=8.8)
        lo = min(s.forced_lower_bound_pct.min(), s.delay_bounded_retention_pct.min(), s.static_ff_retention_pct.min())
        ax.set_xlim(np.floor(lo) - 1.0, 100.6)
        ax.set_xlabel("edges kept (%)")
        ax.set_title(f"t = {t:g}", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.36, -0.02))
    _panel_label(axes[0], "a")
    hax = fig.add_subplot(gs[0, 4])
    piv = d.pivot(index="city", columns="t", values="gap").loc[order]
    im = hax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(5.0, piv.values.max()))
    hax.set_xticks(range(len(ts)))
    hax.set_xticklabels([f"t = {t:g}" for t in ts])
    hax.set_yticks(range(len(order)))
    hax.set_yticklabels([])
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            v = piv.values[r, c]
            hax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8.6, color="white" if v > 0.55 * piv.values.max() else "#1b2b3a")
    hax.set_title("gap to the lower bound (points)", fontsize=10.5, fontweight="bold")
    hax.grid(False)
    for sp in hax.spines.values():
        sp.set_visible(False)
    _panel_label(hax, "b")
    fig.suptitle("3.5 km street networks of twelve cities: every spanner certified, retention close to the structural lower bound",
                 fontsize=12, fontweight="bold", y=1.0)
    _save(fig, out, "fig7_scale_12cities")


# --------------------------------------------------------------------------------------------------
# fig8: exact optimum on small instances
# --------------------------------------------------------------------------------------------------
def fig8(res, out):
    d = _read(res, "optimality_gap_raw.csv")
    if d is None:
        return
    d["gap"] = d.greedy - d.optimum
    ts = sorted(d.t.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.24))
    ax = axes[0]
    series = [("forced", "forced edges (lower bound)", "#bdc3c7"), ("optimum", "exact optimum", PURPLE), ("greedy", "delay-bounded greedy", NAVY),
              ("union", "union of per-hour greedy", TEAL)]
    mk = {"forced": "v", "optimum": "D", "greedy": "o", "union": "s"}
    for col, lab, colr in series:
        vals = [100 * (d[d.t == t][col] / d[d.t == t].m).mean() for t in ts]
        ax.plot(range(len(ts)), vals, "-" + mk[col], color=colr, lw=2.2, ms=7, label=lab, ls="--" if col == "forced" else "-")
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"t = {t:g}" for t in ts])
    ax.set_xlim(-0.25, len(ts) - 0.75)
    ax.set_ylabel("edges kept, % of the network (mean)")
    ax.legend(loc="lower left", fontsize=8.4)
    ax.set_title("Mean edges kept", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "a")
    ax = axes[1]
    labels = ["0", "1", "2", "3 or more"]
    cols = [TEAL, "#7fb3d5", ORANGE, RED]
    bottom = np.zeros(len(ts))
    for k, lab in enumerate(labels):
        share = np.array([100 * (np.minimum(d[d.t == t].gap, 3) == k).mean() for t in ts])
        ax.bar(range(len(ts)), share, 0.62, bottom=bottom, color=cols[k], label=f"{lab} extra edge{'s' if lab != '1' else ''}")
        for i, v in enumerate(share):
            if v > 6:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
        bottom += share
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"t = {t:g}" for t in ts])
    ax.set_ylabel("instances (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.2)
    ax.set_title("Greedy excess over the optimum", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "b")
    inst = d.drop_duplicates("inst")
    fig.suptitle(f"Exact optimum on {len(inst)} small random time-varying digraphs ({int(inst.m.min())} to {int(inst.m.max())} edges)",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out, "fig8_optimality_gap")


# --------------------------------------------------------------------------------------------------
# fig9: continuous-time guarantee
# --------------------------------------------------------------------------------------------------
def fig9(res, out):
    frames = {"METR-LA": _read(res, "offgrid_guarantee_metrla.csv"), "PEMS-BAY": _read(res, "offgrid_guarantee_pemsbay.csv")}
    if any(v is None for v in frames.values()):
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), sharex="col", gridspec_kw=dict(hspace=0.14, wspace=0.26))
    for j, (name, d) in enumerate(frames.items()):
        ts = d["T"].values
        x = np.arange(len(ts))
        ax = axes[0, j]
        ax.plot(x, d.grid_only_edges_pct, "-o", color=ORANGE, lw=2, ms=7, label="certified on the 24 grid hours only")
        ax.plot(x, d.peredge_edges_pct, "-s", color=NAVY, lw=2.2, ms=7, label="per-edge continuous-time certificate (Theorem 3b)")
        ok = d.robust_certified_on_subgrid.astype(bool).values
        ax.plot(x[ok], d.robust_edges_pct.values[ok], "D", color=PURPLE, ms=8, label="global certificate (Theorem 3), where feasible")
        lo, hi = ax.get_ylim()
        ax.scatter(x[~ok], [lo + 0.04 * (hi - lo)] * int((~ok).sum()), marker="x", s=60, color=PURPLE, linewidth=2, zorder=5,
                   label="global certificate infeasible at this T")
        ax.set_ylim(lo, hi)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("edges kept (%)" if j == 0 else "")
        _panel_label(ax, "ab"[j])
        ax = axes[1, j]
        for col, lab, colr, mk in (("grid_only_worst_offgrid", "grid hours only", ORANGE, "o"), ("peredge_worst_offgrid", "per-edge certificate", NAVY, "s")):
            ax.vlines(x + (0.06 if mk == "s" else -0.06), 0.8, d[col] / d["T"], color=colr, lw=1.4, alpha=0.5)
            ax.plot(x + (0.06 if mk == "s" else -0.06), d[col] / d["T"], mk, color=colr, ms=8)
        ax.axhline(1.0, color=RED, lw=1.3, ls="--")
        ax.text(len(ts) - 0.55, 1.004, "bound", color=RED, fontsize=8, ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T = {t:g}" for t in ts])
        ax.set_xlim(-0.4, len(ts) - 0.6)
        ax.set_ylim(0.8, 1.04)
        ax.set_ylabel("worst sampled stretch / T" if j == 0 else "")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Between the grid hours: price and effect of the continuous-time guarantee", fontsize=12, fontweight="bold", y=0.95)
    _save(fig, out, "fig9_offgrid_cost")


# --------------------------------------------------------------------------------------------------
# fig10: severity of the congestion model
# --------------------------------------------------------------------------------------------------
def fig10(res, out):
    d = _read(res, "severity_sensitivity.csv")
    if d is None:
        return
    assert d.delay_bounded_certified.all(), "a spanner of the severity study is not certified"
    cities = list(d.city.unique())
    pal = dict(zip(cities, [NAVY, TEAL, ORANGE, PURPLE]))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw=dict(wspace=0.26))
    for ax, t in zip(axes[:2], (1.5, 2.0)):
        for c in cities:
            s = d[(d.city == c) & (d.t == t)].sort_values("mean_edge_slowdown")
            ax.plot(s.mean_edge_slowdown, s.delay_bounded_retention_pct, "-o", color=pal[c], lw=2, label=c)
            ax.plot(s.mean_edge_slowdown, s.forced_lower_bound_pct, "--", color=pal[c], lw=1.2, alpha=0.8)
        ax.set_xlabel("mean edge slowdown at the peak")
        ax.set_ylabel("edges kept (%)" if t == 1.5 else "")
        ax.set_title(f"t = {t:g}", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=8.4, title="solid: greedy, dashed: lower bound", title_fontsize=8, loc="lower right")
    _panel_label(axes[0], "a")
    ax = axes[2]
    for c in cities:
        s = d[(d.city == c) & (d.t == 1.5)].sort_values("mean_edge_slowdown")
        ax.plot(s.mean_edge_slowdown, s.static_ff_violating_edges, "-^", color=pal[c], lw=2)
    ax.set_xlabel("mean edge slowdown at the peak")
    ax.set_ylabel("edges violating the bound")
    ax.set_title("static free-flow baseline, t = 1.5", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "c")
    fig.suptitle("Sensitivity to the severity of the congestion model: all delay-bounded spanners stay certified", fontsize=12, fontweight="bold", y=1.03)
    _save(fig, out, "fig10_severity")


# --------------------------------------------------------------------------------------------------
# fig11: certification matrix on measured speeds
# --------------------------------------------------------------------------------------------------
def fig11(res, out):
    from matplotlib.colors import ListedColormap
    frames = {"METR-LA": _read(res, "metrla_table.csv"), "PEMS-BAY": _read(res, "pemsbay_table.csv")}
    if any(v is None for v in frames.values()):
        return
    methods = [("DelayBounded", "delay-bounded greedy"), ("UnionOfSnapshots", "union of per-hour greedy"),
               ("MeanTemporal", "mean-temporal weights"), ("StaticFreeFlow", "static free-flow weights")]
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 6.2), gridspec_kw=dict(hspace=0.42))
    for ax, (name, d) in zip(axes, frames.items()):
        ts = sorted(d.t.unique())
        ok = np.array([[bool(d[(d.method == m) & (d.t == t)].certified.iloc[0]) for t in ts] for m, _ in methods])
        ax.imshow(ok.astype(int), cmap=ListedColormap(["#f2c4be", "#bfe5d3"]), aspect="auto", vmin=0, vmax=1)
        for i, (m, _) in enumerate(methods):
            for j, t in enumerate(ts):
                r = d[(d.method == m) & (d.t == t)].iloc[0]
                ax.text(j, i - 0.12, f"{r.edges_kept_pct:.1f}%", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#14304d")
                ax.text(j, i + 0.24, "certified" if r.certified else f"{int(r.violating_edges)} edge{'' if int(r.violating_edges) == 1 else 's'} violate{'s' if int(r.violating_edges) == 1 else ''}", ha="center", va="center", fontsize=7.6,
                        color="#1b6b49" if r.certified else "#9b2c20")
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"t = {t:g}" for t in ts])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([lab for _, lab in methods], fontsize=9.2)
        ax.set_xticks(np.arange(-0.5, len(ts), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
    fig.suptitle("Who keeps the bound on measured speeds: edges kept (bold) and certification result per method and t", fontsize=12, fontweight="bold", y=0.97)
    _save(fig, out, "fig11_certification_matrix")


# --------------------------------------------------------------------------------------------------
# fig12: anatomy of the edges
# --------------------------------------------------------------------------------------------------
def fig12(res, out):
    d = _read(res, "retention_edge_ratios.csv")
    if d is None:
        return
    bins = [("no detour at all", lambda r: np.isinf(r)), ("detour ratio above 5", lambda r: np.isfinite(r) & (r > 5)),
            ("3 to 5", lambda r: (r > 3) & (r <= 5)), ("2 to 3", lambda r: (r > 2) & (r <= 3)), ("1.5 to 2", lambda r: (r > 1.5) & (r <= 2)),
            ("1.1 to 1.5", lambda r: (r > 1.1) & (r <= 1.5)), ("at most 1.1", lambda r: r <= 1.1)]
    cols = ["#0b1b2e", "#1f3a93", "#3b6fb6", "#6aa3d6", "#17a398", "#7fd1b9", "#d5f0e6"]
    shares = {}
    for city, g in d.groupby("city"):
        r = g.max_ratio.values.astype(float)
        sh = np.array([100 * b(r).mean() for _, b in bins])
        assert abs(sh.sum() - 100) < 1e-6, "the ratio classes do not cover every edge"
        shares[city] = sh
    order = sorted(shares, key=lambda c: -shares[c][0])
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    left = np.zeros(len(order))
    for k, (lab, _) in enumerate(bins):
        vals = np.array([shares[c][k] for c in order])
        ax.barh(range(len(order)), vals, left=left, color=cols[k], label=lab, height=0.72, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v >= 5:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8.2, color="white" if k < 4 else "#14304d")
        left += vals
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the edges (%)")
    ax.legend(title="best detour, worst hour, relative to the edge's own cost", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8.6, title_fontsize=9)
    ax.set_title("Most street segments have no cheap alternative: they are forced for every small t", fontsize=12, fontweight="bold")
    _save(fig, out, "fig12_edge_anatomy")


# --------------------------------------------------------------------------------------------------
# fig13: construction cost
# --------------------------------------------------------------------------------------------------
def fig13(res, out):
    import json
    path = os.path.join(os.path.dirname(res), "spanner_report.json")
    if not os.path.exists(path):
        return
    sc = pd.DataFrame(json.load(open(path))["scaling_study"])
    slope, icpt = np.polyfit(np.log(sc.edges), np.log(sc.seconds), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw=dict(wspace=0.28))
    ax = axes[0]
    ax.loglog(sc.edges, sc.seconds, "o-", color=NAVY, lw=2.2, ms=8)
    xs = np.array([sc.edges.min(), sc.edges.max()])
    ax.loglog(xs, np.exp(icpt) * xs ** slope, "--", color=GREY, lw=1.4, label=f"fitted growth: edges^{slope:.2f}")
    ax.set_xlabel("edges of the subnetwork")
    ax.set_ylabel("construction time (s)")
    ax.legend(fontsize=9)
    ax.set_title("Time against network size", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "a")
    ax = axes[1]
    ax.semilogx(sc.edges, sc.probes_per_edge, "s-", color=TEAL, lw=2.2, ms=8)
    ax.set_ylim(0, max(2.0, sc.probes_per_edge.max() * 1.3))
    ax.set_xlabel("edges of the subnetwork")
    ax.set_ylabel("bounded Dijkstra runs per edge")
    ax.set_title("Probes per edge", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "b")
    fig.suptitle("Construction cost on growing subnetworks of the focus city", fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out, "fig13_scaling")


# --------------------------------------------------------------------------------------------------
# fig14: scaling study
# --------------------------------------------------------------------------------------------------
def fig14(res, out):
    path = os.path.join(res, "scaling_large.csv")
    if not os.path.exists(path):
        return
    from scaling_fit import fit_exponent
    d = pd.read_csv(path)
    d = d[d.t == d.t.min()]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), gridspec_kw=dict(wspace=0.3))
    kinds = [("synthetic", "thinned Delaunay networks", NAVY, "o"), ("real", "OpenStreetMap networks", ORANGE, "s")]
    ax = axes[0]
    for kind, lab, colr, mk in kinds:
        g = d[d.kind == kind].sort_values("edges")
        if len(g) < 4:
            continue
        slope, lo, hi = fit_exponent(g.edges, g.seconds)
        ax.loglog(g.edges, g.seconds, mk, color=colr, ms=7, alpha=0.9, label=f"{lab}: edges^{slope:.2f} ({lo:.2f} to {hi:.2f})")
        xs = np.array([g.edges.min(), g.edges.max()])
        ax.loglog(xs, np.exp(np.polyval(np.polyfit(np.log(g.edges), np.log(g.seconds), 1), np.log(xs))), "-", color=colr, lw=1.4, alpha=0.7)
    xs = np.array([d.edges.min(), d.edges.max()])
    ref = d.seconds.iloc[d.edges.argmin()] * (xs / d.edges.min())
    ax.loglog(xs, ref, ":", color=GREY, lw=1.4, label="linear reference")
    ax.set_xlabel("edges")
    ax.set_ylabel("construction time (s)")
    ax.set_ylim(top=ax.get_ylim()[1] * 40)  # head-room so the legend does not cover data
    ax.legend(fontsize=7.2, loc="upper left")
    ax.set_title(f"Time against size (t = {d.t.min():g})", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "a")
    ax = axes[1]
    for kind, lab, colr, mk in kinds:
        g = d[d.kind == kind].sort_values("edges")
        if len(g):
            ax.semilogx(g.edges, g.settled_per_probe, ("-" if kind == "synthetic" else "") + mk, color=colr, ms=6, lw=1.6)
    ax.set_ylim(0, max(10, d.settled_per_probe.max() * 1.3))
    ax.set_xlabel("edges")
    ax.set_ylabel("nodes settled per probe")
    ax.set_title("Work per probe", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "b")
    ax = axes[2]
    for kind, lab, colr, mk in kinds:
        g = d[d.kind == kind].sort_values("edges")
        if len(g):
            ax.semilogx(g.edges, g.probes_per_edge, ("-" if kind == "synthetic" else "") + mk, color=colr, ms=6, lw=1.6)
    ax.set_ylim(0, max(2.0, d.probes_per_edge.max() * 1.3))
    ax.set_xlabel("edges")
    ax.set_ylabel("probes per edge")
    ax.set_title("Probes per edge", fontsize=10.5, fontweight="bold")
    _panel_label(ax, "c")
    fig.suptitle("Scaling of the construction: time, and the machine-independent work behind it", fontsize=12, fontweight="bold", y=1.03)
    _save(fig, out, "fig14_scaling_large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "benchmarks"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "figures"))
    a = ap.parse_args()
    fig0(a.out_dir)
    fig7(a.results_dir, a.out_dir)
    fig8(a.results_dir, a.out_dir)
    fig9(a.results_dir, a.out_dir)
    fig10(a.results_dir, a.out_dir)
    fig11(a.results_dir, a.out_dir)
    fig12(a.results_dir, a.out_dir)
    fig13(a.results_dir, a.out_dir)
    fig14(a.results_dir, a.out_dir)


if __name__ == "__main__":
    main()
