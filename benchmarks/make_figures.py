#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the claims in the README.
Reads the CSVs written by the benchmark scripts and writes PNG (README) + PDF (vector)
to results/figures/. Figures whose input CSV is missing are skipped with a message.

  python benchmarks/make_figures.py
  python benchmarks/make_figures.py --results-dir results/benchmarks --out-dir results/figures

  fig1_certification_real_traffic   METR-LA / PEMS-BAY: edges kept + attained stretch, certified vs not
  fig2_forced_edge_bound            Theorem 2 lower bound vs achieved retention (+ city x t gap matrix)
  fig3_morphology_matrix            Spearman matrices (BH-corrected) + retention vs bridge share
  fig4_baselines                    union-of-snapshots, Baswana-Sen and Yao/Theta comparisons
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C = {"DelayBounded": "#1f3a93", "UnionOfSnapshots": "#17a398", "StaticFreeFlow": "#e67e22", "MeanTemporal": "#8e44ad"}
LAB = {"DelayBounded": "Delay-bounded (ours)", "UnionOfSnapshots": "Union of per-hour greedy",
       "StaticFreeFlow": "Static free-flow", "MeanTemporal": "Mean-temporal"}
MK = {"DelayBounded": "o", "UnionOfSnapshots": "s", "StaticFreeFlow": "^", "MeanTemporal": "D"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220, "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[fig]", name)


def _read(res, name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        print(f"[skip] missing {name}")
        return None
    return pd.read_csv(p)


def _panel_label(ax, s):
    ax.text(-0.12, 1.07, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


# ---------------------------------------------------------------------------
def fig1(res, out):
    sets = [("metrla_table.csv", "METR-LA (207 sensors, Los Angeles)"), ("pemsbay_table.csv", "PEMS-BAY (325 sensors, SF Bay Area)")]
    data = [(_read(res, f), t) for f, t in sets]
    data = [(d, t) for d, t in data if d is not None]
    if not data:
        return
    fig, axes = plt.subplots(2, len(data), figsize=(5.8 * len(data), 7.0), squeeze=False)
    for j, (df, title) in enumerate(data):
        a1, a2 = axes[0, j], axes[1, j]
        for m in ["StaticFreeFlow", "MeanTemporal", "UnionOfSnapshots", "DelayBounded"]:
            d = df[df.method == m].sort_values("t")
            for ax, col in ((a1, "edges_kept_pct"), (a2, "attained_worst_edge_stretch")):
                ax.plot(d.t, d[col], color=C[m], lw=2.2 if m == "DelayBounded" else 1.4, alpha=0.95, zorder=3 if m == "DelayBounded" else 2)
                cert = d.certified.astype(bool).values
                ax.scatter(d.t[cert], d[col][cert], s=48, marker=MK[m], color=C[m], zorder=4, edgecolor="white", linewidth=0.8,
                           label=LAB[m] if (ax is a1 and cert.any()) else None)
                ax.scatter(d.t[~cert], d[col][~cert], s=48, marker=MK[m], facecolor="white", edgecolor=C[m], linewidth=1.6, zorder=4,
                           label=None)
        tmax = df.t.max()
        a2.plot([1, tmax], [1, tmax], ls="--", color="black", lw=1, label="requested bound  y = t")
        a2.fill_between([1, tmax], [1, tmax], [1, df.attained_worst_edge_stretch.max() * 1.1], color="#c0392b", alpha=0.06)
        a2.text(1.05, df.attained_worst_edge_stretch.max() * 1.02, "bound violated", color="#c0392b", fontsize=8.5, va="top")
        a1.set_title(title, loc="left", fontweight="bold")
        a1.set_ylabel("edges kept (%)")
        a2.set_ylabel("worst edge stretch over all 24 h")
        a2.set_xlabel("requested stretch bound  t")
        a1.set_xlabel("")
    h = [plt.Line2D([], [], color=C[m], marker=MK[m], lw=2, markersize=7, label=LAB[m]) for m in
         ["DelayBounded", "UnionOfSnapshots", "StaticFreeFlow", "MeanTemporal"]]
    h += [plt.Line2D([], [], color="gray", marker="o", ls="", markerfacecolor="gray", label="certified (filled)"),
          plt.Line2D([], [], color="gray", marker="o", ls="", markerfacecolor="white", markeredgewidth=1.5, label="not certified (hollow)"),
          plt.Line2D([], [], color="black", ls="--", label="requested bound y = t")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.suptitle("Measured traffic: certified constructions stay on the bound; uncertified baselines can exceed it",
                 fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, out, "fig1_certification_real_traffic")


# ---------------------------------------------------------------------------
def fig2(res, out):
    R = _read(res, "retention_large_t.csv")
    if R is None:
        return
    ts = sorted(R.t.unique())
    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    for city, g in R.groupby("city"):
        g = g.sort_values("t")
        ax.plot(g.t, g.retention_pct, color=C["DelayBounded"], lw=0.8, alpha=0.22)
        ax.plot(g.t, g.forced_pct, color="#c0392b", lw=0.8, alpha=0.22)
    mean = R.groupby("t")[["retention_pct", "forced_pct"]].mean()
    lo = R.groupby("t")[["retention_pct", "forced_pct"]].min()
    hi = R.groupby("t")[["retention_pct", "forced_pct"]].max()
    ax.fill_between(ts, lo.forced_pct, hi.forced_pct, color="#c0392b", alpha=0.10)
    ax.fill_between(ts, lo.retention_pct, hi.retention_pct, color=C["DelayBounded"], alpha=0.10)
    ax.plot(ts, mean.retention_pct, color=C["DelayBounded"], lw=2.6, marker="o", label="delay-bounded greedy (mean of 12 cities)")
    ax.plot(ts, mean.forced_pct, color="#c0392b", lw=2.6, marker="s", label="forced-edge lower bound, any algorithm (mean)")
    ax.fill_between(ts, mean.forced_pct, mean.retention_pct, color="#7f8c8d", alpha=0.25, label="gap (bound not claimed tight)")
    ax.set_xlabel("requested stretch bound  t")
    ax.set_ylabel("edges retained (%)")
    ax.set_ylim(30, 101)
    ax.set_title("Retention vs. forced-edge lower bound", loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.text(-0.11, 1.07, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.set_xticks(ts)
    ax.set_xticklabels([f"{t:g}" for t in ts])
    ax.annotate("shaded bands: min-max over cities\nthin lines: individual cities", xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=7.8, color="#555")

    ax2 = fig.add_subplot(gs[1])
    P = R.pivot(index="city", columns="t", values="gap_pct")
    P = P.loc[P[ts[-1]].sort_values().index]
    im = ax2.imshow(P.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(40, np.nanmax(P.values)))
    ax2.set_xticks(range(len(ts)))
    ax2.set_xticklabels([f"{t:g}" for t in ts])
    ax2.set_yticks(range(len(P)))
    ax2.set_yticklabels(P.index)
    ax2.grid(False)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]
            ax2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > 22 else "black")
    ax2.set_xlabel("requested stretch bound  t")
    ax2.set_title("Gap to the bound (percentage points)", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cb.set_label("points")
    ax2.text(-0.30, 1.07, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")
    _save(fig, out, "fig2_forced_edge_bound")


# ---------------------------------------------------------------------------
def fig3(res, out):
    C_ = _read(res, "retention_morphology_correlations.csv")
    Th = _read(res, "retention_thresholds_by_city.csv")
    R = _read(res, "retention_large_t.csv")
    if C_ is None:
        return
    desc = ["bridge_pct", "orientation_entropy", "circuity_mean", "mean_degree", "edge_length_cv"]
    names = {"bridge_pct": "bridge share (structural)", "orientation_entropy": "orientation entropy", "circuity_mean": "mean circuity",
             "mean_degree": "mean degree", "edge_length_cv": "edge-length CV"}
    has_c = Th is not None and R is not None
    fig = plt.figure(figsize=(16.5 if has_c else 11.5, 4.9))
    gs = fig.add_gridspec(1, 5 if has_c else 3, width_ratios=([1, 1, 0.045, 0.32, 1.15] if has_c else [1, 1, 0.045]), wspace=0.42)
    axA, axB, axCB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ts = sorted(C_.t.unique())
    im = None
    sig_desc = set()
    for ax, target, ttl in ((axA, "retention_pct", "Spearman rho with retention"), (axB, "gap_pct", "Spearman rho with gap to bound")):
        M = np.full((len(desc), len(ts)), np.nan)
        S = np.zeros_like(M, dtype=bool)
        for i, d in enumerate(desc):
            for j, t in enumerate(ts):
                r = C_[(C_.t == t) & (C_.target == target) & (C_.descriptor == d)]
                if len(r):
                    M[i, j] = r.spearman_rho.values[0]
                    S[i, j] = bool(r["significant_BH_0.05"].values[0])
                    if S[i, j]:
                        sig_desc.add(d)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"{t:g}" for t in ts])
        ax.set_yticks(range(len(desc)))
        ax.set_yticklabels([names[d] for d in desc] if ax is axA else [])
        ax.grid(False)
        for i in range(len(desc)):
            for j in range(len(ts)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if S[i, j] else ""), ha="center", va="center", fontsize=8,
                            fontweight="bold" if S[i, j] else "normal", color="white" if abs(M[i, j]) > 0.6 else "black")
        ax.set_xlabel("requested stretch bound  t")
        ax.set_title(ttl, loc="left", fontweight="bold")
    fig.colorbar(im, cax=axCB).set_label("Spearman rho")
    axA.text(-0.62, 1.09, "a", transform=axA.transAxes, fontsize=14, fontweight="bold", va="top")
    axB.text(-0.08, 1.09, "b", transform=axB.transAxes, fontsize=14, fontweight="bold", va="top")
    if has_c:
        ax = fig.add_subplot(gs[4])
        t0 = 3.0 if 3.0 in set(R.t) else sorted(R.t)[len(set(R.t)) // 2]
        d = R[R.t == t0].merge(Th[["city", "bridge_pct"]], on="city")
        ax.scatter(d.bridge_pct, d.retention_pct, s=55, color=C["DelayBounded"], edgecolor="white", zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r.city, (r.bridge_pct, r.retention_pct), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        if len(d) > 2:
            k = np.polyfit(d.bridge_pct, d.retention_pct, 1)
            xs = np.linspace(d.bridge_pct.min(), d.bridge_pct.max(), 20)
            ax.plot(xs, np.polyval(k, xs), color="#7f8c8d", lw=1.2, ls="--")
        ax.set_xlabel("arcs without any alternative directed path (%)")
        ax.set_ylabel(f"edges retained at t = {t0:g} (%)")
        ax.set_title("Retention follows the bridge share", loc="left", fontweight="bold")
        ax.text(-0.17, 1.09, "c", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    geo = sorted(sig_desc - {"bridge_pct"})
    ntests = len(C_)
    note = (f"* significant after Benjamini-Hochberg correction ({ntests} tests, n = 12 cities). "
            + ("Geometric descriptors significant after correction: " + ", ".join(names[x] for x in geo) + "."
               if geo else "No geometric morphology descriptor survives correction; the bridge share acts through forced edges (Theorem 2)."))
    fig.text(0.01, -0.03, note, fontsize=8, color="#444")
    _save(fig, out, "fig3_morphology_matrix")


# ---------------------------------------------------------------------------
def fig4(res, out):
    T1 = _read(res, "theorem1_edges_vs_union_baseline.csv")
    G_ = _read(res, "general_graph_table.csv")
    S_ = _read(res, "static_reduction_table.csv")
    panels = [x is not None for x in (T1, G_, S_)]
    if not any(panels):
        return
    fig, axes = plt.subplots(1, sum(panels), figsize=(5.0 * sum(panels), 4.4))
    axes = np.atleast_1d(axes)
    k = 0
    if T1 is not None:
        ax = axes[k]
        k += 1
        x = np.arange(len(T1))
        w = 0.36
        ax.bar(x - w / 2, T1.union_of_snapshots_edges, w, color=C["UnionOfSnapshots"], label=LAB["UnionOfSnapshots"])
        ax.bar(x + w / 2, T1.ours_edges, w, color=C["DelayBounded"], label=LAB["DelayBounded"])
        for i, r in T1.reset_index(drop=True).iterrows():
            ax.text(i + w / 2, r.ours_edges + 0.4, f"{r.ours_vs_union_edge_ratio:.2f}x", ha="center", fontsize=8, color=C["DelayBounded"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"t = {t:g}" for t in T1.t])
        ax.set_ylabel("edges (random time-varying digraphs)")
        ax.set_title("Joint construction vs. certified baseline", loc="left", fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.text(-0.14, 1.09, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    if G_ is not None:
        ax = axes[k]
        k += 1
        g = G_.copy()
        g["family"] = g.method.str.replace(r"-k\d", "", regex=True).replace({"PathGreedy(ref)": "Greedy", "DelayBounded(const)": "Greedy"})
        g = g[g.family.isin(["Greedy", "BaswanaSen"])]
        g = g.groupby(["graph", "t", "family"]).edge_frac.mean().reset_index()
        labels, vals = [], {"Greedy": [], "BaswanaSen": []}
        for (gr, t), sub in g.groupby(["graph", "t"]):
            labels.append(f"{gr}\nt = {int(t)}")
            for f in vals:
                vals[f].append(100 * float(sub[sub.family == f].edge_frac.mean()))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, vals["Greedy"], 0.4, color=C["DelayBounded"], label="Greedy (= ours, constant profile)")
        ax.bar(x + 0.2, vals["BaswanaSen"], 0.4, color="#95a5a6", label="Baswana-Sen (2k-1)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("edges kept (% of input graph)")
        ax.set_ylim(0, 125)
        ax.set_title("General graphs: sparsity at equal stretch bound", loc="left", fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper center", ncol=1)
        ax.text(-0.14, 1.09, "b", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    if S_ is not None:
        ax = axes[k]
        s = S_[(S_.kind == "uniform") & (S_.n == S_.n.max())].copy()
        s["family"] = s.method.str.extract(r"^(DelayBounded|PathGreedy|Yao|Theta)")[0]
        mk = {"DelayBounded": ("o", C["DelayBounded"]), "PathGreedy": ("x", "#c0392b"), "Yao": ("^", "#e67e22"), "Theta": ("s", "#8e44ad")}
        for fam, (m, col) in mk.items():
            d = s[s.family == fam]
            if not len(d):
                continue
            lab = {"DelayBounded": "Delay-bounded (const. profile)", "PathGreedy": "Path-greedy reference"}.get(fam, fam)
            if fam == "Yao":
                ax.scatter(d.lightness, d.stretch, marker="^", s=120, facecolor="none", edgecolor=col, linewidth=1.8, label=lab, zorder=4)
            elif fam == "PathGreedy":
                ax.scatter(d.lightness, d.stretch, marker="x", s=95, color=col, linewidth=1.8, label=lab, zorder=5)
            else:
                ax.scatter(d.lightness, d.stretch, marker=m, s=62, color=col, edgecolor="white", linewidth=0.8, label=lab, zorder=3)
        for _, r in s.iterrows():
            if r.family in ("Yao", "Theta"):
                ax.annotate(r.method, (r.lightness, r.stretch), fontsize=7.5, xytext=(-4, -13) if r.family == "Yao" else (-4, 7), textcoords="offset points", ha="right")
            elif r.family == "DelayBounded":
                ax.annotate(f"t = {r.t:g}", (r.lightness, r.stretch), fontsize=7.5, xytext=(6, 5), textcoords="offset points")
        ax.set_ylim(1.08, 2.06)
        ax.set_xlabel("lightness  (weight / MST weight)")
        ax.set_ylabel("exact stretch factor")
        ax.set_title(f"Classical trade-off, uniform points (n = {int(S_.n.max())})", loc="left", fontweight="bold")
        ax.legend(fontsize=8)
        ax.text(-0.14, 1.09, "c", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    fig.tight_layout()
    _save(fig, out, "fig4_baselines")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "benchmarks"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "figures"))
    a = ap.parse_args()
    for f in (fig1, fig2, fig3, fig4):
        f(a.results_dir, a.out_dir)


if __name__ == "__main__":
    main()
