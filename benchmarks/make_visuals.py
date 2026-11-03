#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
README visuals: header banner, pipeline diagram (SVG) and the construction animation (GIF).

  python benchmarks/make_visuals.py

The numbers on the banner are computed from the raw result files, and the banner refuses to print a claim the data does not support.
Output: docs/assets/banner.svg, docs/assets/pipeline.svg, results/figures/greedy_construction.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
RES = os.path.join(ROOT, "results", "benchmarks")
ASSETS = os.path.join(ROOT, "docs", "assets")


def headline_metrics():
    ls = pd.read_csv(os.path.join(RES, "large_scale_3500m.csv"))
    per_city = ls.groupby("city").delay_bounded_certified.all()
    assert per_city.all(), "a city of the larger-radius run is not certified: the banner would overstate"
    gap = (ls.delay_bounded_retention_pct - ls.forced_lower_bound_pct)[ls.t == 1.5]
    og = pd.read_csv(os.path.join(RES, "optimality_gap_raw.csv"))
    share = 100 * ((og.greedy == og.optimum)[og.t == 1.5]).mean()
    return {"cities": f"{len(per_city)}/{len(per_city)}", "gap": f"{gap.max():.2f} pts", "optimal": f"{share:.0f}%",
            "n_cities": len(per_city), "n_spanners": len(ls), "n_inst": og.inst.nunique()}


def banner_svg():
    m = headline_metrics()
    cards = [(m["cities"], "cities certified at 3.5 km", f"{m['n_spanners']} spanners, every one passes the check"),
             (m["gap"], "largest gap to the structural lower bound", f"t = 1.5, {m['n_cities']} cities"),
             (m["optimal"], "greedy equals the exact optimum", f"t = 1.5, {m['n_inst']} small instances")]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="320" viewBox="0 0 1100 320" role="img" '
           'aria-label="Delay-bounded spanners: headline results">',
           '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1b2e"/><stop offset="1" stop-color="#15304d"/></linearGradient></defs>',
           '<rect width="1100" height="320" rx="18" fill="url(#bg)"/>',
           '<circle cx="1030" cy="40" r="95" fill="#ffffff" fill-opacity="0.04"/>',
           '<text x="48" y="78" font-family="Georgia, \'Times New Roman\', serif" font-size="40" font-weight="700" fill="#ffffff">delay-bounded-spanner-networks</text>',
           '<text x="48" y="112" font-family="Georgia, \'Times New Roman\', serif" font-size="20" font-style="italic" fill="#f0a030">Which streets can go without slowing anyone down, at any hour?</text>',
           '<text x="48" y="142" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#b9c7d8">Certified sparsification of road networks with hourly travel times: a per-edge certificate, a structural lower bound and the exact optimum on small instances</text>']
    for i, (num, lab, sub) in enumerate(cards):
        x = 48 + i * 340
        out += [f'<rect x="{x}" y="176" width="320" height="108" rx="6" fill="#0f2238" stroke="#243f60"/>',
                f'<rect x="{x}" y="176" width="320" height="4" rx="2" fill="#f0a030"/>',
                f'<text x="{x + 20}" y="226" font-family="Georgia, serif" font-size="38" font-weight="700" fill="#f0a030">{num}</text>',
                f'<text x="{x + 20}" y="251" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#ffffff">{lab}</text>',
                f'<text x="{x + 20}" y="272" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#8fa3ba">{sub}</text>']
    out.append("</svg>")
    return "\n".join(out) + "\n"


def pipeline_svg():
    steps = [("Hourly costs", "w(e, hour), 24 values"), ("Order edges", "by their largest cost"),
             ("Check each edge", "bounded Dijkstra in H, per hour"), ("Keep or drop", "keep if some hour has no detour within t"),
             ("Spanner H", "certified in every hour"), ("Verify", "bound, optimum, independent check")]
    w, gap, x0, y0, h = 150, 22, 28, 66, 70
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="190" viewBox="0 0 1100 190" role="img" aria-label="Construction pipeline">',
           '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#46607f"/></marker></defs>',
           '<rect width="1100" height="190" rx="14" fill="#ffffff" stroke="#d0d7de"/>',
           '<text x="550" y="34" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#46607f">from hourly travel times to a certified sparse road network</text>']
    for i, (a, b) in enumerate(steps):
        x = x0 + i * (w + gap)
        fill = "#e8f1fb" if i not in (4,) else "#e3f6ef"
        out += [f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="#46607f"/>',
                f'<text x="{x + w / 2}" y="{y0 + 30}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="#14304d">{a}</text>']
        words = b.split(" ")
        lines, cur = [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 21:
                lines.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
        for k, ln in enumerate(lines[:2]):
            out.append(f'<text x="{x + w / 2}" y="{y0 + 48 + 14 * k}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11.5" fill="#46607f">{ln}</text>')
        if i < len(steps) - 1:
            out.append(f'<line x1="{x + w + 2}" y1="{y0 + h / 2}" x2="{x + w + gap - 3}" y2="{y0 + h / 2}" stroke="#46607f" stroke-width="1.8" marker-end="url(#ar)"/>')
    out.append('<text x="550" y="172" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#6a7b90">every dropped edge keeps a detour of at most t times its own cost in every hour, which makes every pair of places t-close</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def animation(path, t=2.0, fps_ms=650):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from make_figures_extra import CONGESTION, GREY, NAVY, ORANGE, TEAL, toy_network
    import run_spanner_benchmark as core

    G, pts = toy_network()
    H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    hours = list(range(24))
    ordered = core._edge_order(G, "temporal_max", "tt", hours)
    adj = {n: {} for n in G.nodes()}
    events, seen = [], set()
    for u, v, data in ordered:
        costs = data["tt"]
        detour = [core.unbounded_dijkstra(adj, u, v, tau) for tau in hours]
        bound = [t * c for c in costs]
        add = any(d > b + core.EPS for d, b in zip(detour, bound))
        assert add == H.has_edge(u, v), "the replay of the greedy decision disagrees with the library"
        if add:
            adj[u][v] = costs
        key = tuple(sorted((u, v)))
        if key not in seen:
            seen.add(key)
            events.append(dict(u=u, v=v, kept=add, detour=detour, bound=bound, cost=list(costs)))
        else:
            for ev in events:
                if tuple(sorted((ev["u"], ev["v"]))) == key and add:
                    ev["kept"] = True
    decided = {}
    fig = plt.figure(figsize=(11.2, 5.0), dpi=80)
    ax = fig.add_axes([0.01, 0.06, 0.46, 0.80])
    bx = fig.add_axes([0.58, 0.20, 0.39, 0.52])
    n = len(events)

    def draw(i):
        ax.clear()
        bx.clear()
        ev = events[min(i, n - 1)]
        for k in range(min(i, n - 1) + 1):
            e = events[k]
            kept = e["kept"] if k < i or i >= n - 1 else None
            col = NAVY if kept else (GREY if kept is False else ORANGE)
            ax.plot(*zip(pts[e["u"]], pts[e["v"]]), color=col, lw=3.0 if kept is not False else 1.3,
                    ls="-" if kept is not False else (0, (3, 3)), solid_capstyle="round", zorder=2 if kept else 1)
        for k in range(min(i, n - 1) + 1, n):
            e = events[k]
            ax.plot(*zip(pts[e["u"]], pts[e["v"]]), color="#d7dde5", lw=1.0, zorder=0)
        e = ev
        ax.plot(*zip(pts[e["u"]], pts[e["v"]]), color=ORANGE, lw=4.2, zorder=3, solid_capstyle="round")
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color="white", edgecolor="#2c3e50", linewidth=1.1, zorder=4)
        ax.set_aspect("equal")
        ax.axis("off")
        kept_so_far = sum(1 for k in range(min(i, n - 1)) if events[k]["kept"])
        ax.set_title(f"edge {min(i, n - 1) + 1} of {n}   |   kept so far: {kept_so_far}", fontsize=11, fontweight="bold")
        h = np.arange(24)
        det = np.array([d if np.isfinite(d) else np.nan for d in e["detour"]])
        top = max(e["bound"]) * 2.2
        nopath = np.isnan(det)
        bx.bar(h[~nopath], np.minimum(det[~nopath], top), color="#9bb7d4", width=0.75, label="best detour in H so far")
        if nopath.any():
            bx.bar(h[nopath], [top] * int(nopath.sum()), color="#e8b4b0", width=0.75, label="no detour exists yet")
        bx.plot(h, e["bound"], color=NAVY, lw=2.2, label=f"bound t \u00b7 cost (t = {t:g})")
        bx.set_ylim(0, top)
        bx.set_xlim(-0.8, 23.8)
        bx.set_xticks([0, 6, 12, 18, 23])
        bx.set_xlabel("hour of day")
        bx.set_ylabel("travel time")
        viol = [k for k in range(24) if (not np.isfinite(e["detour"][k])) or e["detour"][k] > e["bound"][k] + core.EPS]
        txt = (f"kept: no detour within the bound at hour {viol[0]:02d}" if viol else "dropped: a detour within the bound exists in all 24 hours")
        bx.set_title(txt, fontsize=10.5, fontweight="bold", color=NAVY if viol else "#5a6b7d")
        bx.legend(loc="upper left", fontsize=8)
        fig.suptitle("Greedy construction on a small network", fontsize=12.5, fontweight="bold", y=0.97)

    frames = list(range(n)) + [n] * 5
    anim = FuncAnimation(fig, draw, frames=frames, interval=fps_ms)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=1000 / fps_ms), dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    open(os.path.join(ASSETS, "banner.svg"), "w", encoding="utf-8", newline="").write(banner_svg())
    open(os.path.join(ASSETS, "pipeline.svg"), "w", encoding="utf-8", newline="").write(pipeline_svg())
    if not a.no_gif:
        animation(os.path.join(ROOT, "results", "figures", "greedy_construction.gif"))
    print("[visuals] banner.svg, pipeline.svg" + ("" if a.no_gif else ", greedy_construction.gif"))


if __name__ == "__main__":
    main()
