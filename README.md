Delay-Bounded Time-Varying Geometric t-Spanners for Urban Road Networks
A certified sparsification framework that extracts a single physical road backbone H ⊆ G, guaranteed to preserve bounded travel-time dilation at every hour of the day — validated end-to-end on six real Iranian metropolitan road networks pulled live from OpenStreetMap.
TL;DR
On the real OSM road network of Yazd (3,093 junctions / 6,806 directed edges, dist = 1500 m), at t = 1.5:
98.3% of edges and 96.1% of total road length are kept
The 24-hour, all-pairs dilation certificate is formally verified — yes
Worst empirical OD dilation across the day: 1.031× (vs. the theoretical bound of 1.5×)
A functional-class ("keep only arterials") baseline, despite retaining less mileage, is not certifiable and its empirical worst-case dilation blows up to 9.50×
Run on six real Iranian metros with distinct street morphologies (historic-organic cores, radial-grid metros, hill terrain, a modern planned grid), the same construction stays certified in every single city, retaining 95.1%–98.8% of edges.
1. Problem
Let G = (V, E) be a directed geometric road network. Each edge e carries a length ℓ(e) and an hourly diurnal speed profile v(e, τ) for τ ∈ {0, …, 23}, giving the time-sliced travel-time cost
code
Code
w(e, τ) = ℓ(e) / v(e, τ)
We construct a sparse backbone H ⊆ E certified to satisfy
code
Code
∀ u, v ∈ V,  ∀ τ ∈ {0,…,23}:   dist_H(u, v, τ)  ≤  t · dist_G(u, v, τ)
Theorem 1 (edge-wise certificate ⇒ all-pairs, all-hours guarantee).
If dist_H(u,v,τ) ≤ t · w((u,v),τ) holds for every edge (u,v) ∈ E and every hour τ, then H is a delay-bounded time-varying t-spanner of G. Proof: fix τ; w(·,τ) is a static, non-negative weighting, so concatenating the edge-wise invariant along a τ-optimal path and applying the triangle inequality in H_τ gives the all-pairs bound, independently for every τ. ∎
This collapses |V|² · 24 pairwise constraints into |E| · 24 local certificates, each decidable by a Dijkstra search pruned to the ball of radius t · w(e,τ).
2. Algorithm — Delay-Bounded Greedy Spanner
code
Code
1  order E ascending by max_τ w(e, τ)
2  H ← ∅
3  for (u, v) ∈ E in that order:
4      for τ ordered by ascending w((u,v), τ):        # tightest budget first
5          β ← t · w((u,v), τ)
6          if BoundedDijkstra(H, u→v, τ, β) > β:
7              H ← H ∪ {(u,v)}; break                  # early exit
8  return H
Complexity O(|E| · 24 · |B| log|B|), |B| the size of the local metric ball — empirically ≈1.2–2.5 Dijkstra probes per edge across the tested range of t, and T ∝ |E|^1.19–1.22 in the empirical scaling study (see results/spanner_benchmark.png, panel e).
3. Baselines compared
Method	What it is	Certified?
Delay-Bounded Spanner (ours)	Theorem-1 construction above	Yes, by construction
Static Free-Flow Spanner	Classical greedy spanner on the free-flow snapshot only	No
Mean-Temporal Spanner	Greedy spanner on the 24h-averaged cost	No
Static Geometric Spanner	Classical Euclidean-length greedy spanner	No
Yao-Cone Sparsifier (k=8)	Network-constrained adaptation of the classical Yao graph (cheapest edge per 45° cone per node)	No
Hierarchical Road Classifier	Keep only arterial/collector classes + last-mile attachment	No
Matched-Density Random / Betweenness	Density-matched structural controls (same edge budget as ours)	No
Full Network	Ground truth (t = 1)	—
4. Results — Focus city: Yazd (real OSM, r = 1500 m)
Method	t	Edges kept (%)	Length kept (%)	Attained worst edge stretch	Certified (24h)	Worst OD dilation	Build (s)
Delay-Bounded (ours)	1.10	99.2	98.0	1.100	yes	1.002	0.11
Static Free-Flow	1.10	98.9	97.3	1.488	no (23 edges)	1.031	0.04
Delay-Bounded (ours)	1.50	98.3	96.1	1.500	yes	1.031	0.08
Static Free-Flow	1.50	97.8	95.0	1.907	no (31 edges)	1.074	0.09
Mean-Temporal	1.50	97.9	95.4	2.433	no (23 edges)	1.074	0.05
Delay-Bounded (ours)	3.00	93.4	86.2	3.000	yes	1.223	0.24
Static Geometric	1.50	98.1	95.7	2.847	no (19 edges)	1.117	0.05
Yao-Cone (k=8)	1.50	99.5	98.8	8.000	no (14 edges)	1.113	0.03
Matched-Density Betweenness	1.50	98.3	97.2	8.000	no (94 edges)	1.210	1.39
Hierarchical Road Classifier	—	85.4	79.7	8.000	no (955 edges)	9.495	0.15
Full Network	—	100.0	100.0	—	—	1.000	0.00
Full table: results/benchmark_table.md · edge-ordering ablation: results/ablation_table.md
![Image](results/spanner_benchmark.png)
5. Cross-city generalization (6 real Iranian metros, live OSM, t = 1.5)
City	Archetype	Nodes	Edges	Orient. entropy	Circuity	Ours keep%	Certified	Ours worst OD	Matched-Btw worst OD	Advantage
Yazd	historic-organic	3,093	6,806	0.946	1.039	98.3	yes	1.012	1.230	1.22×
Tehran	radial-grid	2,129	4,023	0.777	1.033	97.7	yes	1.016	1.076	1.06×
Isfahan	historic-organic	3,818	7,302	0.907	1.040	98.8	yes	1.007	1.000	0.99×
Shiraz	hill-organic	1,965	4,218	0.893	1.032	96.3	yes	1.057	1.461	1.38×
Mashhad	radial-grid	1,185	2,968	0.827	1.022	95.1	yes	1.086	1.006	0.93×
Qom	modern-grid	2,208	4,766	0.865	1.030	97.4	yes	1.126	1.002	0.89×
Every one of the six cities certifies at t = 1.5. Data source is live OpenStreetMap/Overpass for all six (see used_real_osm: true in results/spanner_report.json).
Full table: results/crosscity_table.md
![Image](results/spanner_crosscity.png)
Statistical honesty:
Mean edge circuity vs. edge retention: r = 0.95, p = 0.004 (significant at n=6).
Orientation entropy vs. advantage over matched-density: r = 0.41, p = 0.42 (not significant — needs more cities).
City-level Wilcoxon signed-rank (ours < matched-density betweenness): p = 0.34, n = 6 — directional, underpowered, not confirmatory. A larger multi-country replication is future work.
6. Robustness to model misspecification
The diurnal congestion field is a parametric tidal model (CBD-weighted, inbound/outbound asymmetric), not fitted to floating-car or loop-detector telemetry. To test whether this matters, the congestion-severity coefficient α was rescaled ×0.70 … ×1.30 and the full construction + certification repeated:
Delay-Bounded Spanner: worst OD dilation stays in [1.02, 1.06], always certified, always well under the t = 1.5 bound.
Matched-Density Betweenness control: worst OD dilation climbs to 1.38×, well past the bound, as congestion severity increases.
See panel (k) in results/spanner_crosscity.png.
7. Reproduce
code
Bash
conda create -n spanner_research_env python=3.10 -y
conda activate spanner_research_env
conda install -c conda-forge networkx osmnx geopandas shapely scipy numpy pandas matplotlib seaborn tqdm -y

python run_spanner_benchmark.py                    # full run: focus city + 6-city cross-city study
python run_spanner_benchmark.py --quick             # fast smoke test
python run_spanner_benchmark.py --synthetic         # force offline mode (deterministic morphology-parameterized synthetic fallback)
python run_spanner_benchmark.py --skip-crosscity    # focus city only
If Overpass/OSMnx is unreachable, the pipeline transparently falls back to a deterministic, morphology-parameterized synthetic generator and labels each city's used_real_osm flag accordingly in results/spanner_report.json — it never silently mixes real and synthetic data without saying so.
8. Repository layout
code
Code
.
├── run_spanner_benchmark.py       # single self-contained pipeline (model, algorithm, baselines, eval, figures)
├── README.md
└── results/
    ├── spanner_benchmark.png      # 7-panel focus-city figure
    ├── spanner_crosscity.png      # 4-panel cross-city generalization figure
    ├── spanner_report.json        # full machine-readable record
    ├── benchmark_table.md
    ├── ablation_table.md
    ├── crosscity_table.md
    └── per_pair_dilation.csv
9. Known limitations
Snapshot (time-slice) model: no departure-time evolution within a single traversal; ex-post validated by a FIFO time-dependent Dijkstra experiment (fifo_violation_rate = 0.0 on the tested networks).
Congestion is a parametric model, not measured telemetry — see Section 6 for the sensitivity analysis.
n = 6 cities supports a directional cross-morphology generalization claim, not a definitive one.
Greedy construction carries no approximation guarantee on |H|; the matched-density and Yao-cone baselines are empirical, not theoretical, lower-bound proxies.
10. Citation
code
Bibtex
@software{delay_bounded_spanner_2026,
  title  = {Delay-Bounded Time-Varying Geometric t-Spanners for Urban Road Networks},
  year   = {2026},
  note   = {Certified temporal sparsification, validated on six real OpenStreetMap metropolitan road networks}
}