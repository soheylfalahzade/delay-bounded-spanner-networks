# Delay-Bounded Time-Varying Geometric $t$-Spanners for Urban Road Networks

A certified sparsification framework that extracts a **single physical road backbone**
$H \subseteq G$ which preserves bounded *travel-time* dilation at **every hour of the
day** — not merely under free-flow conditions.

---

## 1. Problem Formulation

Let $G = (V, E)$ be a directed geometric road network embedded in $\mathbb{R}^2$, with
each edge $e \in E$ carrying a length $\ell(e)$ and a diurnal speed profile
$v(e, \tau)$ for $\tau \in \mathcal{T} = \{0, 1, \dots, 23\}$. The **time-sliced cost**
of an edge is

$$w(e, \tau) \;=\; \frac{\ell(e)}{v(e, \tau)}, \qquad
v(e, \tau) \;=\; \max\Big( \tfrac{1}{4} v_{\text{ff}}(e),\; v_{\text{ff}}(e)\,\big(1 - \alpha(e)\, c(\tau)\big) \Big),$$

where $v_{\text{ff}}(e)$ is the free-flow speed of the edge's functional class,
$\alpha(e) \in (0,1)$ its congestion susceptibility, and $c(\tau)$ the bimodal
congestion index

$$c(\tau) \;=\; \min\Big(1,\; 0.95\,e^{-\frac{(\tau-8)^2}{2\cdot1.35^2}}
\;+\; e^{-\frac{(\tau-17.5)^2}{2\cdot1.75^2}}
\;+\; 0.30\,e^{-\frac{(\tau-12.5)^2}{2\cdot3^2}} \;+\; 0.04\Big).$$

Write $G_\tau = (V, E, w(\cdot,\tau))$ for the snapshot at hour $\tau$ and
$\mathrm{dist}_X(u,v,\tau)$ for the shortest travel time from $u$ to $v$ in
$X \subseteq G$ under $w(\cdot,\tau)$.

> **Objective.** Find $H \subseteq E$ of minimum cardinality such that
> $$\forall u,v \in V,\ \forall \tau \in \mathcal{T}: \quad
> \mathrm{dist}_H(u,v,\tau) \;\le\; t \cdot \mathrm{dist}_G(u,v,\tau).$$

We call such an $H$ a **delay-bounded time-varying $t$-spanner**. The problem is the
temporal generalisation of minimum-size $t$-spanner construction, which is already
NP-hard and $\Omega(\log n)$-inapproximable in the static case; we therefore target a
greedy construction with a *verifiable* certificate rather than optimality.

---

## 2. Algorithmic Core

### Theorem 1 (Edge-wise certificate lifts to all pairs)

*Let $H \subseteq E$ satisfy the edge-wise invariant*
$$\forall (u,v) \in E,\ \forall \tau \in \mathcal{T}: \quad
\mathrm{dist}_H(u,v,\tau) \le t \cdot w((u,v), \tau).$$
*Then $H$ is a delay-bounded time-varying $t$-spanner of $G$.*

**Proof.** Fix $\tau$ and a pair $(u,v)$. Since $w(\cdot, \tau)$ is a fixed, non-negative
edge weighting, $G_\tau$ is an ordinary static digraph. Let
$P = \langle e_1, \dots, e_k \rangle$ be a shortest $u \to v$ path in $G_\tau$. By the
invariant and the triangle inequality of shortest-path distances in $H_\tau$,
$$\mathrm{dist}_H(u,v,\tau) \;\le\; \sum_{i=1}^{k} \mathrm{dist}_H(e_i, \tau)
\;\le\; t \sum_{i=1}^{k} w(e_i, \tau) \;=\; t\cdot \mathrm{dist}_G(u,v,\tau).$$
The argument holds for every $\tau$ independently, hence for all of $\mathcal{T}$. $\blacksquare$

Theorem 1 is what makes the problem tractable: the $|V|^2 \cdot |\mathcal{T}|$ pairwise
constraints collapse to $|E| \cdot |\mathcal{T}|$ *local* constraints.

### Delay-Bounded Greedy Spanner
Input : G = (V,E), cost vectors w(e, ·), dilation t ≥ 1
Output: H ⊆ E satisfying the all-pairs, all-hours guarantee
1 order E ascending by max_τ w(e, τ) (or by centrality-discounted cost)
2 H ← ∅
3 for (u,v) ∈ E in that order:
4 for τ ∈ T ordered by ascending w((u,v), τ): ▷ tightest budget first
5 β ← t · w((u,v), τ)
6 d ← BoundedDijkstra(H, u → v, snapshot τ, cutoff β) ▷ ball of radius β
7 if d > β: H ← H ∪ {(u,v)}; break ▷ early exit, skip slices
8 return H

Two design decisions keep the construction out of combinatorial blow-up:

* **Pruned local search (line 6).** The cutoff is a *single edge* cost scaled by $t$, so
  each probe settles only the nodes inside a small metric ball — never the whole graph.
  Complexity per probe is $O(|B_\beta| \log |B_\beta|)$, with $|B_\beta| \ll |V|$.
* **Tightest-budget-first slice ordering (line 4).** Edges that will be *kept* usually
  fail on their cheapest snapshot, so the $|\mathcal{T}|$-fold loop short-circuits after
  roughly one probe; the full 24 probes are paid only for edges that are genuinely
  redundant.

Overall: $O\!\left(|E| \cdot |\mathcal{T}| \cdot |B_\beta| \log |B_\beta|\right)$, and the
certificate is re-verified exhaustively by `certify_edge_condition`.

---

## 3. System Architecture
OSMnx / Overpass ──▶ graph_from_point → MultiDiGraph │
(primary) │ collapse ∥ edges → DiGraph → largest SCC │
│ │
Synthetic grid ────▶ arterial / collector / local stratified │
(fallback) │ grid + trunk diagonals + one-way pockets │
│ │
│ ENRICH: v(e,τ) from hierarchy × c(τ) │
│ w(e,τ) = ℓ(e) / v(e,τ) │
└─────────────────────┬─────────────────────┘
│ G with 24-vector costs
┌─────────────────────────┼─────────────────────────┐
▼ ▼ ▼
┌───────────────────────┐ ┌────────────────────┐ ┌──────────────────────┐
│ src/spanner.py │ │ src/baselines.py │ │ src/temporal.py │
│ Delay-Bounded Greedy │ │ • Static metric │ │ • bounded_dijkstra │
│ + exhaustive │ │ • Hierarchical │ │ • single_source │
│ certificate check │ │ • Full network │ │ snapshot oracle │
└───────────┬───────────┘ └─────────┬──────────┘ └──────────┬───────────┘
└────────────────────────┼────────────────────────┘
▼
┌──────────────────────────────┐
│ evaluate.py │
│ dilation × sparsity × time │
├──────────────────────────────┤
│ results/spanner_benchmark.png│
│ results/spanner_report.json │
│ results/benchmark_table.md │


---

## 4. Installation & Execution

```bash
conda env create -f environment.yml
conda activate spanner_research_env
python evaluate.py                 # full run (Yazd centre, r = 1500 m)
python evaluate.py --quick         # fast smoke test
python evaluate.py --synthetic     # fully offline, no Overpass call
python evaluate.py --order centrality
```

`src/network.py` degrades gracefully: any Overpass timeout, DNS failure, or rate limit
transparently falls back to the deterministic synthetic arterial-collector grid, so the
pipeline never fails to produce results.

---

## 5. Benchmark Results

Auto-generated by `evaluate.py` (also written to `results/benchmark_table.md`).

<!-- BENCHMARK_TABLE_START -->
_Run `python evaluate.py` to populate this table._
<!-- BENCHMARK_TABLE_END -->

**Figure** — `results/spanner_benchmark.png`

| Panel | Content |
|---|---|
| (a) | Worst-case dilation vs. hour of day. The static metric spanner is certified only for the free-flow snapshot and breaches $t$ during the shaded morning/evening peaks; the delay-bounded spanner stays under the bound for all 24 snapshots. |
| (b) | Percentage of edges removed as a function of $t$ — the sparsity/fidelity frontier. |
| (c) | Spatial map of the extracted emergency road backbone $H$ over the full network $G$, with arterial segments emphasised. |

---

## 6. Modelling Assumptions (stated explicitly for review)

1. **Time-slice (snapshot) model.** Each hour is treated as a static metric; we do not
   model FIFO departure-time evolution *within* a trip. This is the standard assumption
   for backbone/infrastructure design, where the artefact must hold for every operating
   condition rather than for one departure time. Theorem 1 depends on it.
2. **Synthetic diurnal profiles.** Speed curves are hierarchy-conditioned analytic
   models, not probe data. Replacing `speed_curve` with empirical floating-car or loop
   detector series requires no change to `spanner.py`.
3. **Directed semantics.** One-way restrictions are preserved; parallel OSM edges are
   collapsed to the shortest representative.
4. **Connectivity.** All experiments run on the largest strongly connected component;
   the hierarchical baseline is re-attached with cheapest inter-component edges so that
   dilation remains well defined.

---

## 7. Roadmap — from topological backbone to operations

| Stage | Deliverable | Link to this repository |
|---|---|---|
| **S1 — Backbone (this work)** | Certified delay-bounded $t$-spanner $H$ | `src/spanner.py` |
| **S2 — Emergency fleet routing** | Ambulance/fire dispatch restricted to $H$: the $t$-certificate bounds response-time inflation *by construction*, at every hour, so routing can be solved on a graph an order of magnitude smaller. | consumes `results/spanner_report.json` + $H$ |
| **S3 — Signal-priority allocation** | Green-wave and pre-emption budgets are scarce; $H$ identifies the minimal edge set whose control guarantees network-wide travel-time fidelity, making it the natural deployment target for adaptive signal control. | pairs with a stacking-ensemble signal controller |
| **S4 — Learned edge ordering** | Replace the hand-designed greedy ordering (line 1) with a GNN that predicts an edge's *spanner utility* from geometry + temporal profile, retaining the exhaustive certificate as a safety filter (learn to propose, verify to guarantee). | `_edge_order` is the single injection point |
| **S5 — Robust / stochastic extension** | Replace deterministic $w(e,\tau)$ with distributions and certify $\Pr[\mathrm{dist}_H \le t\,\mathrm{dist}_G] \ge 1-\delta$ via chance-constrained probes. | extends `bounded_dijkstra` |

---

## 8. Repository Layout
delay-bounded-spanner-networks/
├── environment.yml
├── requirements.txt
├── evaluate.py # benchmark harness, figures, JSON report
├── run.sh # one-line pipeline entry point
├── src/
│ ├── config.py # study area, temporal grid, experiment constants
│ ├── network.py # OSMnx acquisition + resilient synthetic fallback
│ ├── temporal.py # snapshot Dijkstra primitives (pruned + full)
│ ├── spanner.py # Delay-Bounded Greedy Spanner + certificate
│ └── baselines.py # static / hierarchical / full-network baselines
└── results/
├── spanner_benchmark.png
├── spanner_report.json
└── benchmark_table.md

## 9. Citation

```bibtex
@software{delay_bounded_spanner_2026,
  title  = {Delay-Bounded Time-Varying Geometric t-Spanners for Urban Road Networks},
  year   = {2026},
  note   = {Certified temporal sparsification for emergency routing backbones}
}
```
