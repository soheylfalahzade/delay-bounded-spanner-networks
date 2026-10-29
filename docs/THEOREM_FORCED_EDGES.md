# Theorem 2 (forced edges) - paper-ready statement

**Setting.** Let G = (V, E) be a strongly connected digraph with positive, hour-dependent arc weights
w_tau(e), tau in {0,...,23}. A spanning subgraph H = (V, E') with E' subset of E is a *t-spanner of G at hour tau*
if d_{H,tau}(x, y) <= t * d_{G,tau}(x, y) for all x, y in V. For an arc e = (u, v) write G - e for G with e removed and define

    r(e) = max over tau of  d_{G-e,tau}(u, v) / w_tau(e)      (r(e) = infinity if v is unreachable from u in G - e).

**Theorem 2.** Let t >= 1 and let H be an edge-subset of G that is a t-spanner of G at every hour. Then every arc e with r(e) > t belongs to H.

**Proof.** Let e = (u, v) with r(e) > t, and pick an hour tau with d_{G-e,tau}(u, v) > t * w_tau(e). Suppose e is not in H. Then H is a subgraph of G - e, so
d_{H,tau}(u, v) >= d_{G-e,tau}(u, v) > t * w_tau(e) >= t * d_{G,tau}(u, v),
where the last step uses d_{G,tau}(u, v) <= w_tau(e) (the arc itself is a u-v path). This contradicts the spanner property at hour tau. Hence e is in H. QED

**Corollary (universal lower bound on retention).** For every edge-subset t-spanner H of G at all hours (in particular for the delay-bounded greedy spanner),

    |E(H)| / |E|  >=  F(t) := |{e in E : r(e) > t}| / |E|.

The bound depends on G and t only, not on the construction; it is non-increasing in t.
It also holds for the weaker edge-wise certificate used in this repository: if every arc e satisfies
d_{H,tau}(u, v) <= t * w_tau(e), the same argument applies verbatim.

**Remarks for the manuscript.**
1. Theorem 2 explains why road networks sparsify little at small t: most street segments have no alternative path with detour ratio <= t (bridges, cul-de-sacs, one-way couplets, long blocks).
2. The bound is generally not tight for large t (empirically a gap of about 13 percentage points at t = 5 on the twelve OSM city cores), so it supports no optimality claim for the greedy construction.
3. Empirical computation: `python benchmarks/retention_diagnostics.py`; large-t and morphology analysis: `python benchmarks/retention_morphology.py`.
