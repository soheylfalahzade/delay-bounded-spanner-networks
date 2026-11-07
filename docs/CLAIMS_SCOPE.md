# Scope of the claims (derived from the audit and the measured results)

**Supported claims**
1. Theorem 1: an edge-wise certificate at hour tau implies the all-pairs bound at hour tau; the greedy construction attains the certificate at all 24 grid hours (exhaustively verified on 236 random instances; certified on 12 OSM city cores, METR-LA and PEMS-BAY).
2. Theorems 3 / 3b: certification on a finer sub-grid with interpolation factors gives a bound for every time of day under the piecewise-linear profile (verified on METR-LA and PEMS-BAY; cost on METR-LA / PEMS-BAY: +10.6 / +4.3 points of edges at T = 1.5, +2.8 / +0.3 at T = 2, no increase at T = 3; the global version (Theorem 3) is infeasible at T = 1.5 and T = 2 on both data sets).
3. Theorem 2: edges with detour ratio above t are forced; retention equals this lower bound within 0.4 points at t <= 1.5 (12 city cores) and the gap grows to 25.6 points at t = 10.
4. Uncertified baselines (static free-flow, mean-temporal) violate the bound on measured traffic for most t; the union of per-hour greedy spanners is also certified and only 1.4-3.8 points larger on METR-LA/PEMS-BAY.

**Not claimed**
- Guarantee for time-dependent *traversal* (path-dependent arrival times): empirical only.
- Optimality or any size/approximation bound; the joint greedy is not provably <= the union baseline.
- Superiority over matched-density controls on OD dilation as a guarantee: the advantage ratio is above 1 in 11 of 12 cities and below 1 in Melbourne; the city-level test has n = 12 and is directional evidence only.
- Metropolitan-scale validity beyond the 3.5 km runs on three cities (README section 5g).
- Validity of the congestion model on urban street speeds beyond what `tools/validate_congestion_shape.py` shows.

**Added scope notes (region faults, forced edges by hour, slack captured)**
- Region faults (README 5l): fault tolerance is certified for the sampled family of disc faults only (Theorem 1 applied to each G-F); other regions are covered by an empirical held-out column, not by a guarantee. Faults are discs; a segment touches a disc if its stored polyline does. The forced bound over the family is a lower bound for that family only.
- Forced edges by hour (README 5m): describes the parametric congestion model, not measured street speeds.
- Slack captured (README 5d, subsection "Slack captured"): the share of the removable slack removed is a lower bound on the share of the best possible removal only when the forced bound is the binding limit; the exact optimum is known only for the small random instances of 5j.
- Theorem 1 and Theorem 2 are elementary (the standard edge-wise argument; the detour test behind the greedy spanner having no proper t-spanner of itself); they are not presented as new theorems.

**Suggested summary sentence**
"We give an edge-wise certificate that makes a subgraph of a road network a t-spanner for every hourly snapshot of a time-varying travel-time profile (and, with a per-edge interpolation factor, for every time of day), prove that edges without a detour of ratio at most t are forced in every such spanner, and show on twelve OpenStreetMap city cores and two measured-speed sensor networks that this forced-edge bound explains why little sparsification is possible at small t."
