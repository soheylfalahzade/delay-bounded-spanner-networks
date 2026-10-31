# What the paper may and may not claim (derived from the audit and the measured results)

**Supported claims**
1. Theorem 1: an edge-wise certificate at hour tau implies the all-pairs bound at hour tau; the greedy construction attains the certificate at all 24 grid hours (exhaustively verified on 236 random instances; certified on 12 OSM city cores, METR-LA and PEMS-BAY).
2. Theorems 3 / 3b: certification on a finer sub-grid with interpolation factors gives a bound for every time of day under the piecewise-linear profile (verified on METR-LA and PEMS-BAY; cost: +10.6 / +4.3 points of edges at T = 1.5, none at T >= 2 on the two data sets).
3. Theorem 2: edges with detour ratio above t are forced; retention equals this lower bound within 0.4 points at t <= 1.5 (12 city cores) and the gap grows to 25.6 points at t = 10.
4. Uncertified baselines (static free-flow, mean-temporal) violate the bound on measured traffic for most t; the union of per-hour greedy spanners is also certified and only 1.4-3.8 points larger on METR-LA/PEMS-BAY.

**Not claimed**
- Guarantee for time-dependent *traversal* (path-dependent arrival times): empirical only.
- Optimality or any size/approximation bound; the joint greedy is not provably <= the union baseline.
- Superiority over matched-density controls on OD dilation (ratio below 1 in 8 of 12 cities).
- Metropolitan-scale validity from 1.5 km cores alone (see `benchmarks/large_scale_run.py`).
- Validity of the congestion model on urban street speeds beyond what `tools/validate_congestion_shape.py` shows.

**Suggested abstract sentence**
"We give an edge-wise certificate that makes a subgraph of a road network a t-spanner for every hourly snapshot of a time-varying travel-time profile (and, with a per-edge interpolation factor, for every time of day), prove that edges without a detour of ratio at most t are forced in every such spanner, and show on twelve OpenStreetMap city cores and two measured-speed sensor networks that this forced-edge bound explains why little sparsification is possible at small t."
