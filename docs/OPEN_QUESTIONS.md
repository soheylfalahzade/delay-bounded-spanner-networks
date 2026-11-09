# Open questions and status

This is the baseline list of what is open in the repository, kept so that results are not confused with open questions. It is
not a list of defects: some items are modelling or validation limits, some are open theory, some concern how the algorithm is
evaluated. Nothing here is a plan to extract a thesis topic; the point is to know, for each claim, what supports it.

Status words: **done** (implemented and tested, results in the README), **partial** (done for a restricted setting),
**open** (not done), **limit** (a standing limitation of the data or model).

| # | Question | Status | Where | What it would change in a paper | Cost |
|---|---|---|---|---|---|
| 1 | Is the resistance of road networks to sparsification structural? | partial | README 5c, 5d, 5m, 5q, 5s, 5w; docs/RELATED_WORK.md section 9 | Central empirical claim. Forced share equals the share of arcs whose shortest cycle exceeds (t+1) times their length; one-way streets are a large contributor (5s, 5w) | Done; causal reading limited to the two-way counterfactual |
| 2 | Is the forced-edge bound correct, and how strong is it? | done / partial | Theorem 2 and `tests/`; 5j, 5o, slack in 5d | Elementary but exact; the bound is far from the greedy size at large t and close for small t; exact optimum known only on small windows | Exact optimum on large instances is open |
| 3 | How close is the greedy spanner to optimal? | partial | 5j (random digraphs), 5o (windows of real networks), 5n (pruning) | Empirical near-optimality on small road-like windows; no approximation guarantee | An ILP or branch-and-bound on larger windows; a theorem is open |
| 4 | Is the greedy spanner edge-minimal? | done | 5n | Not in general: the undirected argument does not cover digraphs; pruning removes a small share | Done |
| 5 | Does the processing order matter? | done | 5u | Large before pruning, small after | Done (four cities) |
| 6 | Does time variation matter for the forced bound? | done | 5m | In the parametric model only 1 to 2 percent of the forced arcs are forced because of the hourly profile; the claim "time-varying" cannot rest on the lower bound | Done for the model; real speeds only for one city |
| 7 | Region-fault tolerance for sampled faults | done | 5l | A fault-tolerant spanner for a sample generalises poorly | Done |
| 8 | Guarantee for a continuous family of faults | partial | 5v | Exact guarantee for all discs of one radius (finite scenario set) and an exact failure area; other radii, shapes and unions of families are open | Multi-radius families; non-convex regions |
| 9 | Worst fault and critical scenarios | partial | 5v (worst scenario per radius) | The worst scenario of the disc family is reported; there is no general characterisation | Open |
| 10 | Size bound for road networks | open | docs/RELATED_WORK.md sections 3 and 5 | Hardness is likely for planar digraphs (t >= 5, edge count); one-way spanners need not be sparse in general digraphs; a bound for near-planar road networks is open | Theory |
| 11 | Calibration of travel times with real street speeds | limit | 5h (one city), 5g (freeway sensors) | Needed for any claim about real congestion; only the shape for one city is checked | Data access |
| 12 | Validation with real regional faults (floods, closures) | limit | none | Would turn the disc model into a validated one | Data access |
| 13 | Scalability | partial | 5f; 5r (up to 140k arcs, real); 3M arcs synthetic | Time grows near linearly; memory and cost in the number of hours or scenarios are not measured | Profiling runs |
| 14 | Literature positioning | partial | docs/RELATED_WORK.md | No work found on the same problem; the search is short and several records are unchecked | A systematic search and a reading of the closest papers |
| 15 | Emergency routing as an application | done (weak) | 5p | Effect on response time is negligible at t = 2 | Done |

## Items that are not weaknesses of the algorithm

* Items 11 and 12 are data limits and move only when data is obtained.
* Items 8 and 9 are open because the problem is hard to state, not because the implementation is incomplete; the exact analysis of one disc radius (5v) is a restricted answer, not a full one.
* Item 13 is an evaluation gap: no claim in the README depends on memory use.

## Items that would change what can be claimed

1. 14 (literature): determines whether any novelty claim is possible.
2. 8 and 10: a guarantee or a size theorem moves the work from empirical to theoretical.
3. 11: determines whether statements about time variation carry over to real traffic.
