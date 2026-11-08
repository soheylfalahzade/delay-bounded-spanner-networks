# Related work and positioning

Status: working document. It records what a targeted literature search found (about ten searches, November 2026), how each work
relates to the claims of this repository, and what is still unchecked. It is not a systematic review. Entries marked **[verified]**
were confirmed against a publisher, arXiv or proceedings record during the search; entries marked **[to check]** are recalled from
memory or from a secondary page and must be checked before they are cited in a paper. No search found a work that studies the same
problem (a single subgraph of a directed road network that is a t-spanner at every hour of the day), but absence from a short search
is not evidence of absence: see "How to extend this search" at the end.

## 1. Where each claim of the repository stands

| Claim of the repository | Closest prior work found | Relation | Consequence for the paper |
|---|---|---|---|
| Hourly greedy construction with bounded Dijkstra and an exact edge-wise certificate (README 2-3) | Greedy spanner (Althoefer et al. 1993); Farshi and Gudmundsson style experiments; Farshi and Hekmat Nasab, *Scientia Iranica* 21 (2014) 2142-2152 [verified: page header] | The construction is the greedy rule applied to 24 weight functions; the certificate is the standard edge-wise argument | No algorithmic novelty is claimed (already stated in README). The contribution is the application and the exact certificate at scale |
| Forced edges are in every spanner; their share is a lower bound (Theorem 2) | Only t-spanner of a greedy t-spanner is itself (Filtser and Solomon) | Same detour test, applied to G instead of H | Elementary. The measurement on real networks, by hour and under faults, is the content |
| Exact minimum is hard, so a computable lower bound matters | Brandes and Handke (1998) [verified] | See section 3 | Gives the lower bound a reason to exist |
| Region faults with H-F against G-F | Abam, de Berg, Farshi, Gudmundsson (2009, geometric); Chechik et al. (STOC 2009) and Dinitz and Krauthgamer (graphs) | Same reference convention (the fault is applied to the reference graph too) | The definition is not new; the road-network experiment and the forced bound over a family of faults are |
| Time variation adds little to the forced bound (1.1 to 1.6 percentage points, README 5m) | Temporal spanners (Casteigts et al.; Bilo et al.) use time-respecting journeys, a different model | Not comparable | State the finding as it is: for the parametric profiles, the hourly forced set is almost the free-flow forced set |

## 2. Spanners: static case and optimality of the greedy construction

* Althoefer, Das, Dobkin, Joseph, Soares, *On sparse spanners of weighted graphs*, Discrete Comput. Geom. 9 (1993) 81-100 (already README reference 6).
* Filtser and Solomon, *The greedy spanner is existentially optimal*. arXiv:1605.06852 [verified]. Preliminary version PODC 2016; the arXiv record states SICOMP 2020. **[to check: volume and pages, number of the lemma that the only t-spanner of a greedy t-spanner is itself]**. Existential optimality is a worst-case statement over a graph family; it does not say that the greedy spanner is near-optimal on a given instance.
* Le and Solomon, *Truly optimal Euclidean spanners*, FOCS 2019 [verified: proceedings listing]. **[to check: pages]**
* Le, Solomon, Than, Toth, Zhang, *Towards instance-optimal Euclidean spanners*, arXiv [verified: authors and title]. **[to check: venue]** The point to use: in Euclidean point sets the greedy spanner is not instance-optimal. The repository's observation that the greedy construction is near the exact optimum on small road windows (README 5o) is instance-level, empirical, and about road-like graphs; it is not an instance-optimality theorem.

## 3. Hardness

Brandes and Handke, *NP-completeness results for minimum planar spanners*, Discrete Mathematics and Theoretical Computer Science 3(1) (1998) 1-10, doi:10.46298/dmtcs.257 [verified]. As stated in the abstract: finding a minimum t-spanner is NP-hard for planar weighted graphs and digraphs when t >= 3 (objective: minimum total edge weight), and for planar unweighted graphs and digraphs when t >= 5 (objective: minimum number of edges).

**Proposition (proposed here; proof to be checked).** With the number of edges as the objective, the minimum delay-bounded spanner problem is NP-hard for every fixed t >= 5 on planar digraphs.
*Argument.* Take an unweighted planar digraph and give every arc the same constant travel time at every hour. The hourly spanner condition is then the ordinary spanner condition, so a solution of one is a solution of the other, and the problem of Brandes and Handke is a special case.
*Caveats.* (i) Road networks are not exactly planar (bridges, tunnels, overpasses), so this shows hardness of the model class, not of each real instance. (ii) For 3 <= t < 5 the cited result concerns total weight, not the number of edges, so it does not transfer. (iii) It says nothing about approximability on road networks.

## 4. Fault tolerance

* Abam, de Berg, Farshi, Gudmundsson, *Region-fault tolerant geometric spanners*, Discrete Comput. Geom. 41(4) (2009) 556-582 [as cited in README reference 1]. The definition used in README 5l: for a convex region F, the spanner minus F is a t-spanner of the complete graph minus F [verified: abstract].
* The same group has a greedy construction for region-fault tolerant geometric spanners (Persian-language journal page found in the search). **[to check: authors, journal, year; ask Dr. Farshi for the reference]**
* Chechik, Langberg, Peleg, Roditty, *Fault tolerant spanners for general graphs*, STOC 2009, pp. 435-444; SIAM J. Comput. 39(7) (2010) 3403-3423 [verified]. An f-edge fault tolerant (2k-1)-spanner with O(f n^{1+1/k}) edges; the fault is applied to the reference graph as well as the spanner.
* Dinitz and Krauthgamer, *Fault-tolerant spanners: better and simpler*, PODC 2011, arXiv:1101.5753 [verified: arXiv listing]. **[to check: PODC pages]**
* Bilo, D'Angelo, Guala, Leucci, Rossi, *Blackout-tolerant temporal spanners*, J. Comput. System Sci. 141 (2024) 103495, doi:10.1016/j.jcss.2023.103495 [verified]. A blackout is the failure of every edge carrying one time label; different fault model from regional faults.

What is specific here: faults are geographic discs on a real road network, the reference is G-F, and the bound is measured by forced edges. What the general-graph results suggest is that a worst-case guarantee for all regions would need size growing with the number of fault scenarios; the repository's held-out column (README 5l) shows the same in practice: tolerance for a sampled family generalises poorly.

## 5. Directed graphs

For one-way distance, directed graphs need not have sparse spanners at all: a complete directed bipartite graph (all arcs from one side to the other) requires every arc [verified: Pachocki, Roditty, Sidford, Tov, Vassilevska Williams, arXiv:1611.00721 and Chechik et al. arXiv:1907.10779]. The literature therefore works with roundtrip distance d(u,v) + d(v,u) [verified: Roditty, Thorup, Zwick, *Roundtrip spanners and roundtrip routing in directed graphs*, ACM Trans. Algorithms 4(3), 2008; (2k-1)-roundtrip spanners with O(k n^{1+1/k} log n) edges by Cen, Duan, Gu, ICALP 2020, arXiv:1911.12411].

Consequences for this repository. (a) The size question has no general answer for one-way spanners, so a size theorem must use the structure of road networks (sparsity, near-planarity); the trivial bound is |E|. (b) **Remark (proposed, short proof).** The certificate of Theorem 1 is for one-way distances, and a one-way t-spanner is also a roundtrip t-spanner, because d_H(u,v) <= t d_G(u,v) and d_H(v,u) <= t d_G(v,u) add up. So the repository's guarantee is stronger than the roundtrip guarantee studied in the directed-spanner literature.

## 6. Temporal graphs

* Casteigts, Peters, Schoeters, *Temporal cliques admit sparse spanners*, ICALP 2019 (README reference 18).
* Bilo, D'Angelo, Guala, Leucci, Rossi, *Sparse temporal spanners with low stretch*, ESA 2022, LIPIcs 244, 19:1-19:16, doi:10.4230/LIPIcs.ESA.2022.19 [verified]. Edges carry time labels, paths are time-respecting, and distance is the number of edges.

Different model: here the graph is fixed, travel times vary with the hour, and the guarantee is per hourly snapshot (with the continuous-time variants of README 5f). The comparison belongs in the introduction as a paragraph of scope, not as a competitor.

## 7. Road networks

* Road networks are not planar in practice, but their crossing structure is sparse: a result reported on D. Eppstein's publication list (title not captured). **[to check: D. Eppstein and M. T. Goodrich, *Studying (non-)planar road networks through an algorithmic lens*, ACM SIGSPATIAL GIS 2008]**
* A recent arXiv study compares resistance-ranked and betweenness-based edge pruning on urban road networks (arXiv:2609.29540) [verified: title only; not read]. Empirical neighbour of the matched-density baselines; it is not a spanner paper and it gives no distance guarantee.
* Inserting edges into a road network to reduce constrained diameters (arXiv:2404.19164) [verified: abstract]. The opposite direction (adding rather than removing edges).

## 8. Several weight functions at once

No work was found that asks for one subgraph that is a t-spanner for several edge-weight functions simultaneously (the 24 hours here). Neighbouring formulations: multi-level spanners for nested vertex sets (arXiv:1904.01135) and multipath spanners (Gavoille et al.), both different. Treat this as the main open literature question; see below.

## 9. An exact characterisation that explains the forced share

For an arc e = (u,v), the forced condition d_{G-e}(u,v) > t w(e) says that the cheapest alternative route from u to v that avoids e is longer than t w(e). In an undirected graph this is the same as: the shortest cycle through e has weight above (t+1) w(e). In a unit-weight graph of girth g, every edge is therefore forced for all t < g-1, which is the classical girth-spanner connection (the greedy t-spanner has girth above t+1). The unit grid has girth 4, so every edge is forced for t < 3 and none for t >= 3, which the test `tests/test_forced_by_hour.py` checks and which explains the step in strict-grid cities. The share of forced edges as a function of t is thus the distribution of "cycle ratio" over edges, a local structural property of the street network; this is the quantity the morphology analysis (README 5q) relates to street-network descriptors.

## 10. How to extend this search (what a referee will do)

1. Google Scholar and dblp, with and without quotes: "time-dependent spanner", "spanner for multiple weight functions", "multi-criteria spanner", "robust spanner uncertain edge weights", "distance-preserving sparsification road network", "highway dimension spanner", "road network skeleton", "t-spanner OpenStreetMap".
2. Forward citations ("cited by") of Filtser and Solomon (2016/2020), Abam et al. (2009), Brandes and Handke (1998), and Bilo et al. (2022).
3. Route-planning venues (ESA, ALENEX, SEA, SIGSPATIAL): time-dependent contraction hierarchies and travel-time sparsification are solutions to a related problem (fast queries) with no stretch guarantee on a subgraph.
4. Ask the advisors: they will know the geometric-spanner literature, including the greedy region-fault construction of Dr. Farshi's group.
