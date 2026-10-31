# Hostile-referee audit (status as of this commit)

Severity: **A** = can sink the paper, **B** = will draw a major-revision comment, **C** = polish / artifact review.
Status: **fixed** (in repo), **partial**, **open** (needs data/runs or author decision).

| # | Sev | Objection a referee will raise | Evidence in this repo | Status / action |
|---|---|---|---|---|
| 1 | A | "Time-varying" is claimed, but Theorem 1 certifies only the 24 grid hours. | `benchmarks/offgrid_guarantee.py` and random instances: grid-certified spanners exceed `t` slightly at intermediate times (up to about 0.5% above `t`). | **fixed (method)**: Theorem 3 + `offgrid_guarantee.py` give a provable continuous-time bound by certifying at `t/rho^2` on a 15-min grid (costs about 8-10 more percentage points of edges on random graphs). Run it on METR-LA/PEMS-BAY and the OSM cities and add the numbers. |
| 2 | A | Guarantee is for snapshots, not for time-dependent traversal (path-dependent arrival times). | `fifo_violation_rate` and `time_dependent_validation` are empirical only. | **partial**: stated in README limitations; no theorem. Either restrict the title/abstract to "snapshot" or prove a TD bound under FIFO. |
| 3 | A | The "backbone" barely sparsifies (93-99% of edges kept at t = 1.5). | `retention_diagnostics`: retention equals the forced-edge lower bound at t <= 1.5. | **partial**: Theorem 2 explains it (structural, algorithm-independent). The paper must be framed around this finding, not around sparsification gains. |
| 4 | A | Nearly-as-good trivial baseline: union of per-hour static greedy spanners is also certified. | METR-LA/PEMS-BAY: joint construction saves 1.4-3.8 points of edges. | **partial**: baseline added to the benchmarks. The novelty claim must be stated as a modest, measured gain; it is not yet in the main cross-city table. |
| 5 | A | No size/approximation guarantee; joint greedy is not provably <= union baseline. | In 6 of 236 random instances the joint greedy kept 1-2 more edges than the union. | **fixed (claim)**: README no longer says "equal or fewer". Open: any size bound. |
| 6 | B | Advantage over the matched-density control is not consistent. | `results/crosscity_table.md`: ratio below 1 in 8 of 12 cities. | **fixed (claim)** in limitations; do not state superiority on OD dilation. |
| 7 | B | "Metropolitan" claim vs. 1.5 km city cores. | `CITY_REGISTRY` uses `dist = 1500` for every city. | **fixed (wording)**. Open: run at least 3 cities at about 5 km (edit `dist`; no CLI flag exists). |
| 8 | B | Tiny OD samples (7 x 7 pairs, 3 repeats per city) behind pair-level tests. | Defaults in `run_spanner_benchmark.py`. | **open**: rerun with `--city-sources 30` and more repeats for the submission run. |
| 9 | B | Congestion model validated only on two US freeway sensor networks; severity overstated (1.50x vs 1.34-1.38x). | METR-LA r = 0.86, PEMS-BAY r = 0.94. | **partial**: README updated. Open: run `tools/validate_congestion_shape.py` (NYC DOT) now that the network is reachable; calibrate severity or report sensitivity (already ±30%). |
| 10 | B | Real-OSM results are not reproducible by a reviewer (graph pickles untracked; CI runs synthetic only). | `results/graph_cache/*.pkl` are git-ignored. | **open**: archive the graph cache with a DOI (Zenodo), cite the OSM snapshot date in the paper. |
| 11 | B | Unpinned dependencies (`>=` only), no lock file. | `requirements.txt`. | **open**: `pip freeze > requirements-lock.txt` from the environment used for the reported numbers. |
| 12 | C | OSM licence/attribution missing. | README had none. | **fixed** (Data and attribution section). |
| 13 | C | README says numbers are never hand-copied, but sections 5b-5d quote numbers. | Contradiction. | **fixed**: wording + `tests/test_offgrid_and_readme.py` checks 5d against its CSV. |
| 14 | C | Authorship inconsistency: `LICENSE` says "Soheil", `CITATION.cff` says "Fallahzadeh" with no given name, no ORCID, advisor absent. | Files. | **open (author decision)**: harmonize full names, add ORCID, decide whether the advisor is listed. |
| 15 | C | Promotional tone (emoji in diagrams, "most retractions come from skipping") in a research artifact. | README sections 3 and 5. | **open**: neutralize before artifact review. |
| 16 | C | Related-work references with unconfirmed venues. | `docs/LITERATURE_VERIFIED.md`, rows marked C. | **open**: confirm each against the publisher page. |

Larger-scale commands for the open items:

```bash
python run_spanner_benchmark.py --city-sources 30 --repeats 5          # item 8 (check --help for the targets flag)
python benchmarks/offgrid_guarantee.py --dataset metrla  --speeds ~/data/METR-LA.csv  --T 1.2 1.5 2.0   # item 1
python benchmarks/offgrid_guarantee.py --dataset pemsbay --speeds ~/data/PEMS-BAY.csv --T 1.2 1.5 2.0
python tools/validate_congestion_shape.py --help                        # item 9
pip freeze > requirements-lock.txt                                      # item 11
```
