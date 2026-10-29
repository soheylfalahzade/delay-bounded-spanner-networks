# Benchmark-bearing literature (peer-reviewed venues only)

Status column: **V** = venue confirmed against a publisher / conference / institutional record during this review;
**C** = venue must still be confirmed against the publisher page before the reference goes into the manuscript.

| # | Reference | Venue | Status | What it contributes as a benchmark / protocol | Implemented in |
|---|---|---|---|---|---|
| 1 | M. Farshi, J. Gudmundsson. *Experimental Study of Geometric t-Spanners.* | ESA 2005, LNCS 3669, pp. 556-567 | V | Point-set families; metrics: #edges, weight, max degree, diameter, crossings | `static` |
| 2 | M. Farshi, J. Gudmundsson. *Experimental Study of Geometric t-Spanners: A Running Time Comparison.* | WEA 2007, LNCS 4525, pp. 270-284 | V | Running-time protocol for Greedy / Yao / Theta / WSPD etc. | `static` (timing column) |
| 3 | Bounded-Degree Plane Geometric Spanners in Practice | ACM J. Experimental Algorithmics | V (journal) / C (year, volume, article no.) | Stretch-factor estimation as the bottleneck; lightness, degree | `static` |
| 4 | G.V. Batz, R. Geisberger, P. Sanders, C. Vetter. *Minimum Time-Dependent Travel Times with Contraction Hierarchies.* | ACM J. Exp. Algorithmics 18, Art. 1.4, 2013 | V | Time-dependent road-network protocol (German network, realistic profiles), earliest-arrival queries | `metrla`, `theorem1` (TD baseline reasoning) |
| 5 | D. Delling, A.V. Goldberg, T. Pajor, R.F. Werneck. *Customizable Route Planning in Road Networks.* | Transportation Science 51(2):566-591, 2017 | V | Metric-independent preprocessing; real-time traffic metrics | related-work positioning |
| 6 | Y. Li, R. Yu, C. Shahabi, Y. Liu. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* | ICLR 2018 | V | METR-LA (207 sensors, Los Angeles) and PEMS-BAY (325 sensors): measured 5-min speeds + directed road-distance sensor graph | `metrla` |
| 7 | A. Casteigts, J.G. Peters, J. Schoeters. *Temporal Cliques Admit Sparse Spanners.* | ICALP 2019 (LIPIcs 134); JCSS 119:60-77, 2021 | V (ICALP) / C (JCSS pages) | Temporal-graph spanners (journey-based model) - different model, must be contrasted | related-work positioning |
| 8 | Angrick et al. *Spanner Approximations in Practice* | arXiv:2107.02018 | C (publication venue) | General-graph spanner experiments (Greedy ADDJS vs Baswana-Sen vs others) | `general` |
| 9 | I. Althofer, G. Das, D. Dobkin, D. Joseph, J. Soares. *On Sparse Spanners of Weighted Graphs.* | Discrete & Computational Geometry 9:81-100, 1993 | already in README references | Path-greedy spanner; reference for the static reduction | `static`, `general` |
| 10 | S. Baswana, S. Sen. *A Simple and Linear Time Randomized Algorithm for Computing Sparse Spanners in Weighted Graphs.* | Random Structures & Algorithms 30(4):532-563, 2007 | C | Standard randomized (2k-1)-spanner baseline | `general` |
| 11 | DIMACS 9th Implementation Challenge - Shortest Paths (USA road networks) | challenge data set | C (cite per challenge page) | Standard static road-network instances | `dimacs` |

## Benchmarks that now exist in the repo

| Check | Command | Pass criterion |
|---|---|---|
| Static reduction to classical greedy (edge-for-edge vs independent reference), Yao/Theta bounds | `python benchmarks/literature_benchmarks.py static` | 0 mismatches, stretch <= t |
| General graphs vs Baswana-Sen | `python benchmarks/literature_benchmarks.py general` | stretch <= 2k-1, equality with reference greedy |
| Exhaustive all-pairs/all-hours Theorem 1 + negative controls + certified union-of-snapshots baseline | `python benchmarks/literature_benchmarks.py theorem1` | 0 soundness violations |
| Real telemetry (METR-LA) | `python benchmarks/literature_benchmarks.py metrla --speeds METR-LA.csv` | table + congestion-shape correlation |
| DIMACS static road network | `python benchmarks/literature_benchmarks.py dimacs --gr USA-road-d.NY.gr.gz` | table |

Data for `metrla`: https://zenodo.org/records/5724362 (METR-LA.csv). Sensor graph is fetched automatically from github.com/liyaguang/DCRNN.
Data for `dimacs`: DIMACS-9 challenge site (USA-road-d.*.gr.gz).
